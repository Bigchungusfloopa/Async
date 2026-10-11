"""Supabase client + DB/Storage helpers.

Design rules
* **Never raise to the UI.** Every public function catches everything, logs it, and returns an empty /
  local-file result instead.
* **One request at a time.** supabase-py shares a single HTTP/2 connection; concurrent use from several
  Streamlit threads (or our background price sync) corrupts it. All remote calls go through ``_remote``,
  which serialises them.
* **Circuit breaker.** After a *network* failure Supabase is skipped for ``COOLDOWN_S`` seconds so a dead
  connection costs one timeout, not one per call. API errors (missing table, RLS) do not trip it.
* **Local fallback.** Portfolios, runs and the LLM cache are mirrored to JSON files in ``data/cache`` so
  the app keeps working with no Supabase at all (and in offline demos).
* No Streamlit imports. Credentials come from env vars ``SUPABASE_URL`` / ``SUPABASE_KEY`` or from
  ``.streamlit/secrets.toml`` (read with ``tomllib``); nothing is ever hard-coded or logged.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

log = logging.getLogger("engine.db")

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = ROOT / "data" / "cache"
COOLDOWN_S = 60
CHUNK = 1000          # rows per upsert request, and the PostgREST page size
MODELS_BUCKET = "models"
REPORTS_BUCKET = "reports"
_SCHEMA_MISSING_CODES = {"PGRST205", "42P01", "PGRST200"}

_lock = threading.Lock()          # guards client creation
_io_lock = threading.RLock()      # one Supabase request at a time (shared HTTP/2 connection)
_client: Any = None
_client_failed_at = 0.0
_disabled_until = 0.0


# --------------------------------------------------------------------------- configuration / client
def _read_secrets() -> tuple[str | None, str | None]:
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if url and key:
        return url, key
    path = ROOT / ".streamlit" / "secrets.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return url or data.get("SUPABASE_URL") or None, key or data.get("SUPABASE_KEY") or None
    except (OSError, ValueError):
        return url, key


def is_configured() -> bool:
    url, key = _read_secrets()
    return bool(url and key)


def get_client():
    """Return a cached Supabase client, or ``None`` when unconfigured / unreachable / cooling down."""
    global _client, _client_failed_at
    if time.time() < _disabled_until:
        return None
    if _client is not None:
        return _client
    if time.time() - _client_failed_at < COOLDOWN_S:
        return None
    url, key = _read_secrets()
    if not (url and key):
        return None
    with _lock:
        if _client is not None:
            return _client
        try:
            from supabase import ClientOptions, create_client

            _client = create_client(url, key, options=ClientOptions(
                postgrest_client_timeout=10, storage_client_timeout=20,
                auto_refresh_token=False, persist_session=False))
        except Exception as exc:  # noqa: BLE001 - must never raise to the UI
            log.warning("Supabase client unavailable: %s", type(exc).__name__)
            _client_failed_at = time.time()
            _client = None
    return _client


def _is_transport_error(exc: Exception) -> bool:
    import httpx

    return isinstance(exc, (httpx.TransportError, OSError, TimeoutError))


def _trip(exc: Exception) -> None:
    """Open the circuit breaker on transport-level failures (not on API / schema errors)."""
    global _disabled_until
    if _is_transport_error(exc):
        _disabled_until = time.time() + COOLDOWN_S
        log.warning("Supabase unreachable (%s); using local fallbacks for %ss", type(exc).__name__, COOLDOWN_S)


def reset_circuit() -> None:
    """Clear the cooldown (e.g. after the user fixed their connection)."""
    global _disabled_until, _client_failed_at
    _disabled_until = 0.0
    _client_failed_at = 0.0


def _remote(fn: Callable[[Any], Any], what: str = "request") -> tuple[bool, Any]:
    """Run ``fn(client)`` serialised and exception-safe. Returns ``(ok, result)``; never raises."""
    client = get_client()
    if client is None:
        return False, None
    try:
        with _io_lock:
            return True, fn(client)
    except Exception as exc:  # noqa: BLE001
        _trip(exc)
        code = getattr(exc, "code", None)
        log.warning("supabase %s failed: %s%s", what, type(exc).__name__, f" ({code})" if code else "")
        return False, exc


def health() -> dict:
    """``{"configured", "reachable", "schema_ready"}`` for the UI status badge. Never raises.

    ``reachable`` means the server answered (even with an API error); ``schema_ready`` means the
    tables from ``sql/schema.sql`` exist and are readable with the configured key."""
    out = {"configured": is_configured(), "reachable": False, "schema_ready": False}
    if get_client() is None:
        return out
    ok, res = _remote(lambda c: c.table("llm_cache").select("key").limit(1).execute(), "health")
    if ok:
        out.update(reachable=True, schema_ready=True)
    elif isinstance(res, Exception) and not _is_transport_error(res):
        out["reachable"] = True            # server answered: schema missing / permission problem
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- local JSON store
def _local_path(name: str) -> Path:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_DIR / name


def _read_local(name: str, default):
    try:
        return json.loads(_local_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_local(name: str, data) -> None:
    try:
        path = _local_path(name)
        tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic
    except OSError as exc:
        log.warning("could not write local store %s: %s", name, exc)


# --------------------------------------------------------------------------- price_cache (Supabase tier)
def read_price_cache(tickers: Iterable[str], start, end) -> pd.DataFrame | None:
    """Cached adjusted closes for ``[start, end]`` as a wide frame (date index, ticker columns).
    Returns ``None`` when Supabase is unavailable and an empty frame when it has nothing."""
    tickers = list(tickers)
    if not tickers or get_client() is None:
        return None
    s, e = pd.Timestamp(start).date().isoformat(), pd.Timestamp(end).date().isoformat()

    def fetch(client) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:  # PostgREST caps responses at 1000 rows -> paginate one ordered query
            res = (client.table("price_cache").select("ticker,date,adj_close").in_("ticker", tickers)
                   .gte("date", s).lte("date", e).order("ticker").order("date")
                   .range(offset, offset + CHUNK - 1).execute())
            rows += res.data or []
            if len(res.data or []) < CHUNK:
                return rows
            offset += CHUNK

    ok, rows = _remote(fetch, "price_cache read")
    if not ok:
        return None
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    long["date"] = pd.to_datetime(long["date"])
    return long.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def upsert_prices(prices: pd.DataFrame) -> int:
    """Upsert a wide price frame into ``price_cache`` in chunks of ~1000 rows. Returns rows written."""
    if prices is None or prices.empty or get_client() is None:
        return 0
    long = prices.rename_axis("date").reset_index().melt(id_vars="date", var_name="ticker", value_name="adj_close")
    long = long.dropna(subset=["adj_close"])
    rows = [{"ticker": str(r.ticker), "date": pd.Timestamp(r.date).date().isoformat(),
             "adj_close": float(r.adj_close)} for r in long.itertuples(index=False)]
    written = 0
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        ok, _ = _remote(lambda c, ch=chunk: c.table("price_cache").upsert(ch, on_conflict="ticker,date").execute(),
                        "price_cache upsert")
        if not ok:
            break
        written += len(chunk)
    return written


# --------------------------------------------------------------------------- llm_cache (used by P2's llm.py)
def llm_cache_get(key: str) -> dict | None:
    """``{"response","provider","task","lang"}`` for a cache key (sha256 of task+prompt+lang), else None.
    The local mirror is checked first (instant); a remote hit is written through to it."""
    local = _read_local("llm_cache.json", {})
    if key in local:
        return local[key]
    ok, res = _remote(lambda c: c.table("llm_cache").select("response,provider,task,lang")
                      .eq("key", key).limit(1).execute(), "llm_cache get")
    if ok and res.data:
        row = res.data[0]
        local[key] = row
        _write_local("llm_cache.json", local)
        return row
    return None


def llm_cache_put(key: str, task: str, lang: str, response: str, provider: str = "") -> bool:
    """Store a response locally (always) and in Supabase (when reachable). True if stored anywhere."""
    row = {"key": key, "task": task, "lang": lang, "response": response, "provider": provider}
    local = _read_local("llm_cache.json", {})
    local[key] = {k: row[k] for k in ("response", "provider", "task", "lang")}
    _write_local("llm_cache.json", local)
    _remote(lambda c: c.table("llm_cache").upsert(row, on_conflict="key").execute(), "llm_cache put")
    return True


# --------------------------------------------------------------------------- portfolios
def _holding_row(pid: str, h: dict, total_amount: float) -> dict:
    qty = float(h.get("quantity") or 0)
    buy = float(h.get("buy_price") or 0)
    amount = float(h.get("amount") or (qty * buy))
    weight = h.get("weight")
    if weight is None:
        weight = amount / total_amount if total_amount else 0.0
    return {"portfolio_id": pid, "ticker": str(h["ticker"]).strip().upper(),
            "amount": amount, "quantity": qty, "weight": float(weight)}


def _holding_out(row: dict) -> dict:
    qty = float(row.get("quantity") or 0)
    amount = float(row.get("amount") or 0)
    return {"ticker": row["ticker"], "amount": amount, "quantity": qty,
            "weight": float(row.get("weight") or 0.0),
            # the schema stores invested amount, so the buy price is amount / quantity
            "buy_price": (amount / qty) if qty else None}


def save_portfolio(name: str, session_name: str, risk_level: str, language: str, holdings: list[dict]) -> str:
    """Persist a portfolio and its holdings; returns the portfolio id (Supabase uuid, or a local uuid
    when Supabase is unreachable). Holdings: ``{"ticker", "quantity", "buy_price"}`` and/or
    ``{"amount", "weight"}``. A half-written remote save is rolled back before falling back to local."""
    holdings = [h for h in (holdings or []) if h.get("ticker")]
    total = sum(float(h.get("amount") or float(h.get("quantity") or 0) * float(h.get("buy_price") or 0))
                for h in holdings)
    meta = {"session_name": session_name, "name": name, "risk_level": risk_level, "language": language or "en"}

    def remote_save(c) -> str:
        pid = c.table("portfolios").insert(meta).execute().data[0]["id"]
        try:
            rows = [_holding_row(pid, h, total) for h in holdings]
            if rows:
                c.table("holdings").insert(rows).execute()
        except Exception:
            try:
                c.table("portfolios").delete().eq("id", pid).execute()   # no orphan portfolio rows
            except Exception:  # noqa: BLE001
                pass
            raise
        return pid

    ok, pid = _remote(remote_save, "save_portfolio")
    if ok:
        return pid
    pid = str(uuid.uuid4())
    store = _read_local("local_portfolios.json", {})
    store[pid] = {"id": pid, **meta, "created_at": _now(),
                  "holdings": [_holding_out(_holding_row(pid, h, total)) for h in holdings]}
    _write_local("local_portfolios.json", store)
    return pid


def list_portfolios(session_name: str) -> list[dict]:
    """Saved portfolios for a session (newest first): ``{"id","name","risk_level","language","created_at"}``."""
    out: dict[str, dict] = {}
    ok, res = _remote(lambda c: c.table("portfolios").select("id,name,session_name,risk_level,language,created_at")
                      .eq("session_name", session_name).order("created_at", desc=True).execute(), "list_portfolios")
    if ok:
        out.update({r["id"]: r for r in res.data or []})
    for pid, p in _read_local("local_portfolios.json", {}).items():
        if p.get("session_name") == session_name:
            out.setdefault(pid, {k: p.get(k) for k in ("id", "name", "session_name", "risk_level", "language", "created_at")})
    return sorted(out.values(), key=lambda r: r.get("created_at") or "", reverse=True)


def load_portfolio(portfolio_id: str) -> dict | None:
    """Full portfolio incl. ``holdings`` (each with ticker, amount, quantity, weight, buy_price) or None."""
    local = _read_local("local_portfolios.json", {}).get(portfolio_id)
    if local:
        return local

    def fetch(c):
        p = c.table("portfolios").select("*").eq("id", portfolio_id).limit(1).execute().data
        if not p:
            return None
        rows = c.table("holdings").select("ticker,amount,quantity,weight").eq("portfolio_id", portfolio_id).execute().data
        return {**p[0], "holdings": [_holding_out(r) for r in rows or []]}

    ok, res = _remote(fetch, "load_portfolio")
    return res if ok else None


def save_run(portfolio_id: str, label: str, metrics: dict, weights: dict) -> bool:
    """Store an analysis run (risk label + metrics + weights). True if it reached Supabase, False if it
    was kept in the local mirror only."""
    from .serialize import to_jsonable

    row = {"portfolio_id": portfolio_id, "risk_label": label,
           "metrics": to_jsonable(metrics), "weights": to_jsonable(weights)}
    ok, _ = _remote(lambda c: c.table("analysis_runs").insert(row).execute(), "save_run")
    if ok:
        return True
    runs = _read_local("local_runs.json", [])
    runs.append({**row, "created_at": _now()})
    _write_local("local_runs.json", runs)
    return False


# --------------------------------------------------------------------------- Storage (models / reports)
def upload_model(path: str | Path, bucket: str = MODELS_BUCKET) -> bool:
    """Upload a local file (e.g. ``risk_model.joblib``) to the private ``models`` bucket (overwrites)."""
    p = Path(path)
    if not p.is_file():
        return False
    data = p.read_bytes()
    ok, _ = _remote(lambda c: c.storage.from_(bucket).upload(
        p.name, data, {"upsert": "true", "content-type": "application/octet-stream"}), "upload_model")
    return ok


def download_model(name: str, dest_dir: str | Path | None = None, bucket: str = MODELS_BUCKET) -> Path | None:
    """Download ``name`` from the bucket into ``dest_dir`` (default ``data/cache/models``).
    Returns the local path (an existing local copy is returned if Supabase is unavailable), else None."""
    dest = Path(dest_dir) if dest_dir else LOCAL_DIR / "models"
    out = dest / Path(name).name
    ok, data = _remote(lambda c: c.storage.from_(bucket).download(name), "download_model")
    if ok and data:
        try:
            dest.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            return out
        except OSError as exc:
            log.warning("could not save model locally: %s", exc)
    return out if out.is_file() else None
