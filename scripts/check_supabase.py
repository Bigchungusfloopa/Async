"""Live round-trip check of every Supabase path the engine uses. Writes clearly-marked test rows and removes
them afterwards.   python scripts/check_supabase.py
"""
from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import db  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


h = db.health()
check("configured + reachable + schema", all(h.values()), str(h))
if not all(h.values()):
    raise SystemExit(1)
c = db.get_client()
tag = "__check_" + uuid.uuid4().hex[:6]

try:
    # price_cache: > 1000 rows exercises pagination
    idx = pd.bdate_range("2015-01-01", periods=1300)
    frame = pd.DataFrame({tag + "_A": np.linspace(10, 20, 1300), tag + "_B": np.linspace(5, 6, 1300)}, index=idx)
    n = db.upsert_prices(frame)
    check("price_cache upsert (2600 rows, chunked)", n == 2600, f"wrote {n}")
    back = db.read_price_cache([tag + "_A", tag + "_B"], idx[0], idx[-1])
    check("price_cache read + pagination", back is not None and back.shape == (1300, 2)
          and np.allclose(back[tag + "_A"].to_numpy(), frame[tag + "_A"].to_numpy()), f"shape={None if back is None else back.shape}")
    check("price_cache upsert is idempotent", db.upsert_prices(frame.iloc[:5]) == 10)
    part = db.read_price_cache([tag + "_A"], idx[10], idx[19])
    check("price_cache date filter", part is not None and len(part) == 10)

    # llm_cache: remote write, then a fresh local mirror must read it back from Supabase
    key = tag + "_k"
    db.llm_cache_put(key, "news", "mr", "नमस्कार", "gemini")
    local_file = db._local_path("llm_cache.json")
    d = db._read_local("llm_cache.json", {}); d.pop(key, None); db._write_local("llm_cache.json", d)
    got = db.llm_cache_get(key)
    check("llm_cache remote round trip (unicode)", got is not None and got["response"] == "नमस्कार" and got["provider"] == "gemini")

    # portfolios + holdings
    pid = db.save_portfolio(tag, tag + "_sess", "Medium", "hi",
                            [{"ticker": "tcs.ns", "quantity": 10, "buy_price": 3200},
                             {"ticker": "AAPL", "quantity": 4, "buy_price": 150}])
    remote_row = c.table("portfolios").select("id").eq("id", pid).execute().data
    check("save_portfolio went to Supabase", len(remote_row) == 1, f"id={pid}")
    local_store = db._read_local("local_portfolios.json", {}); local_store.pop(pid, None)
    listed = db.list_portfolios(tag + "_sess")
    check("list_portfolios", [p["id"] for p in listed] == [pid] and listed[0]["language"] == "hi")
    loaded = db.load_portfolio(pid)
    by_t = {x["ticker"]: x for x in loaded["holdings"]} if loaded else {}
    check("load_portfolio (buy_price rebuilt)", set(by_t) == {"TCS.NS", "AAPL"} and abs(by_t["TCS.NS"]["buy_price"] - 3200) < 1e-9
          and abs(sum(x["weight"] for x in loaded["holdings"]) - 1) < 1e-9)
    check("save_run", db.save_run(pid, "Medium", {"sharpe": 1.2}, {"TCS.NS": 1.0}) is True)
    runs = c.table("analysis_runs").select("risk_label,metrics").eq("portfolio_id", pid).execute().data
    check("analysis_runs row content", len(runs) == 1 and runs[0]["metrics"]["sharpe"] == 1.2)

    # Storage
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / (tag + ".bin"); f.write_bytes(b"model-bytes-" + tag.encode())
        up = db.upload_model(f)
        check("upload_model (models bucket)", up)
        up2 = db.upload_model(f)
        check("upload_model overwrite (upsert)", up2)
        out = db.download_model(f.name, dest_dir=Path(td) / "dl")
        check("download_model", out is not None and out.read_bytes() == f.read_bytes())
        db._remote(lambda cl: cl.storage.from_("models").remove([f.name]), "cleanup")
finally:
    # cleanup: remove every test row
    try:
        for t in (tag + "_A", tag + "_B"):
            c.table("price_cache").delete().eq("ticker", t).execute()
        c.table("llm_cache").delete().eq("key", tag + "_k").execute()
        c.table("portfolios").delete().like("name", tag + "%").execute()   # holdings/runs cascade
    except Exception as exc:  # noqa: BLE001
        print("cleanup problem:", type(exc).__name__, str(exc)[:120])
    for fname in ("local_portfolios.json", "llm_cache.json", "local_runs.json"):
        try:
            db._local_path(fname).unlink()
        except OSError:
            pass

bad = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(bad)}/{len(results)} passed" + (f"; FAILED: {bad}" if bad else ""))
raise SystemExit(1 if bad else 0)
