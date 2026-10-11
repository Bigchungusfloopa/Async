"""LLM provider chain with cache, cooldowns, usage counters and an Offline mode.

Chain: Gemini -> Groq -> Mistral (Mistral is last for non-English). Plain ``httpx`` REST calls, so there is no
SDK version churn and no extra dependency. No Streamlit imports: state is module-level; P3 reads
:func:`usage_stats` and calls :func:`set_offline`.

Rules (shared plan section 9)
* cache first: in-process dict, then ``engine.db`` llm_cache (local mirror, then Supabase);
  key = ``sha256(task + prompt + lang)``, independent of provider;
* on 429 / 5xx / timeout: retry once with backoff, put the provider in cooldown (~60 s), move on;
* 401/403 -> that key is bad, rotate to the provider's next key; 404 -> try the next model name;
* everything failed -> :class:`LLMUnavailable` (callers fall back to templates / cached output);
* Offline mode: cache only, zero network calls.

Free-tier model names change; they are configurable (``GEMINI_MODEL`` etc. in secrets/env, comma separated
fallback lists) and verified by the first successful call, not hard-wired.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import tomllib
from collections import deque
from pathlib import Path
from typing import Any

import httpx

from engine import db

log = logging.getLogger("ml.llm")

ROOT = Path(__file__).resolve().parent.parent
CHAIN = ("gemini", "groq", "mistral")
DEFAULT_MODELS = {
    # Verified against the live API (Oct 2026): the 2.5 family is closed to new users; "-latest" aliases keep
    # working across releases. Order = preference; a failing model falls through to the next one.
    "gemini": ["gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"],
    # Verified live (Oct 2026): llama-3.3-70b-versatile is gone from Groq; gpt-oss-120b/20b follow instructions
    # (incl. "risk kam" -> Low), qwen3.8-27b mis-read it in a probe, so it is last.
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    "mistral": ["mistral-small-latest"],
}
TIMEOUT_S = 15.0
COOLDOWN_S = 60.0
RETRY_BACKOFF_S = 1.0
MAX_OUTPUT_TOKENS = 1500
GEMINI_THINKING_HEADROOM = 1024     # Gemini 3.x "thinking" tokens are counted against maxOutputTokens
RATE_WINDOW_S = 600.0

_sleep = time.sleep          # patched in tests
_lock = threading.RLock()
_offline = False
_stats: dict[str, dict[str, int]] = {p: {"calls": 0, "cache_hits": 0, "failures": 0} for p in CHAIN}
_cooldown_until: dict[str, float] = {p: 0.0 for p in CHAIN}
_rate_limited_at: dict[str, deque] = {p: deque() for p in CHAIN}
_calls_at: dict[str, deque] = {p: deque() for p in CHAIN}
_mem_cache: dict[str, dict] = {}
_config: dict[str, Any] | None = None


class LLMUnavailable(RuntimeError):
    """No provider could answer (offline, no keys, quota, outage). Callers must fall back gracefully."""


# --------------------------------------------------------------------------- config
def _secret_sources() -> dict:
    try:
        return tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("["):
            try:
                v = json.loads(v)
            except ValueError:
                v = [v]
        else:
            v = [x for x in re.split(r"[,\s]+", v) if x]
    return [str(x).strip() for x in v if str(x).strip()]


def _load_config() -> dict:
    global _config
    if _config is None:
        secrets = _secret_sources()
        cfg: dict[str, Any] = {"keys": {}, "models": {}}
        for p in CHAIN:
            name = p.upper()
            cfg["keys"][p] = _as_list(os.environ.get(f"{name}_API_KEYS") or secrets.get(f"{name}_API_KEYS"))
            override = _as_list(os.environ.get(f"{name}_MODEL") or secrets.get(f"{name}_MODEL"))
            cfg["models"][p] = override or list(DEFAULT_MODELS[p])
        cfg["whisper_models"] = _as_list(os.environ.get("WHISPER_MODEL") or secrets.get("WHISPER_MODEL")) \
            or ["whisper-large-v3-turbo", "whisper-large-v3"]
        _config = cfg
    return _config


def reload_config() -> None:
    """Forget cached keys/model names (after editing secrets at runtime)."""
    global _config
    _config = None


def get_keys(provider: str) -> list[str]:
    return list(_load_config()["keys"].get(provider, []))


def get_whisper_models() -> list[str]:
    return list(_load_config()["whisper_models"])


def providers_configured() -> list[str]:
    return [p for p in CHAIN if get_keys(p)]


# --------------------------------------------------------------------------- public state API
def set_offline(flag: bool) -> None:
    """Offline mode: serve cache only and make zero network calls."""
    global _offline
    _offline = bool(flag)


def is_offline() -> bool:
    return _offline


def usage_stats() -> dict:
    """``{provider: {"calls", "cache_hits", "failures"}}`` for this process (the sidebar usage panel)."""
    with _lock:
        return {p: dict(s) for p, s in _stats.items()}


def reset_state() -> None:
    """Clear counters, cooldowns and the in-process cache (tests / new demo run)."""
    with _lock:
        for p in list(_stats):
            _stats[p] = {"calls": 0, "cache_hits": 0, "failures": 0}
        for p in CHAIN:
            _cooldown_until[p] = 0.0
            _rate_limited_at[p].clear()
            _calls_at[p].clear()
        _mem_cache.clear()


def cache_key(task: str, prompt: str, lang: str = "en") -> str:
    return hashlib.sha256(f"{task}\n{prompt}\n{lang}".encode("utf-8")).hexdigest()


def parse_json(text: str) -> Any:
    """Parse model output as JSON, tolerating code fences and surrounding prose. Raises ValueError."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found in model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unterminated JSON in model output")


# --------------------------------------------------------------------------- HTTP + providers
class _Transient(Exception):
    """429 / 5xx / timeout / connection problem."""


def _http_post(url: str, headers: dict, body: dict, timeout: float = TIMEOUT_S) -> tuple[int, Any]:
    """POST JSON; returns ``(status, parsed_json_or_text)``. Network errors raise ``_Transient``."""
    try:
        r = httpx.post(url, headers=headers, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise _Transient(type(exc).__name__) from exc
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def _build_request(provider: str, model: str, key: str, prompt: str, system: str, json_mode: bool,
                   max_tokens: int) -> tuple[str, dict, dict]:
    if provider == "gemini":
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens + GEMINI_THINKING_HEADROOM},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        return (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                {"x-goog-api-key": key, "content-type": "application/json"}, body)
    url = {"groq": "https://api.groq.com/openai/v1/chat/completions",
           "mistral": "https://api.mistral.ai/v1/chat/completions"}[provider]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
    if provider == "groq" and model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"                 # reasoning models: keep hidden reasoning short
        body["max_tokens"] = max_tokens + GEMINI_THINKING_HEADROOM
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return url, {"authorization": f"Bearer {key}", "content-type": "application/json"}, body


def _extract_text(provider: str, data: Any) -> str:
    if provider == "gemini":
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    return data["choices"][0]["message"]["content"].strip()


def _provider_order(lang: str) -> list[str]:
    now = time.time()
    with _lock:
        usable = [p for p in CHAIN if get_keys(p) and _cooldown_until[p] <= now]

        def headroom(p: str) -> tuple:
            recent_429 = sum(1 for t in _rate_limited_at[p] if now - t < RATE_WINDOW_S)
            last_min = sum(1 for t in _calls_at[p] if now - t < 60)
            weak = 1 if (p == "mistral" and lang != "en") else 0     # Mistral is weaker on Hindi/Marathi
            return (weak, recent_429, last_min, CHAIN.index(p))

        return sorted(usable, key=headroom)


def _mark(provider: str, field: str) -> None:
    with _lock:
        _stats.setdefault(provider, {"calls": 0, "cache_hits": 0, "failures": 0})[field] += 1


def _cooldown(provider: str) -> None:
    with _lock:
        _cooldown_until[provider] = time.time() + COOLDOWN_S
        _rate_limited_at[provider].append(time.time())


def _try_provider(provider: str, prompt: str, system: str, json_mode: bool, max_tokens: int, errors: list[str]) -> str | None:
    """One provider: keys x models. A transient error (429/5xx/timeout) is retried once, then the NEXT model is
    tried; only when every model failed transiently does the provider cool down. Returns text or None."""
    transient = False
    for key in get_keys(provider):
        bad_key = False
        for model in _load_config()["models"][provider]:
            url, headers, body = _build_request(provider, model, key, prompt, system, json_mode, max_tokens)
            for attempt in range(2):                       # initial try + one retry
                with _lock:
                    _calls_at[provider].append(time.time())
                _mark(provider, "calls")
                try:
                    status, data = _http_post(url, headers, body)
                except _Transient as exc:
                    status, data = 599, str(exc)
                if status == 200:
                    try:
                        return _extract_text(provider, data)
                    except (KeyError, IndexError, TypeError):
                        errors.append(f"{provider}: unexpected response shape")
                        _mark(provider, "failures")
                        return None
                _mark(provider, "failures")
                if status == 429 or status >= 500:
                    errors.append(f"{provider}: HTTP {status}")
                    if attempt == 0:
                        _sleep(RETRY_BACKOFF_S)
                        continue
                    transient = True
                    break                                  # next model name
                if status == 404:                          # unknown model name -> next candidate
                    errors.append(f"{provider}: model {model} not found")
                    break
                if status in (401, 403):                   # bad / unauthorised key -> next key
                    errors.append(f"{provider}: key rejected (HTTP {status})")
                    bad_key = True
                    break
                errors.append(f"{provider}: HTTP {status}")  # 400 etc.: request problem, give up on provider
                return None
            if bad_key:
                break
    if transient:
        _cooldown(provider)
    return None


# --------------------------------------------------------------------------- main entry point
def ask_llm(task: str, prompt: str, system: str = "", json_mode: bool = False, lang: str = "en",
            max_tokens: int | None = None) -> dict:
    """Ask the provider chain (cache first). Returns ``{"text", "provider", "cached"}``.

    Raises :class:`LLMUnavailable` when nothing can answer. Call only from explicit user actions.
    """
    key = cache_key(task, prompt, lang)
    with _lock:
        hit = _mem_cache.get(key)
    if hit is None:
        row = db.llm_cache_get(key, remote=not _offline)
        if row and row.get("response"):
            hit = {"text": row["response"], "provider": row.get("provider") or "cache"}
            with _lock:
                _mem_cache[key] = hit
    if hit is not None:
        _mark(hit["provider"], "cache_hits")
        return {"text": hit["text"], "provider": hit["provider"], "cached": True}

    if _offline:
        raise LLMUnavailable("offline mode is on and there is no cached answer")
    order = _provider_order(lang)
    if not order:
        raise LLMUnavailable("no LLM provider is available (no API keys configured, or all cooling down)")
    if json_mode and "json" not in (prompt + system).lower():
        prompt = prompt + "\n\nRespond with valid JSON only."
    errors: list[str] = []
    for provider in order:
        text = _try_provider(provider, prompt, system, json_mode, max_tokens or MAX_OUTPUT_TOKENS, errors)
        if text is None:
            continue
        if json_mode:
            try:
                parse_json(text)
            except ValueError:
                errors.append(f"{provider}: invalid JSON")
                _mark(provider, "failures")
                continue
        with _lock:
            _mem_cache[key] = {"text": text, "provider": provider}
        db.llm_cache_put(key, task, lang, text, provider, remote=True)
        return {"text": text, "provider": provider, "cached": False}
    raise LLMUnavailable("; ".join(errors) or "no provider answered")
