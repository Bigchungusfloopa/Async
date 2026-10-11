"""Headless smoke test of the Streamlit app (no browser): run the whole script and report anything that raised
or warned. Usage:  python scripts/smoke_app.py [--real] [--lang hi] [--offline]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def run(real: bool, lang: str, offline: bool, timeout: int = 180) -> int:
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=timeout)
    at.secrets["USE_MOCKS"] = not real
    at.session_state["lang"] = lang
    at.session_state["offline"] = offline
    at.run()
    exc = [e.value if hasattr(e, "value") else str(e) for e in at.exception]
    errs = [e.value for e in at.error]
    warns = [w.value for w in at.warning]
    print(f"mode={'REAL' if real else 'MOCK'} lang={lang} offline={offline}: "
          f"{len(at.tabs)} tabs, {len(at.metric)} metrics, {len(exc)} exceptions, {len(errs)} errors, {len(warns)} warnings")
    for label, items in (("EXCEPTION", exc), ("ERROR", errs), ("WARNING", warns)):
        for it in items:
            print(f"  [{label}] {str(it)[:260]}")
    return 1 if exc else 0




def interact(real: bool = True) -> int:
    """Cycle every select box (scenarios, regime presets, benchmark...) and the risk radio; report failures."""
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=240)
    at.secrets["USE_MOCKS"] = not real
    at.session_state["offline"] = True            # no network LLM calls while clicking around
    at.run()
    bad = 0

    def check(label: str) -> None:
        nonlocal bad
        problems = [str(e.value)[:160] for e in at.exception] + [str(e.value)[:160] for e in at.error]
        status = "ok " if not problems else "FAIL"
        bad += bool(problems)
        print(f"  [{status}] {label}" + ("" if not problems else f" -> {problems[0]}"))

    check("initial render")
    for i in range(len(at.selectbox)):
        sb = at.selectbox[i]
        for opt in list(sb.options)[:8]:
            try:
                at.selectbox[i].set_value(opt).run()
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] selectbox {i} -> {opt!r}: {type(exc).__name__}: {str(exc)[:100]}")
                bad += 1
                continue
            check(f"selectbox[{i}] ({sb.label[:28]!r}) = {str(opt)[:34]!r}")
    for r in range(len(at.radio)):
        for opt in list(at.radio[r].options):
            at.radio[r].set_value(opt).run()
            check(f"radio[{r}] = {opt!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use the real engine/ml backend (USE_MOCKS=false)")
    ap.add_argument("--lang", default="en", choices=["en", "hi", "mr"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--interact", action="store_true", help="also cycle every selectbox / radio option")
    a = ap.parse_args()
    rc = run(a.real, a.lang, a.offline)
    if a.interact:
        rc |= interact(a.real)
    raise SystemExit(rc)
