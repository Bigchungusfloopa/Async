"""Whole-app smoke test: run app/main.py headlessly (Streamlit AppTest) against the REAL backend, offline.

Prices come from the committed fallback parquet files; LLM calls are disabled (Offline mode); no Supabase.
Fails on any uncaught exception or any tab-level "module error".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from engine import data, db  # noqa: E402
from ml import llm  # noqa: E402


@pytest.fixture(autouse=True)
def offline_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "prices")
    monkeypatch.setattr(data, "SYNC_TO_SUPABASE", False)
    monkeypatch.setattr(data, "_sleep", lambda s: None)
    monkeypatch.setattr(data, "_download", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    monkeypatch.setattr(db, "LOCAL_DIR", tmp_path / "local")
    monkeypatch.setattr(db, "_read_secrets", lambda: (None, None))
    monkeypatch.setattr(db, "_client", None)
    data._tail_tried.clear()
    data._head_tried.clear()
    llm.set_offline(True)
    st.cache_data.clear()
    yield
    llm.set_offline(False)
    st.cache_data.clear()


def _run(lang: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=180)
    at.secrets["USE_MOCKS"] = False
    at.session_state["lang"] = lang
    at.session_state["offline"] = True
    at.run()
    return at


@pytest.mark.parametrize("lang", ["en", "hi", "mr"])
def test_every_tab_renders_with_the_real_backend(lang):
    at = _run(lang)
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    assert not at.error, [str(e.value)[:200] for e in at.error]       # includes "<tab> module error" messages
    assert len(at.tabs) == 7


def test_changing_risk_level_and_scenario_does_not_break_the_app():
    at = _run("en")
    at.radio[0].set_value("High").run()
    assert not at.exception and not at.error
    scenario_box = next(sb for sb in at.selectbox if "SCENARIO" in sb.label.upper())
    for option in list(scenario_box.options)[:4]:
        scenario_box.set_value(option).run()
        assert not at.exception and not at.error, option
