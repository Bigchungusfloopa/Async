"""Guards against the three visual defects seen in the first real run of the merged app:
raw translation keys on screen, header HTML rendered as a code block, unreadable multiselect chips."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml import i18n  # noqa: E402

APP = ROOT / "app"


def _literal_keys() -> dict[str, set[str]]:
    keys: dict[str, set[str]] = {}
    for p in APP.rglob("*.py"):
        for m in re.finditer(r"""\bt\(\s*["']([A-Za-z_0-9]+)["']""", p.read_text(encoding="utf-8")):
            keys.setdefault(m.group(1), set()).add(p.name)
    return keys


def test_every_translation_key_used_by_the_app_exists_in_all_languages():
    used = _literal_keys()
    assert len(used) > 30                                              # the scan really found the app's keys
    problems = {}
    for key, files in used.items():
        absent = [lang for lang in ("en", "hi", "mr") if key not in i18n.STRINGS[lang]]
        if absent:
            problems[key] = (absent, sorted(files))
    assert not problems, f"keys that would render as raw text: {problems}"


def test_ui_helpers_emit_html_without_blank_lines(monkeypatch):
    """A blank line inside an HTML block makes Markdown render the rest as a code block."""
    from app import ui

    captured: list[str] = []
    monkeypatch.setattr(ui.st, "markdown", lambda body, *a, **k: captured.append(body))
    ui.card_header("Risk label", right="MODEL CONF: 69%")             # no subtitle: the original trigger
    ui.card_header("Risk label", subtitle="sub", right="x")
    ui.card_header("Only title")
    ui.section("Title")                                               # no subtitle
    ui.section("Title", "subtitle")
    assert len(captured) == 5
    for html in captured:
        assert "\n" not in html, html
        assert not re.search(r"^\s{4,}<", html, re.M)


def test_multiselect_chip_css_matches_span_tags():
    css = (APP / "style.css").read_text(encoding="utf-8")
    assert re.search(r'(?<![a-z])\[data-baseweb="tag"\]\s*\{', css)    # any element, not only div
    assert 'div[data-baseweb="tag"] {' not in css


def test_current_streamlit_chip_markup_is_styled():
    css = (APP / "style.css").read_text(encoding="utf-8")
    assert '[data-testid="stMultiSelectTagsContainer"] [data-tag]' in css


def test_ticker_tape_never_invents_quotes():
    from app.components import ticker_tape
    import inspect
    src = inspect.getsource(ticker_tape.render_ticker_tape)
    assert "4210.00" not in src and "2980.50" not in src               # the old hard-coded demo prices


def test_title_and_offline_sync():
    main = (APP / "main.py").read_text(encoding="utf-8")
    assert 'page_title="Async"' in main
    assert 'set_offline(st.session_state["offline"])' in main            # backend re-synced every run
    assert "REFRESH DATA" in main and "st.cache_data.clear()" in main


def test_no_hard_coded_figures_left_in_the_tabs():
    tabs = {p.name: p.read_text(encoding="utf-8") for p in (APP / "tabs").glob("*.py")}
    banned = ["24.5, -11.2", "0.125", '"-8.4%"', "+32.1%", "var_95=-0.022", "rng.uniform(-8, 9",
              "{t: 0.35 for t in tickers}", "beats Q3 margin guidance", "* 1.045", "* 1.05 for h"]
    found = {b: [n for n, src in tabs.items() if b in src] for b in banned}
    assert not any(found.values()), found


def test_sidebar_collapse_control_is_hidden():
    css = (APP / "style.css").read_text(encoding="utf-8")
    assert '[data-testid="stSidebarCollapseButton"]' in css and '[data-testid="stExpandSidebarButton"]' in css


def test_sidebar_is_forced_open_even_if_a_browser_stored_it_collapsed():
    css = (APP / "style.css").read_text(encoding="utf-8")
    assert 'section[data-testid="stSidebar"][aria-expanded="false"]' in css and "transform: none !important" in css
    assert "stSidebarCollapsed" in (APP / "main.py").read_text(encoding="utf-8")   # stale localStorage flag cleared
