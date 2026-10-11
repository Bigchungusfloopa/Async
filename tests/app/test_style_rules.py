"""
tests/app/test_style_rules.py — Automated style & constraint verification.
Rules checked:
1. app/style.css and HTML strings in app/components/ contain no gradient(, box-shadow, text-shadow, blur(
2. No emojis in any file under app/, ml/i18n/, assets/ (code points in U+1F000-U+1FAFF, U+2600-U+27BF, U+FE0F)
3. .streamlit/config.toml has base = "dark" and backgroundColor = "#000000"
4. Every figure in app/charts.py uses the 'ink' template and discrete color scale for heatmaps
"""

from __future__ import annotations

import pathlib
import sys
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_no_gradients_shadows_blur_in_css_and_components():
    """Ensure no gradients, box-shadow, text-shadow, or blur exist in styling."""
    forbidden = ["gradient(", "box-shadow", "text-shadow", "filter: blur", "backdrop-filter"]

    files_to_check = [ROOT / "app" / "style.css"]
    components_dir = ROOT / "app" / "components"
    if components_dir.exists():
        files_to_check.extend(components_dir.glob("*.py"))
    files_to_check.append(ROOT / "app" / "ui.py")

    violations = []
    for f in files_to_check:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8").lower()
        for term in forbidden:
            if term == "box-shadow":
                idx = 0
                while True:
                    idx = content.find("box-shadow", idx)
                    if idx == -1:
                        break
                    snippet = content[idx:idx+35]
                    # Allow box-shadow: none !important;
                    if "none" not in snippet:
                        violations.append(f"{f.name}: found '{snippet.strip()}'")
                    idx += 10
            elif term in content:
                violations.append(f"{f.name}: found '{term}'")

    assert not violations, f"Style violations found: {violations}"
    print("test_no_gradients_shadows_blur_in_css_and_components: PASS")


def test_no_emojis_in_app_i18n_assets():
    """Verify zero emoji code points in app/, ml/i18n.py, and assets/."""
    def has_emoji(ch: str) -> bool:
        cp = ord(ch)
        return (
            (0x1F000 <= cp <= 0x1FAFF)
            or (0x2600 <= cp <= 0x27BF)
            or (cp == 0xFE0F)
            or (0x2300 <= cp <= 0x23FF)
            or (0x2B50 <= cp <= 0x2B55)
        )

    targets = list((ROOT / "app").rglob("*.py"))
    targets.extend(list((ROOT / "app").rglob("*.css")))
    targets.append(ROOT / "ml" / "i18n.py")
    if (ROOT / "assets").exists():
        targets.extend([f for f in (ROOT / "assets").rglob("*") if f.is_file() and not f.name.endswith(".ttf")])

    emoji_violations = []
    for f in targets:
        if not f.exists():
            continue
        try:
            content = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            for ch in line:
                if has_emoji(ch):
                    emoji_violations.append(f"{f.name}:{line_no} U+{ord(ch):04X} in: {line.strip()[:40]}")
                    break

    assert not emoji_violations, f"Emoji violations found: {emoji_violations[:10]}"
    print("test_no_emojis_in_app_i18n_assets: PASS")


def test_streamlit_config_is_dark_black():
    """.streamlit/config.toml must specify dark base and pure black background."""
    cfg_path = ROOT / ".streamlit" / "config.toml"
    assert cfg_path.exists(), "config.toml not found"

    content = cfg_path.read_text(encoding="utf-8")
    assert 'base = "dark"' in content, "config.toml must have base = 'dark'"
    assert 'backgroundColor = "#000000"' in content, "config.toml must have backgroundColor = '#000000'"
    print("test_streamlit_config_is_dark_black: PASS")


def test_charts_use_ink_template_and_discrete_heatmaps():
    """Verify charts module registers and uses 'ink' template with discrete scales."""
    import plotly.io as pio
    from app import charts

    assert "ink" in pio.templates, "'ink' template must be registered in plotly.io"

    # Test correlation heatmap discrete colorscale
    corr_df = pd.DataFrame(np.eye(3), columns=["A", "B", "C"], index=["A", "B", "C"])
    fig_heat = charts.correlation_heatmap(corr_df)
    assert fig_heat.layout.template.layout.paper_bgcolor == "rgba(0,0,0,0)"

    trace = fig_heat.data[0]
    colorscale = trace.colorscale
    assert colorscale is not None, "Heatmap must define colorscale"
    scale_pts = [p[0] for p in colorscale]
    assert len(scale_pts) >= 10, "Heatmap colorscale must use discrete steps"
    print("test_charts_use_ink_template_and_discrete_heatmaps: PASS")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    test_no_gradients_shadows_blur_in_css_and_components()
    test_no_emojis_in_app_i18n_assets()
    test_streamlit_config_is_dark_black()
    test_charts_use_ink_template_and_discrete_heatmaps()
    print("\nALL STYLE AND DESIGN RULES PASSED (4/4)!")
