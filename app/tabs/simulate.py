"""
app/tabs/simulate.py — Simulate tab.
Preset regime buttons + custom date pickers; growth-of-₹100 line chart (log scale toggle);
underwater drawdown chart; metrics table; dropped ticker warning; alternate asset overlay.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

import app.api as api
import app.charts as charts
from app.ui import metric_card, fmt_pct, fmt_number, info_banner
from ml.i18n import t


def render(lang: str, tickers: list[str], weights: dict[str, float],
           start: str, end: str, **kwargs) -> None:
    """Render the Simulate tab."""
    if not tickers:
        st.info(t("no_data", lang))
        return

    presets = api.regime_presets()

    # -----------------------------------------------------------------------
    # Regime preset buttons + custom
    # -----------------------------------------------------------------------
    st.markdown(f"**{t('regime_presets', lang)}**")
    preset_keys = list(presets.keys())
    cols = st.columns(len(preset_keys) + 1)

    sim_start, sim_end = start, end

    # Use session state to track selected preset
    if "_sim_preset" not in st.session_state:
        st.session_state["_sim_preset"] = None

    for i, preset_name in enumerate(preset_keys):
        with cols[i]:
            if st.button(preset_name, key=f"preset_{i}"):
                st.session_state["_sim_preset"] = preset_name
                st.session_state["_sim_custom"] = False

    with cols[-1]:
        if st.button(t("custom", lang), key="preset_custom"):
            st.session_state["_sim_preset"] = None
            st.session_state["_sim_custom"] = True

    # Custom date pickers
    use_custom = st.session_state.get("_sim_custom", False)
    selected_preset = st.session_state.get("_sim_preset")

    if selected_preset and selected_preset in presets:
        sim_start = presets[selected_preset]["start"]
        sim_end   = presets[selected_preset]["end"]
        info_banner(f"Regime: {selected_preset} ({sim_start} → {sim_end})")
    elif use_custom:
        c1, c2 = st.columns(2)
        with c1:
            sim_start = st.date_input(t("start_date", lang), value=pd.to_datetime(start),
                                      key="sim_start_custom").strftime("%Y-%m-%d")
        with c2:
            sim_end = st.date_input(t("end_date", lang), value=pd.to_datetime(end),
                                    key="sim_end_custom").strftime("%Y-%m-%d")

    st.divider()

    # -----------------------------------------------------------------------
    # Log scale toggle
    # -----------------------------------------------------------------------
    log_scale = st.checkbox(t("log_scale", lang), key="sim_log_scale")

    # -----------------------------------------------------------------------
    # Main portfolio simulation
    # -----------------------------------------------------------------------
    tickers_t = tuple(tickers)
    raw_w     = tuple(weights.get(tk, 0.0) for tk in tickers)
    total     = sum(raw_w) or 1.0
    weights_t = tuple(w / total for w in raw_w)

    with st.spinner("Simulating…"):
        try:
            sim_result = api.simulate(tickers_t, weights_t, sim_start, sim_end)
        except Exception as e:
            st.error(f"{t('no_data', lang)}: {e}")
            return

    growth = sim_result.get("growth", {})
    sim_metrics = sim_result.get("metrics", {})

    # Growth chart
    st.markdown(f"**{t('growth_chart', lang)}**")
    fig_growth = charts.growth_chart(growth, "", log_scale=log_scale, label="Portfolio")
    st.plotly_chart(fig_growth, use_container_width=True)

    # Drawdown chart
    st.markdown(f"**{t('underwater_chart', lang)}**")
    fig_dd = charts.drawdown_chart(growth, "")
    st.plotly_chart(fig_dd, use_container_width=True)

    # -----------------------------------------------------------------------
    # Metrics table
    # -----------------------------------------------------------------------
    st.markdown("**Simulation Metrics**")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        metric_card(t("metric_return", lang), fmt_pct(sim_metrics.get("ann_return", 0)))
    with mc2:
        metric_card(t("metric_vol", lang), fmt_pct(sim_metrics.get("ann_vol", 0)))
    with mc3:
        metric_card(t("metric_drawdown", lang), fmt_pct(sim_metrics.get("max_drawdown", 0)))
    with mc4:
        metric_card(t("metric_sharpe", lang), fmt_number(sim_metrics.get("sharpe", 0)))
    with mc5:
        metric_card("Total Return", fmt_pct(sim_metrics.get("total_return", 0)))

    st.divider()

    # -----------------------------------------------------------------------
    # Alternate asset overlay
    # -----------------------------------------------------------------------
    st.markdown(f"**{t('compare_assets', lang)}**")
    import json
    from pathlib import Path
    universe_path = Path(__file__).parent.parent.parent / "data" / "universe.json"
    universe = json.loads(universe_path.read_text()) if universe_path.exists() else {}
    all_tickers = (
        [item["ticker"] for item in universe.get("NSE", [])] +
        [item["ticker"] for item in universe.get("US", [])]
    )

    alt_tickers = st.multiselect(
        "Alternate assets to overlay",
        options=all_tickers,
        default=[],
        key="sim_alt_tickers",
        max_selections=5,
    )

    if alt_tickers:
        alt_t = tuple(alt_tickers)
        eq_w  = tuple(1.0 / len(alt_tickers) for _ in alt_tickers)
        with st.spinner("Simulating alternate portfolio…"):
            try:
                alt_result = api.simulate(alt_t, eq_w, sim_start, sim_end)
                alt_growth = alt_result.get("growth", {})

                combined = {"Your Portfolio": growth, "Alternate Portfolio": alt_growth}
                fig_combined = charts.multi_growth_chart(combined, "Comparison", log_scale)
                st.plotly_chart(fig_combined, use_container_width=True)
            except Exception as e:
                st.warning(f"Alternate simulation failed: {e}")
