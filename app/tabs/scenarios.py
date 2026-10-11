"""
app/tabs/scenarios.py — Scenarios tab.
Scenario picker; result cards (value before/after, % change, new risk label, new vol);
Monte Carlo fan chart; probability of loss; expected shortfall.
"""

from __future__ import annotations

import streamlit as st
import json

import app.api as api
import app.charts as charts
from app.ui import metric_card, fmt_pct, fmt_inr, risk_badge, info_banner
from ml.i18n import t


_SCENARIOS = {
    "crash_20":       {"label_key": "crash_20",    "type": "crash",    "value": -0.20},
    "crash_30":       {"label_key": "crash_30",    "type": "crash",    "value": -0.30},
    "recovery_15":    {"label_key": "recovery_15", "type": "recovery", "value":  0.15},
    "vol_x2":         {"label_key": "vol_x2",      "type": "vol",      "value":  2.0},
    "sector_downturn":{"label_key": "sector_downturn", "type": "sector", "value": -0.30},
}

_SECTORS = ["IT", "Banking", "FMCG", "Energy", "Pharma", "Auto", "Telecom", "Finance"]


def render(lang: str, tickers: list[str], weights: dict[str, float],
           total_investment: float, start: str, end: str, **kwargs) -> None:
    """Render the Scenarios tab."""
    if not tickers:
        st.info(t("no_data", lang))
        return

    tickers_t = tuple(tickers)
    raw_w     = tuple(weights.get(tk, 0.0) for tk in tickers)
    total_raw = sum(raw_w) or 1.0
    weights_t = tuple(w / total_raw for w in raw_w)

    # -----------------------------------------------------------------------
    # Scenario picker
    # -----------------------------------------------------------------------
    st.subheader(t("scenario_picker", lang))
    scenario_labels = {k: t(v["label_key"], lang) for k, v in _SCENARIOS.items()}
    selected_key = st.selectbox(
        t("scenario_picker", lang),
        options=list(_SCENARIOS.keys()),
        format_func=lambda k: scenario_labels[k],
        key="scenario_select",
        label_visibility="collapsed",
    )
    scen_def = _SCENARIOS[selected_key]

    # Sector + slider for sector downturn
    sector_val = None
    downturn_override = scen_def["value"]
    if selected_key == "sector_downturn":
        col_sec, col_slider = st.columns(2)
        with col_sec:
            sector_val = st.selectbox(t("sector_select", lang), _SECTORS, key="scen_sector")
        with col_slider:
            downturn_pct = st.slider(t("downturn_pct", lang), min_value=-50, max_value=-5,
                                     value=-30, step=5, key="scen_downturn_pct")
            downturn_override = downturn_pct / 100.0

    scenario_params = (scen_def["type"], downturn_override)

    st.divider()

    # -----------------------------------------------------------------------
    # Apply scenario
    # -----------------------------------------------------------------------
    with st.spinner("Applying scenario…"):
        try:
            scen_result = api.apply_scenario(
                tickers_t, weights_t, selected_key, scenario_params, start, end
            )
        except Exception as e:
            st.error(f"{t('no_data', lang)}: {e}")
            return

    value_before = scen_result.get("value_before", total_investment)
    value_after  = scen_result.get("value_after",  total_investment * (1 + downturn_override))
    pct_change   = scen_result.get("pct_change",   downturn_override)
    new_vol      = scen_result.get("new_volatility", 0.0)

    # New risk label
    try:
        stressed_risk = api.classify_stressed(tickers_t, weights_t, scenario_params, start, end)
        new_label = stressed_risk.get("label", "High")
        new_probs = stressed_risk.get("probs", {})
    except Exception:
        new_label = scen_result.get("new_risk_label", "High")
        new_probs = {}

    # -----------------------------------------------------------------------
    # Result cards
    # -----------------------------------------------------------------------
    st.markdown("**Scenario Impact**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(t("value_before", lang), fmt_inr(value_before))
    with c2:
        metric_card(t("value_after", lang), fmt_inr(value_after),
                    delta_positive=value_after >= value_before)
    with c3:
        metric_card(t("pct_change", lang), f"{pct_change*100:+.1f}%",
                    delta_positive=pct_change >= 0)
    with c4:
        metric_card(t("new_volatility", lang), fmt_pct(new_vol) if new_vol else "—")

    st.markdown(f"**{t('new_risk_label', lang)}**")
    risk_badge(new_label, lang)

    st.divider()

    # -----------------------------------------------------------------------
    # Monte Carlo fan chart
    # -----------------------------------------------------------------------
    st.subheader(t("monte_carlo_chart", lang))
    with st.spinner("Running Monte Carlo…"):
        try:
            mc_data = api.monte_carlo(tickers_t, weights_t, start, end)
        except Exception as e:
            st.warning(f"Monte Carlo: {e}")
            mc_data = scen_result.get("monte_carlo", {})

    if mc_data and mc_data.get("percentiles"):
        fig_mc = charts.monte_carlo_chart(mc_data, "")
        st.plotly_chart(fig_mc, use_container_width=True)

        # Probability of loss + Expected shortfall
        c_pl, c_es = st.columns(2)
        prob_loss = mc_data.get("prob_of_loss", 0.0)
        exp_short = mc_data.get("expected_shortfall", 0.0)
        with c_pl:
            metric_card(
                t("prob_of_loss", lang),
                fmt_pct(prob_loss),
                delta_positive=prob_loss < 0.30,
            )
        with c_es:
            metric_card(
                t("expected_shortfall", lang),
                fmt_pct(exp_short),
                delta_positive=False,
            )
    else:
        st.info(t("no_data", lang))
