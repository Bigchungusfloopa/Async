"""
app/charts.py — Plotly chart builders.
All functions return a plotly Figure; caller uses st.plotly_chart().
Colorblind-safe palette; consistent with app/ui.py theme.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from app.ui import RISK_COLORS, STRATEGY_COLORS


# ---------------------------------------------------------------------------
# Common layout
# ---------------------------------------------------------------------------
_BASE_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#1A2035"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=11)),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
)

_GRID_STYLE = dict(gridcolor="#f1f5f9", gridwidth=1)
_AXIS_STYLE = dict(showgrid=True, **_GRID_STYLE, zeroline=False,
                   linecolor="#e2e8f0", tickfont=dict(size=11))


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, weight="bold")),
                      **_BASE_LAYOUT)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


# ---------------------------------------------------------------------------
# Growth-of-₹100 line chart
# ---------------------------------------------------------------------------
def growth_chart(growth_data: dict | pd.Series, title: str = "Growth of ₹100",
                 log_scale: bool = False, label: str = "Portfolio") -> go.Figure:
    """Single series growth chart."""
    if isinstance(growth_data, dict):
        dates = list(growth_data.keys())
        values = list(growth_data.values())
    else:
        dates = growth_data.index.astype(str).tolist()
        values = growth_data.tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values, mode="lines",
        name=label,
        line=dict(color="#4F8EF7", width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
        hovertemplate="%{x}<br>₹%{y:.2f}<extra></extra>",
    ))
    # Add 100 baseline
    fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8", line_width=1)
    if log_scale:
        fig.update_yaxes(type="log")
    return _apply_base(fig, title)


# ---------------------------------------------------------------------------
# Multi-strategy growth comparison
# ---------------------------------------------------------------------------
def multi_growth_chart(growth_dict: dict[str, dict], title: str = "Growth Comparison",
                       log_scale: bool = False) -> go.Figure:
    """Multiple growth lines, one per strategy."""
    fig = go.Figure()
    for strategy, data in growth_dict.items():
        color = STRATEGY_COLORS.get(strategy, "#95A5A6")
        if isinstance(data, dict):
            dates = list(data.keys())
            values = list(data.values())
        else:
            dates = data.index.astype(str).tolist()
            values = data.tolist()
        fig.add_trace(go.Scatter(
            x=dates, y=values, mode="lines",
            name=strategy,
            line=dict(color=color, width=2.5),
            hovertemplate=f"{strategy}: ₹%{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8", line_width=1)
    if log_scale:
        fig.update_yaxes(type="log")
    return _apply_base(fig, title)


# ---------------------------------------------------------------------------
# Drawdown / underwater chart
# ---------------------------------------------------------------------------
def drawdown_chart(growth_data: dict | pd.Series, title: str = "Drawdown") -> go.Figure:
    """Underwater / drawdown chart from growth series."""
    if isinstance(growth_data, dict):
        values = np.array(list(growth_data.values()), dtype=float)
        dates = list(growth_data.keys())
    else:
        values = growth_data.values
        dates = growth_data.index.astype(str).tolist()

    rolling_max = np.maximum.accumulate(values)
    drawdown = (values - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=drawdown.tolist(), mode="lines",
        name="Drawdown",
        line=dict(color="#E74C3C", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.12)",
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
    return _apply_base(fig, title)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
def correlation_heatmap(corr_matrix: dict | pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    if isinstance(corr_matrix, dict):
        tickers = list(corr_matrix.keys())
        z = [[corr_matrix[r].get(c, 0) for c in tickers] for r in tickers]
    else:
        tickers = list(corr_matrix.columns)
        z = corr_matrix.values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z, x=tickers, y=tickers,
        colorscale=[[0, "#3b82f6"], [0.5, "#f8fafc"], [1, "#ef4444"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=12, len=0.8),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14, weight="bold")),
                      **_BASE_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Risk contribution vs weight grouped bar
# ---------------------------------------------------------------------------
def risk_contribution_chart(weights: dict, risk_contrib: dict,
                             title: str = "Weight vs Risk Contribution") -> go.Figure:
    tickers = list(weights.keys())
    w_vals = [weights.get(t, 0) * 100 for t in tickers]
    rc_vals = [risk_contrib.get(t, 0) * 100 for t in tickers]

    colors_w = ["#4F8EF7"] * len(tickers)
    colors_rc = []
    for w, rc in zip(w_vals, rc_vals):
        # Highlight if risk contribution greatly exceeds weight (>1.3x)
        if rc > w * 1.3:
            colors_rc.append("#E74C3C")
        else:
            colors_rc.append("#2ECC71")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Weight %", x=tickers, y=w_vals,
                         marker_color=colors_w,
                         hovertemplate="%{x}: %{y:.1f}%<extra>Weight</extra>"))
    fig.add_trace(go.Bar(name="Risk Contribution %", x=tickers, y=rc_vals,
                         marker_color=colors_rc,
                         hovertemplate="%{x}: %{y:.1f}%<extra>Risk Contribution</extra>"))
    fig.update_layout(barmode="group")
    return _apply_base(fig, title)


# ---------------------------------------------------------------------------
# Weight comparison bars (original vs recommended)
# ---------------------------------------------------------------------------
def weight_comparison_chart(original: dict, recommended: dict,
                             title: str = "Weight Comparison") -> go.Figure:
    tickers = sorted(set(original.keys()) | set(recommended.keys()))
    orig_vals = [original.get(t, 0) * 100 for t in tickers]
    rec_vals  = [recommended.get(t, 0) * 100 for t in tickers]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Original", x=tickers, y=orig_vals,
                         marker_color=STRATEGY_COLORS["Original"],
                         hovertemplate="%{x}: %{y:.1f}%<extra>Original</extra>"))
    fig.add_trace(go.Bar(name="Recommended", x=tickers, y=rec_vals,
                         marker_color=STRATEGY_COLORS["Recommended"],
                         hovertemplate="%{x}: %{y:.1f}%<extra>Recommended</extra>"))
    fig.update_layout(barmode="group")
    return _apply_base(fig, title)


# ---------------------------------------------------------------------------
# Allocation pie chart
# ---------------------------------------------------------------------------
def allocation_pie(weights: dict, title: str = "Allocation") -> go.Figure:
    labels = list(weights.keys())
    values = [v * 100 for v in weights.values()]

    PALETTE = ["#4F8EF7","#2ECC71","#F39C12","#E74C3C","#9B59B6",
               "#1ABC9C","#E67E22","#3498DB","#E91E63","#00BCD4"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4,
        marker=dict(colors=PALETTE[:len(labels)],
                    line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    return _apply_base(fig, title)


# ---------------------------------------------------------------------------
# Risk-return scatter
# ---------------------------------------------------------------------------
def risk_return_scatter(points: list[dict], title: str = "Risk-Return") -> go.Figure:
    """Scatter plot: x=ann_vol, y=ann_return, colored by type."""
    type_colors = {"asset": "#94a3b8", "portfolio": "#4F8EF7", "benchmark": "#9B59B6"}
    type_symbols = {"asset": "circle", "portfolio": "star", "benchmark": "diamond"}
    type_sizes   = {"asset": 10, "portfolio": 18, "benchmark": 16}

    fig = go.Figure()
    for ptype in ["asset", "portfolio", "benchmark"]:
        pts = [p for p in points if p.get("type") == ptype]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["ann_vol"] * 100 for p in pts],
            y=[p["ann_return"] * 100 for p in pts],
            mode="markers+text",
            name=ptype.capitalize(),
            text=[p["name"] for p in pts],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                color=type_colors[ptype],
                symbol=type_symbols[ptype],
                size=type_sizes[ptype],
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))
    fig.update_xaxes(title_text="Annualized Volatility (%)")
    fig.update_yaxes(title_text="Annualized Return (%)")
    return _apply_base(fig, title)


# ---------------------------------------------------------------------------
# Metric comparison bar chart (strategies)
# ---------------------------------------------------------------------------
def metric_bar_chart(compare_data: list[dict], metric: str, title: str = "") -> go.Figure:
    strategies = [d.get("strategy", "") for d in compare_data]
    values = [d.get(metric, 0) * 100 for d in compare_data]
    colors = [STRATEGY_COLORS.get(s, "#94a3b8") for s in strategies]

    fig = go.Figure(go.Bar(
        x=strategies, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    return _apply_base(fig, title)


# ---------------------------------------------------------------------------
# Monte Carlo fan chart
# ---------------------------------------------------------------------------
def monte_carlo_chart(mc_data: dict, title: str = "Monte Carlo (1,000 Paths)") -> go.Figure:
    """Fan chart with percentile bands."""
    percentiles = mc_data.get("percentiles", {})
    labels = mc_data.get("time_labels", list(range(len(next(iter(percentiles.values()), [])))))

    BAND_COLORS = {
        "p5":  "rgba(231,76,60,0.15)",
        "p25": "rgba(243,156,18,0.20)",
        "p50": "rgba(79,142,247,1.0)",
        "p75": "rgba(46,204,113,0.20)",
        "p95": "rgba(46,204,113,0.15)",
    }
    LINE_COLORS = {
        "p5":  "#E74C3C",
        "p25": "#F39C12",
        "p50": "#4F8EF7",
        "p75": "#2ECC71",
        "p95": "#27AE60",
    }
    NAMES = {"p5": "5th pct", "p25": "25th pct", "p50": "Median",
             "p75": "75th pct", "p95": "95th pct"}

    fig = go.Figure()
    # Fill between p25 and p75
    if "p25" in percentiles and "p75" in percentiles:
        fig.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=percentiles["p75"] + percentiles["p25"][::-1],
            fill="toself",
            fillcolor="rgba(79,142,247,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, name="IQR",
            hoverinfo="skip",
        ))

    for pct_key in ["p5", "p25", "p50", "p75", "p95"]:
        if pct_key not in percentiles:
            continue
        is_median = pct_key == "p50"
        fig.add_trace(go.Scatter(
            x=labels, y=percentiles[pct_key],
            mode="lines",
            name=NAMES[pct_key],
            line=dict(color=LINE_COLORS[pct_key],
                      width=3 if is_median else 1.5,
                      dash="solid" if is_median else "dot"),
            hovertemplate=f"{NAMES[pct_key]}: ₹%{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8", line_width=1)
    return _apply_base(fig, title)
