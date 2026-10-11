"""
app/charts.py — Plotly chart builders with 'ink' trading terminal template.
Rules:
- Template 'ink': transparent canvas, #171717 grid, #2A2A2A zero line, no axis lines,
  mono 11px ticks, series palette: #F5F5F5, #3B82F6, #14B8A6, #F59E0B, #A78BFA, #6B7280.
- NO gradients anywhere: flat fills only at fixed 0.10 opacity.
- Heatmaps use discrete stepped bins (5 to 7 flat colours), NEVER continuous ramp.
- NO emojis anywhere in titles, axes, or hover templates.
- Config: displayModeBar=False, transition duration 400ms.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Discrete Stepped Palettes & Config
# ---------------------------------------------------------------------------
SERIES_PALETTE = ["#F5F5F5", "#3B82F6", "#14B8A6", "#F59E0B", "#A78BFA", "#6B7280"]

HEAT_STEPS_SEQUENTIAL = ["#111111", "#2A2A2A", "#4D4D4D", "#8A8A8A", "#F5F5F5"]
HEAT_STEPS_DIVERGING = ["#7F1D1D", "#B91C1C", "#EF4444", "#1A1A1A", "#22C55E", "#16A34A", "#166534"]

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def make_discrete_colorscale(colors: list[str]) -> list[list]:
    """Convert a list of flat colors into a discrete stepped Plotly colorscale."""
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale.append([i / n, c])
        scale.append([(i + 1) / n, c])
    return scale


DISCRETE_SCALE_SEQ = make_discrete_colorscale(HEAT_STEPS_SEQUENTIAL)
DISCRETE_SCALE_DIV = make_discrete_colorscale(HEAT_STEPS_DIVERGING)

# ---------------------------------------------------------------------------
# Register 'ink' Template
# ---------------------------------------------------------------------------
pio.templates["ink"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES_PALETTE,
        font=dict(family="Inter, sans-serif", size=12, color="#F5F5F5"),
        margin=dict(l=16, r=16, t=24, b=16),
        xaxis=dict(
            showgrid=True,
            gridcolor="#171717",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#2A2A2A",
            showline=False,
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#8A8A8A"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#171717",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#2A2A2A",
            showline=False,
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#8A8A8A"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Inter, sans-serif", size=11, color="#8A8A8A"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#111111",
            bordercolor="#2E2E2E",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F5F5F5"),
        ),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
)
pio.templates.default = "ink"


def _clean_figure(fig: go.Figure) -> go.Figure:
    """Ensure template ink and clean card styling."""
    fig.update_layout(template="ink")
    return fig


# ---------------------------------------------------------------------------
# 1. Growth Chart (Simulate & Risk)
# ---------------------------------------------------------------------------
def growth_chart(growth_data: dict | pd.Series, log_scale: bool = False,
                 label: str = "Portfolio", regimes: bool = False) -> go.Figure:
    """Growth of 100 line with flat fill (0.10 opacity) and optional regime markers."""
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
        line=dict(color="#F5F5F5", width=1.75),
        fill="tozeroy",
        fillcolor="rgba(245, 245, 245, 0.10)",
        hovertemplate="%{x}<br>INR %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color="#2E2E2E", line_width=1)

    if regimes and len(values) > 30:
        # shade the real worst peak-to-trough period of this series
        v = np.asarray(values, dtype="float64")
        dd = v / np.maximum.accumulate(v) - 1.0
        trough = int(np.argmin(dd))
        peak = int(np.argmax(v[: trough + 1]))
        if trough > peak and dd[trough] < -0.02:
            fig.add_vrect(x0=dates[peak], x1=dates[trough], fillcolor="#EF4444", opacity=0.08, line_width=0,
                          annotation_text=f"MAX DRAWDOWN {dd[trough] * 100:.0f}%", annotation_position="top left",
                          annotation_font=dict(size=9, color="#8A8A8A", family="JetBrains Mono"))

    if log_scale:
        fig.update_yaxes(type="log")
    elif len(values):
        lo, hi = min(min(values), 100.0), max(max(values), 100.0)
        pad = max((hi - lo) * 0.08, 1.0)
        fig.update_yaxes(range=[lo - pad, hi + pad])      # the fill is clipped; do not force the axis to zero
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 2. Underwater Drawdown Area Chart
# ---------------------------------------------------------------------------
def underwater_chart(growth_data: dict | pd.Series) -> go.Figure:
    """Drawdown area with flat fill (opacity 0.10) and max-drawdown marker line."""
    if isinstance(growth_data, dict):
        values = np.array(list(growth_data.values()), dtype=float)
        dates = list(growth_data.keys())
    else:
        values = growth_data.values
        dates = growth_data.index.astype(str).tolist()

    rolling_max = np.maximum.accumulate(values) if len(values) > 0 else np.array([100.0])
    drawdown = (values - rolling_max) / rolling_max * 100 if len(values) > 0 else np.array([0.0])
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=drawdown.tolist(), mode="lines",
        name="Drawdown",
        line=dict(color="#EF4444", width=1.75),
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.10)",
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#2E2E2E", line_width=1)
    if max_dd < 0:
        fig.add_hline(y=max_dd, line_dash="dash", line_color="#EF4444", line_width=1,
                      annotation_text=f"MAX DD: {max_dd:.1f}%",
                      annotation_font=dict(size=9, color="#EF4444", family="JetBrains Mono"))
    return _clean_figure(fig)


def drawdown_chart(growth_data: dict | pd.Series, title: str = "") -> go.Figure:
    """Alias for underwater_chart."""
    return underwater_chart(growth_data)


# ---------------------------------------------------------------------------
# 3. Correlation Heatmap (Discrete Stepped Bins)
# ---------------------------------------------------------------------------
def correlation_heatmap(corr_matrix: dict | pd.DataFrame) -> go.Figure:
    """Correlation matrix with 5 flat discrete bins and value labels."""
    if isinstance(corr_matrix, dict):
        tickers = list(corr_matrix.keys())
        z = [[corr_matrix[r].get(c, 0) for c in tickers] for r in tickers]
    else:
        tickers = list(corr_matrix.columns)
        z = corr_matrix.values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z, x=tickers, y=tickers,
        colorscale=DISCRETE_SCALE_SEQ,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#F5F5F5"),
        hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
        showscale=False,
    ))
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 4. Weight vs Risk Contribution (Grouped Horizontal Bars)
# ---------------------------------------------------------------------------
def weight_vs_risk_bars(weights: dict, risk_contrib: dict) -> go.Figure:
    """Grouped horizontal bars flagging assets where risk exceeds weight."""
    tickers = list(weights.keys())
    w_vals = [weights.get(t, 0) * 100 for t in tickers]
    rc_vals = [risk_contrib.get(t, 0) * 100 for t in tickers]

    rc_colors = ["#EF4444" if rc > (w * 1.25) else "#3B82F6" for w, rc in zip(w_vals, rc_vals)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Weight %", y=tickers, x=w_vals, orientation="h",
        marker_color="#F5F5F5",
        hovertemplate="%{y} Weight: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Risk Contribution %", y=tickers, x=rc_vals, orientation="h",
        marker_color=rc_colors,
        hovertemplate="%{y} Risk Share: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(barmode="group")
    return _clean_figure(fig)


def risk_contribution_chart(weights: dict, risk_contrib: dict, title: str = "") -> go.Figure:
    """Alias for weight_vs_risk_bars."""
    return weight_vs_risk_bars(weights, risk_contrib)


# ---------------------------------------------------------------------------
# 5. Allocation Treemap
# ---------------------------------------------------------------------------
def allocation_treemap(weights: dict) -> go.Figure:
    """Flat allocation treemap using series palette."""
    labels = list(weights.keys())
    values = [weights[t] * 100 for t in labels]
    colors = [SERIES_PALETTE[i % len(SERIES_PALETTE)] for i in range(len(labels))]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=values,
        marker=dict(colors=colors, line=dict(width=1, color="#1F1F1F")),
        textinfo="label+value+percent root",
        textfont=dict(family="JetBrains Mono, monospace", size=12, color="#000000"),
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>",
    ))
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 6. Rolling 60-Day Volatility Line
# ---------------------------------------------------------------------------
def rolling_vol_chart(prices_df: pd.DataFrame) -> go.Figure:
    """Rolling 60-day annualized volatility line."""
    returns = prices_df.pct_change().dropna()
    rolling_vol = (returns.mean(axis=1).rolling(60).std() * np.sqrt(252) * 100).dropna()
    dates = rolling_vol.index.astype(str).tolist()
    vals = rolling_vol.tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=vals, mode="lines",
        name="60D Volatility",
        line=dict(color="#3B82F6", width=1.75),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.10)",
        hovertemplate="%{x}<br>Vol: %{y:.1f}%<extra></extra>",
    ))
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 7. Daily Return Distribution Histogram with VaR and CVaR
# ---------------------------------------------------------------------------
def return_distribution_hist(returns_list: list[float] | pd.Series,
                             var_95: float = -0.02, cvar_95: float = -0.035) -> go.Figure:
    """Histogram of daily returns with VaR and CVaR vertical markers."""
    if isinstance(returns_list, pd.Series):
        vals = (returns_list * 100).tolist()
    else:
        vals = [r * 100 for r in returns_list]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals,
        marker_color="#2A2A2A",
        marker_line=dict(color="#1F1F1F", width=1),
        name="Daily Return %",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=var_95 * 100, line_dash="dash", line_color="#F59E0B", line_width=1.5,
                  annotation_text=f"VaR 95%: {var_95*100:.1f}%",
                  annotation_font=dict(size=9, color="#F59E0B", family="JetBrains Mono"))
    fig.add_vline(x=cvar_95 * 100, line_dash="dash", line_color="#EF4444", line_width=1.5,
                  annotation_text=f"CVaR 95%: {cvar_95*100:.1f}%",
                  annotation_font=dict(size=9, color="#EF4444", family="JetBrains Mono"))
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 8. Top Factors Signed Horizontal Bars
# ---------------------------------------------------------------------------
def top_factors_bars(factors: list[dict] | list[str]) -> go.Figure:
    """Top model risk factors signed horizontal bars."""
    names = []
    shares = []
    colors = []
    for i, f in enumerate(factors[:5]):
        if isinstance(f, dict):
            nm = f.get("name", f"Factor {i+1}")
            imp = f.get("impact", (i + 1) * -0.15)
        else:
            nm = str(f)
            imp = -0.2 * (i + 1)
        names.append(nm)
        shares.append(abs(imp) * 100)                     # size = share of the model's attribution
        colors.append("#EF4444" if imp < 0 else "#22C55E")  # red = pushes risk up, green = pulls it down

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=shares, orientation="h", marker_color=colors, width=0.55,
        text=[f"{v:.0f}%" for v in shares], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.0f}% of the model's attribution<extra></extra>",
    ))
    fig.update_layout(
        height=max(150, 46 * len(names) + 40), bargap=0.4, showlegend=False,
        margin=dict(l=8, r=40, t=8, b=8),
        xaxis=dict(visible=False, range=[0, max(shares + [10]) * 1.25]),
        yaxis=dict(autorange="reversed", automargin=True),
    )
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 9. Versus Benchmark Grouped Comparison
# ---------------------------------------------------------------------------
def vs_benchmark_chart(port_metrics: dict, bench_metrics: dict) -> go.Figure:
    """Grouped bar comparing Portfolio vs Benchmark with signed deltas."""
    categories = ["Ann. Return", "Volatility", "Max Drawdown", "Sharpe Ratio"]
    port_vals = [
        port_metrics.get("ann_return", 0.15) * 100,
        port_metrics.get("ann_vol", 0.18) * 100,
        abs(port_metrics.get("max_drawdown", -0.12)) * 100,
        port_metrics.get("sharpe", 0.85),
    ]
    bench_vals = [
        bench_metrics.get("ann_return", 0.12) * 100,
        bench_metrics.get("ann_vol", 0.15) * 100,
        abs(bench_metrics.get("max_drawdown", -0.15)) * 100,
        bench_metrics.get("sharpe", 0.70),
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Portfolio", x=categories, y=port_vals,
                         marker_color="#F5F5F5", hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Benchmark", x=categories, y=bench_vals,
                         marker_color="#6B7280", hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 10. Diverging Weight Deltas Bar Chart
# ---------------------------------------------------------------------------
def diverging_weight_deltas(deltas: dict[str, float]) -> go.Figure:
    """Diverging horizontal bar: red reduce, green increase."""
    tickers = list(deltas.keys())
    vals = [deltas[t] * 100 for t in tickers]
    colors = ["#22C55E" if v >= 0 else "#EF4444" for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=tickers, x=vals, orientation="h",
        marker_color=colors,
        hovertemplate="%{y} Delta: %{x:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#2E2E2E", line_width=1)
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 11. Allocation Donuts Side by Side / Single
# ---------------------------------------------------------------------------
def allocation_pie(weights: dict, title: str = "") -> go.Figure:
    """Flat donut chart with thin 1px gaps and series palette."""
    labels = list(weights.keys())
    values = [v * 100 for v in weights.values()]
    colors = [SERIES_PALETTE[i % len(SERIES_PALETTE)] for i in range(len(labels))]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0A0A0A", width=1.5)),
        textinfo="label+percent",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#F5F5F5"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    return _clean_figure(fig)


def weight_comparison_chart(original: dict, recommended: dict, title: str = "") -> go.Figure:
    """Weight comparison grouped bars."""
    tickers = sorted(set(original.keys()) | set(recommended.keys()))
    orig_vals = [original.get(t, 0) * 100 for t in tickers]
    rec_vals  = [recommended.get(t, 0) * 100 for t in tickers]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Original", x=tickers, y=orig_vals, marker_color="#F5F5F5"))
    fig.add_trace(go.Bar(name="Recommended", x=tickers, y=rec_vals, marker_color="#3B82F6"))
    fig.update_layout(barmode="group")
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 12. Risk-Return Scatter with Flat-Dot Cloud
# ---------------------------------------------------------------------------
def risk_return_scatter(points: list[dict], cloud: bool = True,
                        cloud_points: list[dict] | None = None) -> go.Figure:
    """Risk-return scatter; optional cloud of *real* random portfolios (``cloud_points``) behind it."""
    fig = go.Figure()

    if cloud and cloud_points:
        cloud_vols = np.array([c["ann_vol"] for c in cloud_points]) * 100
        cloud_rets = np.array([c["ann_return"] for c in cloud_points]) * 100
        fig.add_trace(go.Scatter(
            x=cloud_vols.tolist(), y=cloud_rets.tolist(),
            mode="markers",
            marker=dict(size=4, color="#1A1A1A"),
            showlegend=False,
            hoverinfo="skip",
        ))

    type_colors = {"asset": "#8A8A8A", "portfolio": "#F5F5F5", "benchmark": "#3B82F6"}
    type_symbols = {"asset": "circle", "portfolio": "square", "benchmark": "diamond"}
    type_sizes   = {"asset": 8, "portfolio": 14, "benchmark": 12}

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
            textfont=dict(family="JetBrains Mono, monospace", size=10, color="#8A8A8A"),
            marker=dict(
                color=type_colors[ptype],
                symbol=type_symbols[ptype],
                size=type_sizes[ptype],
                line=dict(color="#000000", width=1),
            ),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))

    fig.update_xaxes(title_text="ANNUAL VOLATILITY (%)")
    fig.update_yaxes(title_text="ANNUAL RETURN (%)")
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 13. Multi-Strategy Growth Comparison (Distinct Line Dash Styles)
# ---------------------------------------------------------------------------
def multi_growth_chart(growth_dict: dict[str, dict], log_scale: bool = False,
                       title: str = "") -> go.Figure:
    """Growth lines with distinct dashes: Original (solid), Recommended (dash), Equal (dot), Benchmark (longdash)."""
    DASH_MAP = {
        "Original": "solid",
        "Recommended": "dash",
        "Equal Weight": "dot",
        "Benchmark": "longdash",
    }
    COLOR_MAP = {
        "Original": "#F5F5F5",
        "Recommended": "#3B82F6",
        "Equal Weight": "#F59E0B",
        "Benchmark": "#6B7280",
    }

    fig = go.Figure()
    for strat, data in growth_dict.items():
        dash_style = DASH_MAP.get(strat, "solid")
        color = COLOR_MAP.get(strat, "#8A8A8A")
        if isinstance(data, dict):
            dates = list(data.keys())
            vals = list(data.values())
        else:
            dates = data.index.astype(str).tolist()
            vals = data.tolist()

        fig.add_trace(go.Scatter(
            x=dates, y=vals, mode="lines",
            name=strat,
            line=dict(color=color, width=1.75, dash=dash_style),
            hovertemplate=f"{strat}: INR %{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="#2E2E2E", line_width=1)
    if log_scale:
        fig.update_yaxes(type="log")
    return _clean_figure(fig)


def strategy_growth_comparison(growth_dict: dict[str, dict], log_scale: bool = False) -> go.Figure:
    """Alias for multi_growth_chart."""
    return multi_growth_chart(growth_dict, log_scale=log_scale)


# ---------------------------------------------------------------------------
# 14. Strategy Metric Bar Charts
# ---------------------------------------------------------------------------
def metric_bar_chart(compare_data: list[dict], metric: str, title: str = "") -> go.Figure:
    """Grouped small multiple bars for return, vol, drawdown, Sharpe."""
    strategies = [d.get("strategy", "") for d in compare_data]
    vals = [d.get(metric, 0) * 100 if metric != "sharpe" else d.get(metric, 0) for d in compare_data]
    colors = [SERIES_PALETTE[i % len(SERIES_PALETTE)] for i in range(len(strategies))]

    fig = go.Figure(go.Bar(
        x=strategies, y=vals,
        marker_color=colors,
        text=[f"{v:.1f}%" if metric != "sharpe" else f"{v:.2f}" for v in vals],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10, color="#8A8A8A"),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 15. Monte Carlo Fan Chart (Flat Percentile Bands at Different Opacities)
# ---------------------------------------------------------------------------
def monte_carlo_chart(mc_data: dict, title: str = "") -> go.Figure:
    """Monte Carlo fan chart with flat fills at 5/25/50/75/95 percentiles."""
    percentiles = mc_data.get("percentiles", {})
    labels = mc_data.get("time_labels", list(range(len(next(iter(percentiles.values()), [])))))

    fig = go.Figure()

    # Outer band (5th to 95th) flat fill
    if "p5" in percentiles and "p95" in percentiles:
        fig.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=percentiles["p95"] + percentiles["p5"][::-1],
            fill="toself",
            fillcolor="rgba(34, 197, 94, 0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, name="5-95 Pct",
            hoverinfo="skip",
        ))

    # Inner band (25th to 75th) flat fill
    if "p25" in percentiles and "p75" in percentiles:
        fig.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=percentiles["p75"] + percentiles["p25"][::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, name="IQR 25-75",
            hoverinfo="skip",
        ))

    # Median line
    if "p50" in percentiles:
        fig.add_trace(go.Scatter(
            x=labels, y=percentiles["p50"], mode="lines",
            name="Median",
            line=dict(color="#F5F5F5", width=2.0),
            hovertemplate="Median: INR %{y:.1f}<extra></extra>",
        ))

    # 5th percentile line (stress)
    if "p5" in percentiles:
        fig.add_trace(go.Scatter(
            x=labels, y=percentiles["p5"], mode="lines",
            name="5th Pct",
            line=dict(color="#EF4444", width=1.2, dash="dot"),
            hovertemplate="5th Pct: INR %{y:.1f}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dot", line_color="#2E2E2E", line_width=1)
    return _clean_figure(fig)


def mc_fan_chart(mc_data: dict) -> go.Figure:
    """Alias for monte_carlo_chart."""
    return monte_carlo_chart(mc_data)


# ---------------------------------------------------------------------------
# 16. Terminal Values Histogram
# ---------------------------------------------------------------------------
def terminal_values_hist(terminal_vals: list[float] | np.ndarray, threshold: float = 100.0) -> go.Figure:
    """Histogram of terminal values with loss threshold vertical marker."""
    vals = np.array(terminal_vals, dtype=float)
    colors = ["#EF4444" if v < threshold else "#2A2A2A" for v in vals]

    fig = go.Figure(go.Histogram(
        x=vals,
        marker_color="#2A2A2A",
        marker_line=dict(color="#1F1F1F", width=1),
        hovertemplate="Terminal Value: %{x:.1f}<br>Paths: %{y}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#EF4444", line_width=1.5,
                  annotation_text="CAPITAL LOSS THRESHOLD",
                  annotation_font=dict(size=9, color="#EF4444", family="JetBrains Mono"))
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 17. Shock Waterfall Chart
# ---------------------------------------------------------------------------
def shock_waterfall(impacts: dict[str, float]) -> go.Figure:
    """Waterfall chart of shock impact per asset."""
    tickers = list(impacts.keys())
    values = [impacts[t] * 100 for t in tickers]

    fig = go.Figure(go.Waterfall(
        name="Shock Impact",
        orientation="v",
        measure=["relative"] * len(tickers),
        x=tickers,
        y=values,
        connector=dict(line=dict(color="#2E2E2E", width=1)),
        decreasing=dict(marker=dict(color="#EF4444")),
        increasing=dict(marker=dict(color="#22C55E")),
        text=[f"{v:+.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10, color="#8A8A8A"),
        hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
    ))
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 18. Sentiment Diverging Bar & Donut
# ---------------------------------------------------------------------------
def sentiment_diverging_bar(sentiments: dict[str, float]) -> go.Figure:
    """Diverging horizontal bar chart for sentiment scores (-1 to +1)."""
    tickers = list(sentiments.keys())
    scores = [sentiments[t] for t in tickers]
    colors = ["#22C55E" if s >= 0 else "#EF4444" for s in scores]

    fig = go.Figure(go.Bar(
        y=tickers, x=scores, orientation="h",
        marker_color=colors,
        hovertemplate="%{y} Sentiment: %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#2E2E2E", line_width=1)
    fig.update_xaxes(range=[-1, 1])
    return _clean_figure(fig)


def sentiment_share_donut(share: dict[str, float]) -> go.Figure:
    """Sentiment distribution donut."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [share.get("Positive", 0.4), share.get("Neutral", 0.4), share.get("Negative", 0.2)]
    colors = ["#22C55E", "#8A8A8A", "#EF4444"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#0A0A0A", width=1.5)),
        textinfo="label+percent",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#F5F5F5"),
        hovertemplate="<b>%{label}</b>: %{percent}<extra></extra>",
    ))
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 19. Holdings P&L Bar Chart
# ---------------------------------------------------------------------------
def holdings_pnl_bar(holdings: list[dict]) -> go.Figure:
    """Bar chart of P&L per holding in red/green."""
    tickers = [h.get("ticker", "") for h in holdings]
    pnls = [h.get("pnl", 0.0) for h in holdings]
    colors = ["#22C55E" if p >= 0 else "#EF4444" for p in pnls]

    fig = go.Figure(go.Bar(
        x=tickers, y=pnls,
        marker_color=colors,
        text=[f"INR {p:+,.0f}" for p in pnls],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10, color="#8A8A8A"),
        hovertemplate="%{x}: INR %{y:+,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#2E2E2E", line_width=1)
    return _clean_figure(fig)


# ---------------------------------------------------------------------------
# 20. Diversification Bars (Effective N & HHI)
# ---------------------------------------------------------------------------
def diversification_bars(n_eff: float, hhi: float) -> go.Figure:
    """Horizontal metric bars for portfolio diversification."""
    metrics = ["Effective N", "HHI Concentration"]
    vals = [n_eff, hhi * 10]
    fig = go.Figure(go.Bar(
        y=metrics, x=vals, orientation="h",
        marker_color=["#3B82F6", "#F59E0B"],
        text=[f"N = {n_eff:.1f}", f"HHI = {hhi:.2f}"],
        textposition="inside",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#F5F5F5"),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    return _clean_figure(fig)
