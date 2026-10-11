"""
app/components/kpi_strip.py — KPI strip component with count-up animation & sparklines.
Self-contained HTML component rendered via st.components.v1.html.
Progressive enhancement: initial HTML holds final value, JS animates count-up.
"""

from __future__ import annotations

import streamlit.components.v1 as components


def render_kpi_strip(items: list[dict], height: int = 95) -> None:
    """
    Render a responsive row of 4 to 6 KPI tiles.
    Each item in items:
        {
            "label": str,
            "value": str,            # formatted string (e.g. "18.4%", "1.42")
            "num_val": float,        # numeric value for JS count-up target
            "prefix": str,           # optional prefix (e.g. "+", "INR ")
            "suffix": str,           # optional suffix (e.g. "%")
            "delta": str,            # optional signed delta string e.g. "+2.1%"
            "delta_pos": bool,       # True if positive / False if negative
            "sparkline": list[float] # optional list of normalized values [0..1]
        }
    """
    tiles_html = []
    for i, it in enumerate(items):
        label = it.get("label", "").upper()
        final_str = it.get("value", "")
        num_val = it.get("num_val", 0.0)
        prefix = it.get("prefix", "")
        suffix = it.get("suffix", "")
        delta = it.get("delta")
        delta_pos = it.get("delta_pos", True)
        spark = it.get("sparkline", [])

        # Build inline SVG sparkline if provided
        spark_svg = ""
        if spark and len(spark) > 1:
            w, h = 60, 20
            min_v, max_v = min(spark), max(spark)
            rng = max_v - min_v if max_v != min_v else 1.0
            pts = []
            for idx, pt in enumerate(spark):
                x = (idx / (len(spark) - 1)) * w
                y = h - ((pt - min_v) / rng) * (h - 4) - 2
                pts.append(f"{x:.1f},{y:.1f}")
            polyline = " ".join(pts)
            stroke_color = "#22C55E" if delta_pos else "#EF4444"
            spark_svg = f"""
            <svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" fill="none">
                <polyline points="{polyline}" fill="none" stroke="{stroke_color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            """

        delta_html = ""
        if delta:
            d_cls = "pos" if delta_pos else "neg"
            glyph = "▲" if delta_pos else "▼"
            delta_html = f"""<span class="delta {d_cls}"><span class="glyph">{glyph}</span>{delta}</span>"""

        tile = f"""
        <div class="kpi-tile" id="tile-{i}">
            <div class="kpi-header">
                <span class="kpi-label">{label}</span>
                {delta_html}
            </div>
            <div class="kpi-main">
                <span class="kpi-val" id="val-{i}" data-target="{num_val}" data-prefix="{prefix}" data-suffix="{suffix}">{final_str}</span>
                <div class="kpi-spark">{spark_svg}</div>
            </div>
        </div>
        """
        tiles_html.append(tile)

    all_tiles = "\n".join(tiles_html)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #000000;
            color: #F5F5F5;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            overflow: hidden;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat({len(items)}, minmax(0, 1fr));
            gap: 12px;
            width: 100%;
        }}
        .kpi-tile {{
            background: #0A0A0A;
            border: 1px solid #1F1F1F;
            border-radius: 6px;
            padding: 12px 14px;
            transition: border-color 150ms cubic-bezier(0.2, 0.7, 0.2, 1);
        }}
        .kpi-tile:hover {{
            border-color: #2E2E2E;
        }}
        .kpi-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .kpi-label {{
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8A8A8A;
        }}
        .delta {{
            font-family: 'JetBrains Mono', monospace;
            font-variant-numeric: tabular-nums;
            font-size: 11px;
            display: inline-flex;
            align-items: center;
            gap: 3px;
        }}
        .delta.pos {{ color: #22C55E; }}
        .delta.neg {{ color: #EF4444; }}
        .delta .glyph {{ font-size: 8px; }}
        .kpi-main {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .kpi-val {{
            font-family: 'JetBrains Mono', monospace;
            font-variant-numeric: tabular-nums;
            font-size: 22px;
            font-weight: 600;
            color: #F5F5F5;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }}
        .kpi-spark {{
            margin-left: 8px;
        }}
        @media (prefers-reduced-motion: reduce) {{
            * {{ transition: none !important; animation: none !important; }}
        }}
    </style>
    </head>
    <body>
    <div class="kpi-row">
        {all_tiles}
    </div>
    <script>
    (function() {{
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
        const els = document.querySelectorAll('.kpi-val');
        const duration = 900;
        const start = performance.now();

        function easeOut(t) {{
            return 1 - Math.pow(1 - t, 3);
        }}

        function animate(now) {{
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = easeOut(progress);

            els.forEach(el => {{
                const target = parseFloat(el.getAttribute('data-target'));
                if (isNaN(target)) return;
                const prefix = el.getAttribute('data-prefix') || '';
                const suffix = el.getAttribute('data-suffix') || '';
                const current = (target * eased);
                // Formatting based on magnitude
                let formatted;
                if (Math.abs(target) >= 1000) {{
                    formatted = current.toLocaleString('en-IN', {{ maximumFractionDigits: 0 }});
                }} else if (Math.abs(target) < 10) {{
                    formatted = current.toFixed(2);
                }} else {{
                    formatted = current.toFixed(1);
                }}
                if (progress < 1) {{
                    el.textContent = prefix + formatted + suffix;
                }}
            }});

            if (progress < 1) {{
                requestAnimationFrame(animate);
            }}
        }}
        requestAnimationFrame(animate);
    }})();
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)
