"""
app/components/risk_gauge.py — Half-circle SVG risk gauge hero component.
Split into three flat segments: Low (#22C55E), Medium (#F59E0B), High (#EF4444).
Animated needle (900ms ease), hero risk word in 56px, and probability bar underneath.
No gradients, no blur, no shadows.
"""

from __future__ import annotations

import streamlit.components.v1 as components


def render_risk_gauge(label: str, probs: dict[str, float], height: int = 210) -> None:
    """
    Render half-circle SVG risk gauge with animated needle and probability breakdown.
    label: 'Low', 'Medium', or 'High'
    probs: {'Low': float, 'Medium': float, 'High': float}
    """
    norm_label = label.capitalize() if label else "Medium"
    color_map = {
        "Low": "#22C55E",
        "Medium": "#F59E0B",
        "High": "#EF4444",
    }
    target_angle_map = {
        "Low": -60,
        "Medium": 0,
        "High": 60,
    }
    active_color = color_map.get(norm_label, "#F59E0B")
    target_angle = target_angle_map.get(norm_label, 0)

    p_low = probs.get("Low", 0.0) * 100
    p_med = probs.get("Medium", 0.0) * 100
    p_high = probs.get("High", 0.0) * 100

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
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .gauge-wrapper {{
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            max-width: 380px;
        }}
        .svg-container {{
            position: relative;
            width: 220px;
            height: 115px;
        }}
        svg {{
            overflow: visible;
        }}
        .needle {{
            transform-origin: 110px 105px;
            transform: rotate({target_angle}deg);
            transition: transform 900ms cubic-bezier(0.2, 0.7, 0.2, 1);
        }}
        .risk-word {{
            font-family: 'Inter', sans-serif;
            font-size: 48px;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: {active_color};
            text-transform: uppercase;
            line-height: 1;
            margin-top: 4px;
        }}
        .prob-row {{
            display: flex;
            justify-content: space-between;
            width: 100%;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #1F1F1F;
        }}
        .prob-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .prob-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8A8A8A;
        }}
        .prob-val {{
            font-family: 'JetBrains Mono', monospace;
            font-variant-numeric: tabular-nums;
            font-size: 13px;
            font-weight: 600;
            color: #F5F5F5;
            margin-top: 2px;
        }}
        @media (prefers-reduced-motion: reduce) {{
            .needle {{ transition: none !important; }}
        }}
    </style>
    </head>
    <body>
    <div class="gauge-wrapper">
        <div class="svg-container">
            <svg width="220" height="115" viewBox="0 0 220 115">
                <!-- Arc Segments (r=85, center=110,105) -->
                <!-- Low segment (180deg to 120deg) -->
                <path d="M 25 105 A 85 85 0 0 1 67.5 31.4" fill="none" stroke="#22C55E" stroke-width="10" stroke-linecap="butt"/>
                <!-- Med segment (120deg to 60deg) -->
                <path d="M 69.5 30.2 A 85 85 0 0 1 150.5 30.2" fill="none" stroke="#F59E0B" stroke-width="10" stroke-linecap="butt"/>
                <!-- High segment (60deg to 0deg) -->
                <path d="M 152.5 31.4 A 85 85 0 0 1 195 105" fill="none" stroke="#EF4444" stroke-width="10" stroke-linecap="butt"/>
                
                <!-- Center pivot -->
                <circle cx="110" cy="105" r="5" fill="#F5F5F5" />
                
                <!-- Needle (line from center upwards) -->
                <g class="needle" id="gauge-needle">
                    <line x1="110" y1="105" x2="110" y2="35" stroke="#F5F5F5" stroke-width="2" stroke-linecap="round"/>
                </g>
            </svg>
        </div>
        <div class="risk-word">{norm_label}</div>
        <div class="prob-row">
            <div class="prob-item">
                <span class="prob-label">Low</span>
                <span class="prob-val">{p_low:.0f}%</span>
            </div>
            <div class="prob-item">
                <span class="prob-label">Medium</span>
                <span class="prob-val">{p_med:.0f}%</span>
            </div>
            <div class="prob-item">
                <span class="prob-label">High</span>
                <span class="prob-val">{p_high:.0f}%</span>
            </div>
        </div>
    </div>
    <script>
    (function() {{
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
        const needle = document.getElementById('gauge-needle');
        if (!needle) return;
        // Start from -90 and rotate to target
        needle.style.transform = 'rotate(-90deg)';
        requestAnimationFrame(() => {{
            setTimeout(() => {{
                needle.style.transform = 'rotate({target_angle}deg)';
            }}, 50);
        }});
    }})();
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)
