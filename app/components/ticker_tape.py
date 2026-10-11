"""
app/components/ticker_tape.py — Marquee ticker tape of symbols, prices, signed changes.
CSS translateX marquee (60s linear infinite, pauses on hover).
No gradients, flat 1px borders, tabular monospace figures.
"""

from __future__ import annotations

import streamlit.components.v1 as components


def render_ticker_tape(quotes: list[dict], height: int = 34) -> None:
    """
    Render single-line marquee ticker tape.
    quotes: list of {'symbol': str, 'price': float, 'change_pct': float}
    """
    if not quotes:
        quotes = [
            {"symbol": "NIFTY 50", "price": 24850.30, "change_pct": 0.45},
            {"symbol": "S&P 500", "price": 5751.10, "change_pct": -0.22},
            {"symbol": "NASDAQ", "price": 18276.60, "change_pct": 0.31},
            {"symbol": "RELIANCE.NS", "price": 2980.50, "change_pct": 1.15},
            {"symbol": "TCS.NS", "price": 4210.00, "change_pct": -0.38},
            {"symbol": "HDFCBANK.NS", "price": 1665.20, "change_pct": 0.52},
            {"symbol": "INFY.NS", "price": 1895.40, "change_pct": -0.85},
            {"symbol": "AAPL", "price": 229.04, "change_pct": 0.72},
            {"symbol": "NVDA", "price": 134.80, "change_pct": 2.10},
        ]

    # Generate items twice to create seamless loop
    items_html = []
    for _ in range(2):
        for q in quotes:
            sym = q.get("symbol", "")
            price = q.get("price", 0.0)
            chg = q.get("change_pct", 0.0)
            pos = chg >= 0
            cls = "pos" if pos else "neg"
            sign = "+" if pos else ""
            glyph = "▲" if pos else "▼"
            item = f"""
            <span class="ticker-item">
                <span class="ticker-sym">{sym}</span>
                <span class="ticker-price">{price:,.2f}</span>
                <span class="ticker-chg {cls}"><span class="glyph">{glyph}</span> {sign}{chg:.2f}%</span>
            </span>
            """
            items_html.append(item)

    all_items = "\n".join(items_html)

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
        .ticker-container {{
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            background: #0A0A0A;
            border-top: 1px solid #1F1F1F;
            border-bottom: 1px solid #1F1F1F;
            padding: 5px 0;
            display: flex;
        }}
        .ticker-track {{
            display: inline-flex;
            animation: marquee 60s linear infinite;
        }}
        .ticker-track:hover {{
            animation-play-state: paused;
        }}
        .ticker-item {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            margin-right: 36px;
            font-family: 'JetBrains Mono', monospace;
            font-variant-numeric: tabular-nums;
            font-size: 11px;
        }}
        .ticker-sym {{
            color: #F5F5F5;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .ticker-price {{
            color: #8A8A8A;
        }}
        .ticker-chg {{
            font-weight: 500;
        }}
        .ticker-chg.pos {{ color: #22C55E; }}
        .ticker-chg.neg {{ color: #EF4444; }}
        .glyph {{ font-size: 8px; }}
        @keyframes marquee {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .ticker-track {{ animation: none !important; }}
        }}
    </style>
    </head>
    <body>
    <div class="ticker-container">
        <div class="ticker-track">
            {all_items}
        </div>
    </div>
    </body>
    </html>
    """
    components.html(html_code, height=height)
