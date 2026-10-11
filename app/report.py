"""
app/report.py — PDF export using fpdf2.
Generates a multilingual PDF with portfolio table, risk label,
key metrics, recommended allocation, disclaimer.
Charts are matplotlib PNGs (no kaleido needed).
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import streamlit as st

from ml.i18n import t


def _safe_import():
    """Try to import fpdf2."""
    try:
        from fpdf import FPDF
        return FPDF
    except ImportError:
        return None


def generate_pdf(
    lang: str,
    tickers: list[str],
    weights: dict[str, float],
    risk_label: str,
    metrics: dict,
    rec_weights: dict,
    explanation: str,
    session_name: str = "Portfolio",
) -> bytes | None:
    """Generate a PDF report and return bytes. Returns None on error."""
    FPDF = _safe_import()
    if FPDF is None:
        st.error("fpdf2 not installed. Run: pip install fpdf2")
        return None

    try:
        from fpdf import FPDF as _FPDF

        class PDF(_FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 14)
                self.set_text_color(26, 32, 53)
                self.cell(0, 10, "Investment Risk Analysis Report", ln=True, align="C")
                self.set_font("Helvetica", "", 9)
                self.set_text_color(100, 116, 139)
                self.cell(0, 6, session_name, ln=True, align="C")
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(148, 163, 184)
                self.cell(0, 10, t("footer", "en"), align="C")

        pdf = PDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)

        # Try to add Devanagari font for Hindi/Marathi
        font_dir = Path(__file__).parent.parent / "assets" / "fonts"
        devanagari_font = font_dir / "NotoSansDevanagari-Regular.ttf"
        has_devanagari = devanagari_font.exists()
        if has_devanagari:
            pdf.add_font("Devanagari", "", str(devanagari_font))
            pdf.add_font("Devanagari", "B", str(devanagari_font))

        use_devanagari = lang in ("hi", "mr") and has_devanagari

        def _set_body_font():
            if use_devanagari:
                pdf.set_font("Devanagari", size=10)
            else:
                pdf.set_font("Helvetica", size=10)

        def _set_heading_font():
            if use_devanagari:
                pdf.set_font("Devanagari", "B", 12)
            else:
                pdf.set_font("Helvetica", "B", 12)

        # ----------------------------------------------------------------
        # Section 1: Risk Label
        # ----------------------------------------------------------------
        _set_heading_font()
        pdf.set_text_color(26, 32, 53)
        pdf.cell(0, 8, t("risk_label", lang if use_devanagari else "en"), ln=True)

        RISK_COLORS_RGB = {
            "Low":    (46, 204, 113),
            "Medium": (243, 156, 18),
            "High":   (231, 76, 60),
        }
        rc = RISK_COLORS_RGB.get(risk_label, (100, 116, 139))
        pdf.set_fill_color(*rc)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(40, 12, risk_label.upper(), ln=True, fill=True, align="C")
        pdf.set_text_color(26, 32, 53)
        pdf.ln(4)

        # ----------------------------------------------------------------
        # Section 2: Key Metrics
        # ----------------------------------------------------------------
        _set_heading_font()
        pdf.cell(0, 8, "मुख्य मेट्रिक्स" if use_devanagari else "Key Metrics", ln=True)
        pdf.ln(2)

        _set_body_font()
        metric_rows = [
            (t("metric_return", lang if use_devanagari else "en"),   f"{metrics.get('ann_return', 0)*100:.1f}%"),
            (t("metric_vol", lang if use_devanagari else "en"),      f"{metrics.get('ann_vol', 0)*100:.1f}%"),
            (t("metric_drawdown", lang if use_devanagari else "en"), f"{metrics.get('max_drawdown', 0)*100:.1f}%"),
            (t("metric_sharpe", lang if use_devanagari else "en"),   f"{metrics.get('sharpe', 0):.2f}"),
            (t("metric_sortino", lang if use_devanagari else "en"),  f"{metrics.get('sortino', 0):.2f}"),
            (t("metric_cvar", lang if use_devanagari else "en"),     f"{metrics.get('cvar_95', 0)*100:.1f}%"),
        ]
        col_w = 90
        for label_text, value_text in metric_rows:
            _set_body_font()
            pdf.cell(col_w, 7, label_text)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(col_w, 7, value_text, ln=True)
        pdf.ln(4)

        # ----------------------------------------------------------------
        # Section 3: Portfolio Allocation
        # ----------------------------------------------------------------
        _set_heading_font()
        pdf.cell(0, 8, t("original_weights", lang if use_devanagari else "en"), ln=True)
        pdf.ln(2)

        total_w = sum(weights.values()) or 1.0
        _set_body_font()
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(238, 242, 247)
        pdf.set_text_color(26, 32, 53)
        pdf.cell(80, 7, "Ticker", fill=True, border=1)
        pdf.cell(50, 7, "Weight (%)", fill=True, border=1)
        pdf.cell(50, 7, "Amount", fill=True, border=1, ln=True)

        pdf.set_font("Helvetica", "", 10)
        for tkr in tickers:
            w = weights.get(tkr, 0) / total_w
            pdf.cell(80, 7, tkr, border=1)
            pdf.cell(50, 7, f"{w*100:.1f}%", border=1)
            pdf.cell(50, 7, f"INR {w*1e5:,.0f}", border=1, ln=True)
        pdf.ln(4)

        # ----------------------------------------------------------------
        # Section 4: Recommended Allocation
        # ----------------------------------------------------------------
        if rec_weights:
            _set_heading_font()
            pdf.cell(0, 8, t("recommended_weights", lang if use_devanagari else "en"), ln=True)
            pdf.ln(2)
            total_rec = sum(rec_weights.values()) or 1.0
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(238, 242, 247)
            pdf.cell(80, 7, "Ticker", fill=True, border=1)
            pdf.cell(50, 7, "Rec. Weight (%)", fill=True, border=1, ln=True)
            pdf.set_font("Helvetica", "", 10)
            for tkr, w in rec_weights.items():
                w_norm = w / total_rec
                pdf.cell(80, 7, tkr, border=1)
                pdf.cell(50, 7, f"{w_norm*100:.1f}%", border=1, ln=True)
            pdf.ln(4)

        # ----------------------------------------------------------------
        # Section 5: Explanation
        # ----------------------------------------------------------------
        if explanation:
            _set_heading_font()
            pdf.cell(0, 8, t("explanation", lang if use_devanagari else "en"), ln=True)
            pdf.ln(2)
            _set_body_font()
            if use_devanagari:
                safe_expl = explanation
            else:
                safe_expl = explanation.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, safe_expl)
            pdf.ln(4)

        # ----------------------------------------------------------------
        # Footer disclaimer already in footer() method
        # ----------------------------------------------------------------

        buf = io.BytesIO()
        pdf.output(buf)
        return buf.getvalue()

    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None


def export_button(lang: str, tickers: list, weights: dict, risk_label: str,
                  metrics: dict, rec_weights: dict, explanation: str,
                  session_name: str = "Portfolio") -> None:
    """Render an Export PDF download button."""
    if st.button(t("export_pdf", lang), key="export_pdf_btn"):
        with st.spinner("Generating PDF…"):
            pdf_bytes = generate_pdf(
                lang, tickers, weights, risk_label, metrics,
                rec_weights, explanation, session_name
            )
        if pdf_bytes:
            st.download_button(
                label=f"DOWNLOAD {session_name.upper()}.PDF",
                data=pdf_bytes,
                file_name=f"{session_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                key="download_pdf_btn",
            )
