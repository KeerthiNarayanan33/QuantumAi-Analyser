"""
============================================================
QuantumSentinel — Report Generator
============================================================
Generates a downloadable PDF summary report containing:
  • Prediction results
  • Sentiment breakdown
  • Model accuracy comparison
  • Top contributing features
============================================================
"""

import os
import io
from datetime import datetime

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

import pandas as pd


# ────────────────────────────────────────────────────────────
# PDF REPORT
# ────────────────────────────────────────────────────────────
class QuantumSentinelReport:
    """Build a professional PDF report."""

    def __init__(self):
        if not FPDF_AVAILABLE:
            raise ImportError("fpdf2 not installed. Run: pip install fpdf2")
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        self._set_header()

    def _set_header(self):
        pdf = self.pdf
        # ── Title block ──────────────────────────────────────
        pdf.set_fill_color(15, 23, 42)      # dark navy
        pdf.rect(0, 0, 210, 40, "F")
        pdf.set_text_color(99, 179, 237)    # cyber blue
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 15, "QuantumSentinel", ln=True, align="C")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, "Quantum Analytics for Investor Behaviour & Market Sentiment", ln=True, align="C")
        pdf.set_text_color(150, 200, 255)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(8)
        pdf.set_text_color(0, 0, 0)

    def add_prediction_section(self, predictions: list[dict]):
        pdf = self.pdf
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_fill_color(230, 240, 255)
        pdf.cell(0, 10, "  Investor Behaviour Predictions", ln=True, fill=True)
        pdf.ln(3)

        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        for header, w in [("Ticker",30), ("Signal",30), ("Confidence",40), ("Model",80)]:
            pdf.cell(w, 8, header, border=1, fill=True, align="C")
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 9)
        colors = {"Buy": (198, 246, 213), "Hold": (254, 243, 199), "Sell": (254, 215, 215)}
        for pred in predictions:
            signal = pred.get("signal_name", "N/A")
            fill_c = colors.get(signal, (255,255,255))
            pdf.set_fill_color(*fill_c)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(30, 8, pred.get("ticker", ""), border=1, fill=True, align="C")
            pdf.cell(30, 8, signal, border=1, fill=True, align="C")
            conf = f"{pred.get('confidence', 0)*100:.1f}%"
            pdf.cell(40, 8, conf, border=1, fill=True, align="C")
            pdf.cell(80, 8, pred.get("model_used","")[:30], border=1, fill=True)
            pdf.ln()
        pdf.ln(5)

    def add_sentiment_section(self, news_df: pd.DataFrame):
        pdf = self.pdf
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_fill_color(230, 255, 240)
        pdf.cell(0, 10, "  Sentiment Analysis Summary", ln=True, fill=True)
        pdf.ln(3)

        if "label" in news_df.columns:
            dist = news_df["label"].value_counts()
            total = len(news_df)
            pdf.set_font("Helvetica", "", 10)
            for label in ["positive", "neutral", "negative"]:
                count = dist.get(label, 0)
                pct   = count / total * 100 if total > 0 else 0
                bar_w = int(pct * 1.2)
                icon  = "🟢" if label=="positive" else ("🔴" if label=="negative" else "🟡")
                pdf.cell(40, 8, f"{label.capitalize()}", border=0)
                pdf.cell(20, 8, f"{count} ({pct:.0f}%)", border=0)
                pdf.ln()
        pdf.ln(5)

    def add_model_comparison(self, qsvc_acc: float, svm_acc: float):
        pdf = self.pdf
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_fill_color(255, 240, 230)
        pdf.cell(0, 10, "  Model Accuracy Comparison", ln=True, fill=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(80, 10, "QSVC (Quantum SVM):", border=0)
        pdf.cell(0, 10, f"{qsvc_acc*100:.1f}%", ln=True)
        pdf.cell(80, 10, "Classical SVM (RBF):", border=0)
        pdf.cell(0, 10, f"{svm_acc*100:.1f}%", ln=True)
        pdf.ln(5)

    def add_footer(self):
        pdf = self.pdf
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8,
                 "QuantumSentinel | Hackathon Project | Powered by Qiskit + FinBERT | "
                 "Not financial advice", ln=True, align="C")

    def generate(self) -> bytes:
        """Return PDF as bytes for Streamlit download."""
        self.add_footer()
        return bytes(self.pdf.output())


# ────────────────────────────────────────────────────────────
# CSV REPORT
# ────────────────────────────────────────────────────────────
def generate_csv_report(predictions: list[dict],
                         news_df: pd.DataFrame) -> bytes:
    """Generate a CSV report for download."""
    pred_rows = []
    for p in predictions:
        row = {
            "Ticker"    : p.get("ticker"),
            "Signal"    : p.get("signal_name"),
            "Confidence": f"{p.get('confidence',0)*100:.1f}%",
            "Sell_prob" : p.get("probabilities", {}).get("Sell", 0),
            "Hold_prob" : p.get("probabilities", {}).get("Hold", 0),
            "Buy_prob"  : p.get("probabilities", {}).get("Buy", 0),
            "Model"     : p.get("model_used"),
        }
        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    output  = io.StringIO()
    pred_df.to_csv(output, index=False)

    # Append sentiment summary
    if "label" in news_df.columns:
        output.write("\n\nSentiment Summary\n")
        news_df["label"].value_counts().to_csv(output)

    return output.getvalue().encode()


# ────────────────────────────────────────────────────────────
# QUICK TEST
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_preds = [
        {"ticker": "AAPL", "signal_name": "Buy",  "confidence": 0.78,
         "probabilities": {"Sell":0.1,"Hold":0.12,"Buy":0.78}, "model_used": "QSVC"},
        {"ticker": "TSLA", "signal_name": "Sell", "confidence": 0.65,
         "probabilities": {"Sell":0.65,"Hold":0.2,"Buy":0.15}, "model_used": "QSVC"},
    ]
    sample_news = pd.DataFrame({
        "label": ["positive","negative","neutral","positive","negative"]
    })

    csv_bytes = generate_csv_report(sample_preds, sample_news)
    print("CSV Report (first 300 chars):")
    print(csv_bytes.decode()[:300])

    if FPDF_AVAILABLE:
        report = QuantumSentinelReport()
        report.add_prediction_section(sample_preds)
        report.add_sentiment_section(sample_news)
        report.add_model_comparison(0.72, 0.68)
        pdf_bytes = report.generate()
        print(f"\nPDF Report generated: {len(pdf_bytes)} bytes")
