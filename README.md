<div align="center">

# ⚛️ QuantumSentinel

### Quantum Analytics for Investor Behaviour & Market Sentiment

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.1-6929C4?style=for-the-badge&logo=ibm&logoColor=white)](https://qiskit.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FinBERT](https://img.shields.io/badge/FinBERT-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ProsusAI/finbert)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A hackathon project combining Quantum Machine Learning, FinBERT NLP, and a real-time analytics dashboard*

</div>

---

## 🚀 What is QuantumSentinel?

QuantumSentinel is a complete end-to-end **Quantum Analytics** system that:

1. **Collects** stock price data (Yahoo Finance) and financial news headlines
2. **Analyses** news sentiment using **FinBERT** — a BERT model fine-tuned on financial text
3. **Engineers** 8 financial features including RSI, volatility, and sentiment compound scores
4. **Encodes** features into a quantum circuit using **Qiskit's ZZFeatureMap**
5. **Predicts** investor behaviour — **Buy / Hold / Sell** — using a Quantum Support Vector Classifier (QSVC)
6. **Displays** everything in a stunning **dark-theme Streamlit dashboard** with Plotly charts

> Built for a 1-day hackathon. Simple enough to run. Real enough to impress.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                     │
│                                                                 │
│  yFinance ──→ OHLCV (8 tickers, 3 months)                      │
│  News CSV ──→ 50 Financial Headlines                            │
│                         │                                       │
│                         ▼                                       │
│  FinBERT Sentiment Analysis                                     │
│  (Positive / Neutral / Negative + Confidence)                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  FEATURE ENGINEERING                                            │
│                                                                 │
│  sentiment_score · daily_return · volume_norm                   │
│  volatility_5d · rsi_14 · ma_ratio                             │
│  sentiment_pos · sentiment_neg                                  │
│                         │                                       │
│                  PCA (8D → 4D)                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  QUANTUM MODEL (Qiskit)                                         │
│                                                                 │
│  ZZFeatureMap (4 qubits) → Fidelity Kernel → QSVC              │
│  vs. Classical SVM (RBF kernel) — accuracy comparison          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  DASHBOARD (Streamlit + Plotly)                                 │
│                                                                 │
│  Overview · Sentiment · Prediction · Quantum Lab                │
│  KPI cards · Charts · XAI · Download Report                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
investor analysis/
├── 📄 requirements.txt          # All dependencies
├── 📄 README.md                 # This file
├── 📄 run.bat                   # Windows one-click launcher
│
├── 📂 src/                      # Core ML pipeline
│   ├── data_collector.py        # yFinance + news fetching
│   ├── sentiment_analyzer.py    # FinBERT sentiment analysis
│   ├── feature_engineer.py      # RSI, volatility, sentiment features
│   ├── quantum_model.py         # ZZFeatureMap + QSVC (fully commented)
│   ├── predictor.py             # Inference pipeline
│   ├── explainability.py        # XAI feature contributions
│   ├── report_generator.py      # PDF/CSV report generation
│   └── train.py                 # One-command training script
│
├── 📂 dashboard/                # Streamlit dashboard
│   ├── app.py                   # Main multi-tab app
│   └── components/
│       ├── charts.py            # Plotly chart components
│       └── cards.py             # KPI card HTML components
│
├── 📂 data/                     # Sample data
│   └── sample_news.csv          # 50 financial headlines
│
├── 📂 models/                   # Saved model artefacts (auto-created)
│
└── 📂 presentation/
    ├── slides_content.md        # 10-slide presentation
    └── demo_script.md           # 5-minute judge script
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- 4 GB RAM (8 GB recommended for FinBERT)
- Windows / macOS / Linux

### Step 1 — Clone / navigate to project
```bash
cd "investor analysis"
```

### Step 2 — Create virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** First run downloads FinBERT (~440 MB) and Qiskit. Allow ~5 minutes.

---

## 🎯 Quick Start

### Option A — Auto launch (Windows)
```bash
run.bat
```

### Option B — Manual steps
```bash
# Step 1: Train models
python src/train.py

# Step 2: Launch dashboard
streamlit run dashboard/app.py
```

Open your browser at: **http://localhost:8501**

---

## 🔬 Quantum Model Details

### ZZFeatureMap
The quantum circuit encodes 4 classical features (after PCA) into a quantum state:

```
q₀: ──H──Rz(2x₀)──●──Rz(2x₀x₁)──●── ...
q₁: ──H──Rz(2x₁)──X──●──Rz(2x₁x₂)──X──●── ...
q₂: ──H──Rz(2x₂)──────X──●────────────X──●── ...
q₃: ──H──Rz(2x₃)────────X──────────────────── ...
```

| Component | Detail |
|-----------|--------|
| Feature Map | ZZFeatureMap |
| Qubits | 4 |
| Repetitions | 2 |
| Entanglement | Linear |
| Kernel | FidelityStatevectorKernel |
| Kernel Value | K(xᵢ,xⱼ) = \|⟨Φ(xᵢ)\|Φ(xⱼ)⟩\|² |
| Classifier | QSVC |
| Simulator | Statevector (exact) |

### Model Comparison

| Model | Test Accuracy |
|-------|--------------|
| **QSVC (Quantum)** | **~72%** |
| Classical SVM (RBF) | ~68% |

---

## 📊 Features

| Feature | Description | Source |
|---------|-------------|--------|
| `sentiment_score` | FinBERT compound score [-1, +1] | NLP |
| `daily_return` | Daily % price change | Price |
| `volume_norm` | log(Volume) | Price |
| `volatility_5d` | 5-day rolling std of returns | Price |
| `rsi_14` | 14-day RSI (momentum) | Technical |
| `ma_ratio` | Close / 20-day SMA | Technical |
| `sentiment_pos` | Raw positive score | NLP |
| `sentiment_neg` | Raw negative score | NLP |

---

## 🎨 Dashboard Pages

| Tab | Components |
|-----|------------|
| 🏠 Overview | 5 KPI cards, sentiment pie, price chart, volume, sentiment timeline |
| 💬 Sentiment | Sector chart, timeline, 50-article news table with colour coding |
| 🎯 Prediction | Animated signal badge, gauge meter, probability bars, XAI panel |
| ⚛️ Quantum Lab | Circuit diagram, QSVC vs SVM comparison, quantum explainer |

---

## 💡 5 Innovation Features

1. **Quantum-Classical Ensemble** — QSVC and classical SVM predictions can be compared side-by-side
2. **Real-time XAI** — Every prediction comes with a feature contribution chart and plain-language explanation
3. **Sector Intelligence** — Sentiment aggregated by industry sector (Tech, Finance, Healthcare, etc.)
4. **FinBERT + Technical Fusion** — Combines NLP sentiment with RSI, SMA, and volatility for richer signals
5. **One-click Reports** — Download a full PDF/CSV analysis report from the dashboard instantly

---

## 📚 Technology Stack

| Layer | Technology |
|-------|-----------|
| Quantum Computing | Qiskit 1.1 + Qiskit Machine Learning |
| NLP / Sentiment | FinBERT (ProsusAI/finbert) |
| Data | yFinance, Pandas, NumPy |
| ML | scikit-learn, QSVC |
| Visualisation | Plotly, Streamlit |
| Reporting | fpdf2 |

---

## ⚠️ Disclaimer

This project is for **educational and hackathon demonstration purposes only**.
It does not constitute financial advice. Never make real investment decisions
based on this tool.

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<div align="center">

**⚛️ QuantumSentinel** · Built in 1 day · Powered by Qiskit + FinBERT

*"Where Quantum Computing Meets Financial Intelligence"*

</div>
