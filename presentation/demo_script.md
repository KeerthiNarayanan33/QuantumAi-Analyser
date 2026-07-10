# QuantumSentinel — 5-Minute Demo Script

**For Hackathon Judges**

---

## ⏱️ TIMING GUIDE

| Section | Time |
|---------|------|
| Introduction | 0:00 – 0:45 |
| Problem + Solution | 0:45 – 1:30 |
| Live Dashboard Demo | 1:30 – 3:30 |
| Quantum Deep Dive | 3:30 – 4:15 |
| Closing | 4:15 – 5:00 |

---

## 🎤 FULL SCRIPT

---

### [0:00 – 0:45] INTRODUCTION

> *"Good [morning/afternoon], judges. My name is [Name], and we are Team [Name].*
>
> *Every second, thousands of financial news articles flood the internet. Markets move before investors can even read the headlines. The question is — can we use AI and quantum computing to give investors an edge? That's exactly what QuantumSentinel does."*

---

### [0:45 – 1:30] PROBLEM + SOLUTION

> *"The problem is clear: investors face an impossible amount of data. Traditional models can't capture the complex correlations between news sentiment and price momentum.*
>
> *Our solution: QuantumSentinel — a complete analytics pipeline that:*
> - *Collects financial news from Yahoo Finance and our news dataset*
> - *Runs FinBERT — a state-of-the-art financial NLP model — to classify headlines as Positive, Neutral, or Negative*
> - *Engineers 8 financial features including RSI, volatility, and sentiment scores*
> - *Encodes these features into a quantum circuit using Qiskit's ZZFeatureMap*
> - *And trains a Quantum Support Vector Classifier to predict Buy, Hold, or Sell signals"*

---

### [1:30 – 3:30] LIVE DASHBOARD DEMO

*[Open the dashboard on screen: `streamlit run dashboard/app.py`]*

#### Overview Tab [30 sec]
> *"Here's our live dashboard. At the top, you can see five KPI cards — market sentiment is [Positive/Negative], the average FinBERT compound score is [X], and AAPL is currently showing a [Buy/Hold/Sell] signal with [X]% confidence.*
>
> *On the left, you can see our sentiment pie chart — [X]% of today's news is positive. On the right, the AAPL candlestick chart with the 20-day moving average overlay."*

#### Sentiment Tab [30 sec]
> *"In the Sentiment tab, we see the timeline of news sentiment over time — notice how it dips during the Tesla earnings miss. Below that is the sector-wise breakdown — Technology is the most bullish sector, while Healthcare is slightly bearish.*
>
> *And here's our news table — 50 articles analysed by FinBERT in real time. Green means positive, red means negative."*

#### Prediction Tab [40 sec]
> *"This is the most exciting part — our Prediction tab. For AAPL, the model is showing a [BUY] signal with [78]% confidence. The gauge meter on the right confirms this.*
>
> *Below, you can see our Explainable AI panel. The bar chart shows WHY the model made this decision — Sentiment Score and RSI are the top contributing factors. And in plain language: [read explanation aloud].*
>
> *For judges who want the data — here's the download button. One click gives you a complete PDF report."*

#### Change Ticker [20 sec]
> *"Let me switch to Tesla in the sidebar — [select TSLA] — and you can see the signal changes to [Sell/Hold] based on the lower sentiment and higher volatility. The system is fully dynamic."*

---

### [3:30 – 4:15] QUANTUM DEEP DIVE

*[Switch to Quantum Lab tab]*

> *"Let me explain the quantum component — this is what sets us apart.*
>
> *We use Qiskit's ZZFeatureMap with 4 qubits. Each qubit encodes one feature after PCA dimensionality reduction. The Hadamard gates put qubits into superposition. The Rz rotation gates encode each feature as a quantum angle. Then — and this is the key — the ZZ interactions between qubits create entanglement. This captures correlations between features that a classical RBF kernel simply cannot represent.*
>
> *Our quantum kernel computes: K(xᵢ, xⱼ) = |⟨Φ(xᵢ)|Φ(xⱼ)⟩|² — the fidelity between quantum states.*
>
> *The result? QSVC achieves [72]% accuracy versus [68]% for the classical SVM. A [+4]% quantum advantage. In financial prediction, that margin is meaningful."*

---

### [4:15 – 5:00] CLOSING

> *"To summarise — QuantumSentinel is a complete, end-to-end quantum analytics platform. In one day, we built:*
>
> *✅ A FinBERT sentiment pipeline*
> *✅ A real Qiskit quantum circuit with entanglement*
> *✅ A QSVC model that outperforms classical SVM*
> *✅ A stunning dark-theme dashboard with XAI and download reports*
>
> *The future? We'll connect to live news APIs, run on real IBM quantum hardware, and add portfolio optimisation using quantum annealing.*
>
> *QuantumSentinel proves that quantum computing isn't just a research topic — it's a working tool for financial intelligence, built by college students in a single day.*
>
> *Thank you. We're happy to answer questions."*

---

## 💡 JUDGE Q&A — QUICK ANSWERS

| Question | Answer |
|----------|--------|
| "Is this real quantum computing?" | Yes — Qiskit ZZFeatureMap with real entanglement. Simulated on classical hardware (Statevector), but the algorithm is genuinely quantum |
| "What's the dataset?" | Yahoo Finance (live OHLCV) + 50 sample financial news headlines |
| "Why only 4 qubits?" | Simulation scales exponentially — 4 qubits = 16-element statevector, fast enough for demo. Real hardware could use more |
| "How is this better than ChatGPT for sentiment?" | FinBERT is specifically trained on financial text — higher accuracy on financial jargon, earnings reports, analyst notes |
| "Could this run on IBM Quantum?" | Yes — replace FidelityStatevectorKernel with a real backend, same code structure works |

---

*Script prepared for a 5-minute hackathon demo. Adjust [placeholders] with your actual results.*
