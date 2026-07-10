"""
============================================================
QuantumSentinel — Train Script
============================================================
One-command training script. Run this before launching
the dashboard to generate all model artefacts.

Usage:
    python src/train.py

What it does:
  1. Fetches stock data (yfinance) + news (sample CSV)
  2. Runs FinBERT sentiment analysis
  3. Engineers features
  4. Trains QSVC + classical SVM
  5. Saves models to /models/
============================================================
"""

import os
import sys
import time

SRC_DIR = os.path.dirname(__file__)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SRC_DIR, "..")))

from config import cfg  # type: ignore[import]

from data_collector    import get_combined_stock_df, load_news  # type: ignore[import]
from sentiment_analyzer import safe_analyze_news_df  # type: ignore[import]
from feature_engineer   import (build_multi_ticker_dataset, FEATURE_COLS)  # type: ignore[import]
from quantum_model      import train_all  # type: ignore[import]


TICKERS = cfg.data.DEFAULT_TICKERS


def main():
    print("\n" + "=" * 60)
    print("  QuantumSentinel — Training Pipeline")
    print("=" * 60)
    t0 = time.time()

    # ── Step 1: Fetch stock data ────────────────────────────
    print("\n[1/4] Fetching stock data from Yahoo Finance...")
    stock_df = get_combined_stock_df(TICKERS, period=cfg.data.TRAINING_PERIOD)
    print(f"      Total rows: {len(stock_df)}")

    # ── Step 2: Load & analyse news ─────────────────────────
    print("\n[2/4] Running FinBERT sentiment analysis on news...")
    news_raw = load_news()
    news_df  = safe_analyze_news_df(news_raw)
    print(f"      Processed {len(news_df)} articles")
    sentiment_dist = news_df["label"].value_counts().to_dict()
    print(f"      Sentiment: {sentiment_dist}")

    # ── Step 3: Feature engineering ─────────────────────────
    print("\n[3/4] Engineering features across all tickers...")
    X, y, scaler = build_multi_ticker_dataset(stock_df, news_df, TICKERS)
    print(f"      Feature matrix: {X.shape}")
    print(f"      Label dist: Buy={sum(y==2)}, Hold={sum(y==1)}, Sell={sum(y==0)}")

    # Save scaler separately for dashboard use
    import joblib
    cfg.model.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, str(cfg.model.SCALER_PATH))

    # ── Step 4: Train models ─────────────────────────────────
    print("\n[4/4] Training QSVC and classical SVM...")
    results = train_all(X, y, test_size=0.25)

    # ── Summary ─────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  QSVC Accuracy    : {results['qsvc_accuracy']*100:.1f}%")
    print(f"  Classical SVM    : {results['svm_accuracy']*100:.1f}%")
    print(f"  Total time       : {elapsed:.1f}s")
    print(f"  Models saved to  : {os.path.abspath(cfg.model.MODEL_DIR)}")
    print("=" * 60)
    print("\n  [OK] Ready! Launch the dashboard with:")
    print("     streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
