"""
============================================================
QuantumSentinel — Predictor
============================================================
Loads trained models and runs the full inference pipeline
for a given ticker + optional custom headline.

Outputs:
  • Signal        : Buy / Hold / Sell
  • Confidence    : probability of the predicted class
  • Probabilities : [Sell%, Hold%, Buy%]
  • Feature values used for the prediction
============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── Add src to path ─────────────────────────────────────────
SRC_DIR = os.path.dirname(__file__)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SRC_DIR, "..")))

from data_collector     import get_combined_stock_df, load_news  # type: ignore[import]
from sentiment_analyzer import safe_analyze_news_df  # type: ignore[import]
from feature_engineer   import (merge_stock_sentiment, generate_labels,  # type: ignore[import]
                                 build_feature_matrix, FEATURE_COLS)

# ── Load centralized config ────────────────────────────────────
from config import cfg  # type: ignore[import]

# ── Paths (sourced from config) ─────────────────────────────
MODEL_DIR   = str(cfg.model.MODEL_DIR)
QSVC_PATH   = str(cfg.model.QSVC_PATH)
SVM_PATH    = str(cfg.model.SVM_PATH)
PCA_PATH    = str(cfg.model.PCA_PATH)
SCALER_PATH = str(cfg.model.SCALER_PATH)

# ── Label mappings (sourced from config) ──────────────────────
SIGNAL_MAP  = cfg.model.SIGNAL_MAP
SIGNAL_ICON = cfg.model.SIGNAL_ICON


# ────────────────────────────────────────────────────────────
# MODEL LOADER
# ────────────────────────────────────────────────────────────
def load_models() -> dict:
    """
    Load all persisted model artefacts.

    Returns
    -------
    dict with: qsvc, svm, pca, scaler
    (returns None values if a model file doesn't exist yet)
    """
    models = {}
    for name, path in [("qsvc",   QSVC_PATH),
                       ("svm",    SVM_PATH),
                       ("pca",    PCA_PATH),
                       ("scaler", SCALER_PATH)]:
        try:
            models[name] = joblib.load(path)
            print(f"[LOAD] {name} loaded from {path}")
        except FileNotFoundError:
            models[name] = None
            print(f"[WARN] {name} not found at {path} — run train.py first.")
    return models


# ────────────────────────────────────────────────────────────
# FEATURE PREPARATION
# ────────────────────────────────────────────────────────────
def prepare_features_for_ticker(ticker: str,
                                 stock_df: pd.DataFrame,
                                 news_df_with_sentiment: pd.DataFrame,
                                 scaler,
                                 pca,
                                 pca_rescaler=None) -> np.ndarray:
    """
    Build and transform the feature vector for the LATEST day of a ticker.

    Parameters
    ----------
    ticker                  : stock ticker symbol
    stock_df                : full combined stock DataFrame
    news_df_with_sentiment  : news DataFrame with sentiment columns added
    scaler                  : fitted MinMaxScaler from training
    pca                     : fitted PCA from training
    pca_rescaler            : second MinMaxScaler applied after PCA

    Returns
    -------
    X_ready : 2D array  shape (1, N_QUBITS) — ready for QSVC/SVM prediction
    raw_features : dict — human-readable feature values for explainability
    """
    merged = merge_stock_sentiment(stock_df, news_df_with_sentiment, ticker)
    merged = generate_labels(merged)
    df     = merged.dropna(subset=FEATURE_COLS)

    if df.empty:
        raise ValueError(f"No valid feature rows for {ticker}.")

    # Take the LAST available row (most recent trading day)
    latest_row = df.iloc[[-1]]
    raw_values = latest_row[FEATURE_COLS].values.astype(np.float64)

    # ── Apply same scaler used in training ───────────────────
    if scaler is not None:
        raw_scaled = scaler.transform(raw_values)
    else:
        raw_scaled = raw_values

    # ── Apply PCA ────────────────────────────────────────────
    if pca is not None:
        X_pca = pca.transform(raw_scaled)
    else:
        X_pca = raw_scaled[:, :4]   # take first 4 dims as fallback

    # ── Re-scale PCA output to [0, π] ────────────────────────
    if pca_rescaler is not None:
        X_ready = pca_rescaler.transform(X_pca)
    else:
        X_ready = X_pca

    # Build human-readable feature dict for explainability
    raw_features = {
        col: float(latest_row[col].iloc[0])
        for col in FEATURE_COLS
    }

    return X_ready, raw_features


# ────────────────────────────────────────────────────────────
# MAIN PREDICT FUNCTION
# ────────────────────────────────────────────────────────────
def predict(ticker: str,
            models: dict,
            stock_df: pd.DataFrame | None  = None,
            news_df: pd.DataFrame | None   = None,
            use_qsvc: bool                 = True) -> dict:
    """
    Run the full prediction pipeline for a given ticker.

    Parameters
    ----------
    ticker    : e.g. "AAPL"
    models    : dict from load_models()
    stock_df  : pre-loaded stock data (fetched if None)
    news_df   : news DataFrame WITH sentiment columns (fetched if None)
    use_qsvc  : True = use Quantum SVC, False = classical SVM

    Returns
    -------
    dict with:
        ticker, signal, signal_name, signal_icon,
        confidence, probabilities,
        raw_features, model_used
    """
    # ── Lazy-load data if not provided ───────────────────────
    if stock_df is None:
        stock_df = get_combined_stock_df([ticker])
    if news_df is None:
        raw_news = load_news()
        news_df  = safe_analyze_news_df(raw_news)

    # ── Choose model ─────────────────────────────────────────
    model      = models.get("qsvc") if use_qsvc else models.get("svm")
    model_used = "QSVC (Quantum)" if use_qsvc else "Classical SVM"

    # ── Fallback if requested model not available ─────────────
    if model is None:
        model      = models.get("svm")
        model_used = "Classical SVM (fallback)"
        if model is None:
            return _rule_based_fallback(ticker, news_df)

    # ── Prepare features ─────────────────────────────────────
    try:
        X_ready, raw_features = prepare_features_for_ticker(
            ticker, stock_df, news_df,
            models.get("scaler"), models.get("pca")
        )
    except Exception as e:
        print(f"[ERR] Feature prep failed: {e}. Using fallback.")
        return _rule_based_fallback(ticker, news_df)

    # ── Predict ──────────────────────────────────────────────
    signal = int(model.predict(X_ready)[0])

    # Get probabilities (QSVC via decision_function, SVM via predict_proba)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_ready)[0].tolist()
        else:
            # Convert decision function scores to pseudo-probabilities
            df_scores = model.decision_function(X_ready)[0]
            exp_scores = np.exp(df_scores - np.max(df_scores))
            probs = (exp_scores / exp_scores.sum()).tolist()
    except Exception:
        # Uniform fallback probabilities
        probs = [0.33, 0.34, 0.33]
        probs[signal] = 0.6

    confidence = float(probs[signal])

    return {
        "ticker"       : ticker,
        "signal"       : signal,
        "signal_name"  : SIGNAL_MAP[signal],
        "signal_icon"  : SIGNAL_ICON[signal],
        "confidence"   : round(confidence, 4),
        "probabilities": {
            "Sell" : round(probs[0], 4),
            "Hold" : round(probs[1], 4),
            "Buy"  : round(probs[2], 4),
        },
        "raw_features" : raw_features,
        "model_used"   : model_used,
    }


def _rule_based_fallback(ticker: str, news_df: pd.DataFrame) -> dict:
    """
    Simple rule-based prediction when no trained model is available.
    Uses average sentiment compound score.
    """
    if "compound" in news_df.columns:
        avg_compound = news_df[news_df.get("ticker", pd.Series()) == ticker]["compound"].mean()
        if pd.isna(avg_compound):
            avg_compound = news_df["compound"].mean()
    else:
        avg_compound = 0.0

    if avg_compound > 0.1:
        signal = 2   # Buy
    elif avg_compound < -0.1:
        signal = 0   # Sell
    else:
        signal = 1   # Hold

    confidence = min(0.5 + abs(avg_compound), 0.9)

    return {
        "ticker"       : ticker,
        "signal"       : signal,
        "signal_name"  : SIGNAL_MAP[signal],
        "signal_icon"  : SIGNAL_ICON[signal],
        "confidence"   : round(confidence, 4),
        "probabilities": {
            "Sell": round(0.2 if signal != 0 else confidence, 4),
            "Hold": round(0.2 if signal != 1 else confidence, 4),
            "Buy" : round(0.2 if signal != 2 else confidence, 4),
        },
        "raw_features" : {"sentiment_compound": avg_compound},
        "model_used"   : "Rule-based fallback",
    }


def predict_batch(tickers: list[str], models: dict,
                  stock_df=None, news_df=None) -> list[dict]:
    """Run predictions for multiple tickers."""
    results = []
    for t in tickers:
        try:
            r = predict(t, models, stock_df, news_df)
            results.append(r)
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    return results


# ────────────────────────────────────────────────────────────
# QUICK TEST
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  QuantumSentinel — Predictor Test")
    print("=" * 55)

    mods = load_models()
    result = predict("AAPL", mods)

    print(f"\n  Ticker     : {result['ticker']}")
    print(f"  Signal     : {result['signal_icon']} {result['signal_name']}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"  Model used : {result['model_used']}")
    print(f"  Probs      : {result['probabilities']}")
