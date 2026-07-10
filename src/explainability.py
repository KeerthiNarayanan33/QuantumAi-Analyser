"""
============================================================
QuantumSentinel — Explainability Module
============================================================
Provides human-interpretable explanations for predictions:

  1. Feature importance via permutation importance
     (model-agnostic — works with QSVC and classical SVM)
  2. Normalised contribution scores per feature
  3. Plain-language explanation generation

This is a lightweight Explainable AI (XAI) implementation
suited for a hackathon demo — no SHAP dependency required.
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# ── Feature metadata ─────────────────────────────────────────
FEATURE_COLS = [
    "sentiment_score",
    "daily_return",
    "volume_norm",
    "volatility_5d",
    "rsi_14",
    "ma_ratio",
    "sentiment_pos",
    "sentiment_neg",
]

FEATURE_DESCRIPTIONS = {
    "sentiment_score" : "News Sentiment Score (FinBERT compound)",
    "daily_return"    : "Daily Price Change (%)",
    "volume_norm"     : "Trading Volume (log-normalised)",
    "volatility_5d"   : "5-Day Rolling Volatility",
    "rsi_14"          : "RSI-14 (Momentum Indicator)",
    "ma_ratio"        : "Price / 20-Day Moving Average",
    "sentiment_pos"   : "Positive Sentiment Score",
    "sentiment_neg"   : "Negative Sentiment Score",
}

FEATURE_THRESHOLDS = {
    "sentiment_score" : (0.1,  "Positive news sentiment → bullish signal",
                               "Negative news sentiment → bearish signal"),
    "daily_return"    : (0.0,  "Positive price momentum → Buy pressure",
                               "Negative price momentum → Sell pressure"),
    "rsi_14"          : (50,   "RSI > 50 → Overbought / Bullish",
                               "RSI < 50 → Oversold / Bearish"),
    "ma_ratio"        : (1.0,  "Price above SMA → Uptrend",
                               "Price below SMA → Downtrend"),
}


# ────────────────────────────────────────────────────────────
# PERMUTATION IMPORTANCE (model-agnostic)
# ────────────────────────────────────────────────────────────
def compute_feature_importance(model,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                n_repeats: int = 10) -> pd.DataFrame:
    """
    Compute permutation feature importance.

    For each feature: shuffle it and measure how much model
    accuracy drops. Larger drop → more important feature.

    Parameters
    ----------
    model     : fitted QSVC or SVM model
    X_test    : test features (PCA-reduced, 4D)
    y_test    : test labels
    n_repeats : number of shuffle repetitions

    Returns
    -------
    DataFrame sorted by importance (descending)
    """
    # Note: PCA collapses 8 original features into 4 components
    # We assign simplified names to PCA components
    pca_feature_names = [f"PC-{i+1}" for i in range(X_test.shape[1])]

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats     = n_repeats,
        random_state  = 42,
        scoring       = "accuracy",
        n_jobs        = -1,
    )

    importance_df = pd.DataFrame({
        "feature"   : pca_feature_names,
        "importance": result.importances_mean,
        "std"       : result.importances_std,
    }).sort_values("importance", ascending=False)

    return importance_df


# ────────────────────────────────────────────────────────────
# RAW FEATURE CONTRIBUTION SCORES
# ────────────────────────────────────────────────────────────
def compute_raw_contributions(raw_features: dict,
                               signal: int) -> pd.DataFrame:
    """
    Given the raw feature values for a single prediction,
    compute a normalised contribution score for each feature.

    This is a heuristic approach:
      positive contribution → supports the predicted signal
      negative contribution → contradicts the predicted signal

    Parameters
    ----------
    raw_features : dict {feature_name: float_value}
    signal       : 0=Sell, 1=Hold, 2=Buy

    Returns
    -------
    DataFrame with: feature, value, contribution, direction
    """
    rows = []
    for feat, val in raw_features.items():
        if feat not in FEATURE_COLS:
            continue

        val = float(val) if val is not None else 0.0

        # ── Heuristic contribution calculation ──────────────
        if feat == "sentiment_score":
            # Higher compound score → Buy signal
            contribution = val * (1 if signal == 2 else -0.5)
        elif feat == "daily_return":
            contribution = val * 10 * (1 if signal == 2 else -1)
        elif feat == "rsi_14":
            # RSI > 50 is bullish
            norm_rsi = (val - 50) / 50
            contribution = norm_rsi * (1 if signal == 2 else -1)
        elif feat == "ma_ratio":
            # Ratio > 1 means price is above trend
            contribution = (val - 1) * (1 if signal == 2 else -1)
        elif feat == "volatility_5d":
            # High volatility supports Hold signal
            contribution = val * (-1 if signal != 1 else 1)
        elif feat == "volume_norm":
            # High volume confirms the signal direction
            contribution = abs(val - 10) / 10
        elif feat == "sentiment_pos":
            contribution = val * (1 if signal == 2 else -0.3)
        elif feat == "sentiment_neg":
            contribution = val * (-1 if signal == 2 else 0.5)
        else:
            contribution = 0.0

        rows.append({
            "feature"       : feat,
            "description"   : FEATURE_DESCRIPTIONS.get(feat, feat),
            "value"         : round(val, 4),
            "contribution"  : round(contribution, 4),
            "direction"     : "🟢 Supports" if contribution > 0 else "🔴 Contradicts",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Normalise contributions to [-1, +1] range
        max_abs = df["contribution"].abs().max()
        if max_abs > 0:
            df["contribution_norm"] = df["contribution"] / max_abs
        else:
            df["contribution_norm"] = 0.0
        df = df.sort_values("contribution", ascending=False)

    return df


# ────────────────────────────────────────────────────────────
# PLAIN LANGUAGE EXPLANATION
# ────────────────────────────────────────────────────────────
def generate_explanation(signal_name: str,
                          confidence: float,
                          raw_features: dict) -> str:
    """
    Generate a 3-sentence human-readable explanation of the prediction.

    Parameters
    ----------
    signal_name  : "Buy" | "Hold" | "Sell"
    confidence   : 0.0–1.0
    raw_features : dict from predictor.predict()

    Returns
    -------
    str — plain language explanation
    """
    sentiment = raw_features.get("sentiment_score", 0.0)
    rsi       = raw_features.get("rsi_14", 50.0)
    ma_ratio  = raw_features.get("ma_ratio", 1.0)
    vol_5d    = raw_features.get("volatility_5d", 0.0)
    daily_ret = raw_features.get("daily_return", 0.0)

    # ── Sentiment line ─────────────────────────────────────
    if sentiment > 0.15:
        sent_line = "Market news sentiment is predominantly positive (bullish)."
    elif sentiment < -0.15:
        sent_line = "Market news sentiment is predominantly negative (bearish)."
    else:
        sent_line = "Market news sentiment is largely neutral."

    # ── Technical line ─────────────────────────────────────
    tech_parts = []
    if rsi > 60:
        tech_parts.append("RSI indicates overbought conditions")
    elif rsi < 40:
        tech_parts.append("RSI indicates oversold conditions")
    if ma_ratio > 1.02:
        tech_parts.append("price is trading above the 20-day moving average")
    elif ma_ratio < 0.98:
        tech_parts.append("price is trading below the 20-day moving average")
    if daily_ret > 0.01:
        tech_parts.append("recent price momentum is positive")
    elif daily_ret < -0.01:
        tech_parts.append("recent price momentum is negative")
    tech_line = (", ".join(tech_parts).capitalize() + ".") if tech_parts else \
                "Technical indicators are mixed."

    # ── Conclusion line ────────────────────────────────────
    conf_pct = round(confidence * 100, 1)
    conclusion = (f"The Quantum SVC model predicts a **{signal_name}** signal "
                  f"with {conf_pct}% confidence based on the combined "
                  f"quantum feature encoding of sentiment and price data.")

    return f"{sent_line} {tech_line} {conclusion}"


# ────────────────────────────────────────────────────────────
# QUICK TEST
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  QuantumSentinel — Explainability Test")
    print("=" * 55)

    sample_features = {
        "sentiment_score": 0.25,
        "daily_return"   : 0.012,
        "volume_norm"    : 15.2,
        "volatility_5d"  : 0.018,
        "rsi_14"         : 62.0,
        "ma_ratio"       : 1.03,
        "sentiment_pos"  : 0.72,
        "sentiment_neg"  : 0.08,
    }

    contribs = compute_raw_contributions(sample_features, signal=2)
    print("\nFeature Contributions:")
    print(contribs[["description", "value", "contribution", "direction"]]
          .to_string(index=False))

    explanation = generate_explanation("Buy", 0.78, sample_features)
    print(f"\nExplanation:\n{explanation}")
