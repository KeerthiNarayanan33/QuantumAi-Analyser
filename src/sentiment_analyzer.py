"""
============================================================
QuantumSentinel — Sentiment Analyzer
============================================================
Uses FinBERT (ProsusAI/finbert) — a BERT model fine-tuned
on financial text — to classify news headlines as:
  • Positive   (bullish sentiment)
  • Neutral    (no strong signal)
  • Negative   (bearish sentiment)

Also outputs a confidence score (0–1) for each prediction.

Model: https://huggingface.co/ProsusAI/finbert
============================================================
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

warnings.filterwarnings("ignore")

# ── Load centralized config ────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import cfg  # type: ignore[import]

# ── Model config (sourced from config) ──────────────────────
MODEL_NAME = cfg.sentiment.MODEL_NAME
LABELS     = cfg.sentiment.LABELS
CACHE_DIR  = str(cfg.sentiment.CACHE_DIR)

# Singleton holder — loaded once per session
_pipeline = None


def _get_pipeline():
    """
    Lazy-load the FinBERT pipeline (downloads ~440 MB on first run,
    then caches locally). Returns a HuggingFace text-classification pipeline.
    """
    global _pipeline
    if _pipeline is None:
        print("[INFO] Loading FinBERT model (first run may take ~30s)...")
        device = 0 if torch.cuda.is_available() else -1   # GPU if available
        _pipeline = pipeline(
            task          = "text-classification",
            model         = MODEL_NAME,
            tokenizer     = MODEL_NAME,
            top_k         = None,          # return ALL label scores
            device        = device,
            model_kwargs  = {"cache_dir": CACHE_DIR},
        )
        print("[OK]  FinBERT loaded.")
    return _pipeline


# ────────────────────────────────────────────────────────────
# CORE FUNCTION
# ────────────────────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """
    Run FinBERT on a single piece of financial text.

    Parameters
    ----------
    text : str — a news headline or short paragraph

    Returns
    -------
    dict with keys:
        label      : "positive" | "neutral" | "negative"
        positive   : float score 0–1
        neutral    : float score 0–1
        negative   : float score 0–1
        confidence : float — score of the winning label
        compound   : float in [-1, +1]  (positive – negative)
    """
    nlp = _get_pipeline()

    # FinBERT truncates at 512 tokens; truncate text if very long
    text_clean = str(text).strip()[:512]

    results = nlp(text_clean)[0]           # list of {label, score} dicts

    # Build lookup dict
    scores = {r["label"].lower(): r["score"] for r in results}

    pos = scores.get("positive", 0.0)
    neu = scores.get("neutral",  0.0)
    neg = scores.get("negative", 0.0)

    # Winning label
    label = max(scores, key=scores.get)

    return {
        "label"      : label,
        "positive"   : round(pos, 4),
        "neutral"    : round(neu, 4),
        "negative"   : round(neg, 4),
        "confidence" : round(scores[label], 4),
        "compound"   : round(pos - neg, 4),   # simple compound score
    }


def analyze_batch(headlines: list[str], batch_size: int = 16) -> list[dict]:
    """
    Run FinBERT on a list of headlines.

    Parameters
    ----------
    headlines  : list of strings
    batch_size : number of texts to process at once (tune for VRAM)

    Returns
    -------
    list of dicts (same schema as analyze_sentiment)
    """
    results = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        for text in batch:
            results.append(analyze_sentiment(text))
    return results


def analyze_news_df(news_df: pd.DataFrame,
                    text_col: str = "headline") -> pd.DataFrame:
    """
    Enrich a news DataFrame with FinBERT sentiment columns.

    Parameters
    ----------
    news_df  : DataFrame containing at least a text column
    text_col : name of the column holding the headline/text

    Returns
    -------
    Same DataFrame with added columns:
        label, positive, neutral, negative, confidence, compound
    """
    texts   = news_df[text_col].tolist()
    results = analyze_batch(texts)

    sentiment_df = pd.DataFrame(results)
    out = pd.concat([news_df.reset_index(drop=True), sentiment_df], axis=1)
    return out


# ────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK (for demo / offline mode)
# ────────────────────────────────────────────────────────────
_POSITIVE_WORDS = {
    "record", "beat", "surpass", "rally", "grow", "surge", "profit",
    "gain", "outperform", "strong", "upgrade", "rise", "boost", "high",
    "soar", "expand", "recovery", "resilience"
}
_NEGATIVE_WORDS = {
    "miss", "fall", "decline", "drop", "loss", "layoff", "cut", "weak",
    "disappoint", "crash", "risk", "pressure", "concern", "warn", "lower",
    "slump", "collapse", "delay"
}


def analyze_sentiment_fallback(text: str) -> dict:
    """
    Keyword-based sentiment when FinBERT cannot be loaded.
    Accuracy is lower but works fully offline.
    """
    words = set(str(text).lower().split())
    pos_hits = len(words & _POSITIVE_WORDS)
    neg_hits = len(words & _NEGATIVE_WORDS)

    total = pos_hits + neg_hits + 1e-9
    pos   = pos_hits / total
    neg   = neg_hits / total
    neu   = max(0.0, 1.0 - pos - neg)

    if pos > neg:
        label = "positive"
        conf  = pos
    elif neg > pos:
        label = "negative"
        conf  = neg
    else:
        label = "neutral"
        conf  = neu

    return {
        "label"      : label,
        "positive"   : round(pos, 4),
        "neutral"    : round(neu, 4),
        "negative"   : round(neg, 4),
        "confidence" : round(conf, 4),
        "compound"   : round(pos - neg, 4),
    }


def safe_analyze_news_df(news_df: pd.DataFrame,
                         text_col: str = "headline") -> pd.DataFrame:
    """
    Try FinBERT; silently fall back to keyword-based scoring on failure.
    This ensures the dashboard always has sentiment data.
    """
    try:
        return analyze_news_df(news_df, text_col)
    except Exception as e:
        print(f"[WARN] FinBERT failed ({e}). Using keyword fallback.")
        results = [analyze_sentiment_fallback(t) for t in news_df[text_col]]
        sentiment_df = pd.DataFrame(results)
        return pd.concat([news_df.reset_index(drop=True), sentiment_df], axis=1)


# ────────────────────────────────────────────────────────────
# QUICK TEST
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Apple reports record quarterly revenue, beating analyst expectations.",
        "Tesla deliveries miss Q4 targets, shares fall 12% in after-hours trading.",
        "Fed holds interest rates steady at current levels.",
    ]
    print("=" * 55)
    print("  QuantumSentinel — Sentiment Analyzer Test")
    print("=" * 55)
    for s in samples:
        r = safe_analyze_news_df(
            pd.DataFrame({"headline": [s]}), "headline"
        ).iloc[0]
        print(f"\n  Text      : {s[:60]}...")
        print(f"  Label     : {r['label']}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  Compound  : {r['compound']}")
