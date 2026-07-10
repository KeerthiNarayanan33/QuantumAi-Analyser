"""
============================================================
QuantumSentinel — Sentiment Router
============================================================
Endpoints:
  GET  /api/sentiment/news              — all news with sentiment
  GET  /api/sentiment/{ticker}          — sentiment summary for ticker
  POST /api/sentiment/analyze           — analyze custom headline
============================================================
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ── Add project root to path ─────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from config import cfg  # type: ignore[import]
from backend.schemas import (  # type: ignore[import]
    SentimentSummary, SentimentRecord, NewsResponse,
)

router = APIRouter(prefix="/sentiment", tags=["Sentiment Analysis"])


# ── Helpers ──────────────────────────────────────────────────
def _load_and_analyze_news():
    from data_collector     import load_news
    from sentiment_analyzer import safe_analyze_news_df
    raw  = load_news(str(cfg.data.NEWS_CSV_PATH))
    return safe_analyze_news_df(raw)


def _row_to_record(row) -> SentimentRecord:
    return SentimentRecord(
        headline   = str(row.get("headline", "")),
        source     = str(row.get("source", "Unknown")),
        date       = str(row.get("date", ""))[:10],
        ticker     = str(row.get("ticker", "")),
        label      = str(row.get("label", "neutral")),
        positive   = round(float(row.get("positive",   0.0)), 4),
        neutral    = round(float(row.get("neutral",    0.0)), 4),
        negative   = round(float(row.get("negative",   0.0)), 4),
        confidence = round(float(row.get("confidence", 0.0)), 4),
        compound   = round(float(row.get("compound",   0.0)), 4),
    )


# ────────────────────────────────────────────────────────────
# GET /api/sentiment/news
# ────────────────────────────────────────────────────────────
@router.get(
    "/news",
    response_model=NewsResponse,
    summary="All news with sentiment scores",
    description=(
        "Loads all available news headlines, runs FinBERT sentiment analysis, "
        "and returns the enriched dataset. Falls back to keyword-based scoring "
        "if FinBERT is unavailable."
    ),
)
async def get_all_news(
    limit: int = Query(50, ge=1, le=500, description="Max articles to return"),
) -> NewsResponse:
    try:
        df = _load_and_analyze_news()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"News load error: {exc}")

    df = df.head(limit)
    articles = [_row_to_record(row) for _, row in df.iterrows()]
    return NewsResponse(total=len(articles), articles=articles)


# ────────────────────────────────────────────────────────────
# GET /api/sentiment/{ticker}
# ────────────────────────────────────────────────────────────
@router.get(
    "/{ticker}",
    response_model=SentimentSummary,
    summary="Sentiment summary for a ticker",
    description=(
        "Returns aggregated FinBERT sentiment statistics for all news "
        "articles matching the given ticker symbol."
    ),
    responses={404: {"description": "No news found for ticker"}},
)
async def get_ticker_sentiment(
    ticker: str,
    include_market: bool = Query(
        True,
        description="Also include market-wide news (SPY) in aggregation",
    ),
) -> SentimentSummary:
    ticker = ticker.upper()

    try:
        df = _load_and_analyze_news()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

    # Filter to ticker (+ optionally SPY market news)
    if include_market:
        mask = df["ticker"].isin([ticker, "SPY"])
    else:
        mask = df["ticker"] == ticker

    subset = df[mask]
    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No news articles found for ticker '{ticker}'.",
        )

    # Aggregate stats
    avg_compound = round(float(subset["compound"].mean()),  4)
    avg_positive = round(float(subset["positive"].mean()), 4)
    avg_neutral  = round(float(subset["neutral"].mean()),  4)
    avg_negative = round(float(subset["negative"].mean()), 4)

    label_counts  = subset["label"].value_counts()
    dominant      = label_counts.idxmax() if not label_counts.empty else "neutral"

    headlines = [_row_to_record(row) for _, row in subset.iterrows()]

    return SentimentSummary(
        ticker            = ticker,
        articles_analyzed = len(subset),
        avg_compound      = avg_compound,
        avg_positive      = avg_positive,
        avg_neutral       = avg_neutral,
        avg_negative      = avg_negative,
        dominant_label    = dominant,
        headlines         = headlines,
    )


# ────────────────────────────────────────────────────────────
# POST /api/sentiment/analyze
# ────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5, example="Apple reports record quarterly earnings")

    class Config:
        json_schema_extra = {
            "example": {"text": "Apple reports record quarterly earnings, beating analyst expectations."}
        }


class AnalyzeResponse(BaseModel):
    text       : str
    label      : str
    positive   : float
    neutral    : float
    negative   : float
    confidence : float
    compound   : float
    model_used : str


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a custom headline",
    description=(
        "Run FinBERT (or keyword fallback) sentiment analysis on any "
        "financial text you provide. Useful for quick ad-hoc analysis."
    ),
)
async def analyze_custom_text(body: AnalyzeRequest) -> AnalyzeResponse:
    try:
        from sentiment_analyzer import safe_analyze_news_df
        import pandas as pd
        df_in  = pd.DataFrame({"headline": [body.text]})
        df_out = safe_analyze_news_df(df_in, "headline")
        row    = df_out.iloc[0]
        model_used = "FinBERT (ProsusAI/finbert)"
    except Exception:
        from sentiment_analyzer import analyze_sentiment_fallback
        row        = type("Row", (), analyze_sentiment_fallback(body.text))()
        model_used = "Keyword fallback"

    return AnalyzeResponse(
        text       = body.text,
        label      = str(row["label"]),
        positive   = round(float(row["positive"]),   4),
        neutral    = round(float(row["neutral"]),    4),
        negative   = round(float(row["negative"]),   4),
        confidence = round(float(row["confidence"]), 4),
        compound   = round(float(row["compound"]),   4),
        model_used = model_used,
    )
