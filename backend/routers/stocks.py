"""
============================================================
QuantumSentinel — Stock Data Router
============================================================
Endpoints:
  GET /api/stocks/tickers        — list all available tickers
  GET /api/stocks/{ticker}       — OHLCV data for a ticker
  GET /api/stocks/{ticker}/summary — price summary statistics
============================================================
"""

import sys
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

# ── Add project root to path ─────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from config import cfg  # type: ignore[import]
from backend.schemas import StockResponse, TickerListResponse, OHLCVRecord, PeriodEnum  # type: ignore[import]

router = APIRouter(prefix="/stocks", tags=["Stock Data"])


# ── Lazy import to avoid startup delays ─────────────────────
def _get_stock_data(tickers: list[str], period: str):
    from data_collector import get_combined_stock_df
    return get_combined_stock_df(tickers, period)


# ────────────────────────────────────────────────────────────
# GET /api/stocks/tickers
# ────────────────────────────────────────────────────────────
@router.get(
    "/tickers",
    response_model=TickerListResponse,
    summary="List available tickers",
    description="Returns all tickers configured in the system.",
)
async def list_tickers() -> TickerListResponse:
    tickers = cfg.data.EXTENDED_TICKERS
    return TickerListResponse(tickers=tickers, count=len(tickers))


# ────────────────────────────────────────────────────────────
# GET /api/stocks/{ticker}
# ────────────────────────────────────────────────────────────
@router.get(
    "/{ticker}",
    response_model=StockResponse,
    summary="Get OHLCV stock data",
    description=(
        "Fetches Open, High, Low, Close, Volume data for the given ticker "
        "using Yahoo Finance (yfinance). Falls back to synthetic data if "
        "the ticker is unavailable."
    ),
    responses={404: {"description": "No data returned for ticker"}},
)
async def get_stock(
    ticker: str,
    period: PeriodEnum = Query(PeriodEnum.three_month, description="Look-back window"),
) -> StockResponse:
    ticker = ticker.upper()

    try:
        df = _get_stock_data([ticker], period.value)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data fetch error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data returned for ticker '{ticker}'.",
        )

    ticker_df = df[df["Ticker"] == ticker].copy()
    if ticker_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found in returned data.",
        )

    ticker_df = ticker_df.sort_values("Date")

    records = [
        OHLCVRecord(
            date   = str(row["Date"])[:10],
            open   = round(float(row["Open"]),   4),
            high   = round(float(row["High"]),   4),
            low    = round(float(row["Low"]),    4),
            close  = round(float(row["Close"]),  4),
            volume = int(row["Volume"]),
        )
        for _, row in ticker_df.iterrows()
    ]

    return StockResponse(
        ticker = ticker,
        period = period.value,
        rows   = len(records),
        data   = records,
    )


# ────────────────────────────────────────────────────────────
# GET /api/stocks/{ticker}/summary
# ────────────────────────────────────────────────────────────
@router.get(
    "/{ticker}/summary",
    summary="Price summary statistics",
    description="Returns key statistics (latest close, 52w high/low, avg volume, return %) for a ticker.",
    responses={404: {"description": "No data returned for ticker"}},
)
async def get_stock_summary(
    ticker: str,
    period: PeriodEnum = Query(PeriodEnum.three_month, description="Look-back window"),
) -> dict:
    ticker = ticker.upper()

    try:
        df = _get_stock_data([ticker], period.value)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data fetch error: {exc}")

    ticker_df = df[df["Ticker"] == ticker].copy()
    if ticker_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found in returned data.",
        )

    ticker_df = ticker_df.sort_values("Date")
    close      = ticker_df["Close"].dropna()

    if len(close) < 2:
        raise HTTPException(status_code=404, detail="Insufficient data for summary.")

    latest_close  = round(float(close.iloc[-1]),  2)
    prev_close    = round(float(close.iloc[-2]),  2)
    price_change  = round(latest_close - prev_close, 2)
    pct_change    = round((price_change / prev_close) * 100, 2)
    period_high   = round(float(close.max()),  2)
    period_low    = round(float(close.min()),  2)
    avg_volume    = int(ticker_df["Volume"].mean())
    total_return  = round(((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100, 2)

    return {
        "ticker"        : ticker,
        "period"        : period.value,
        "latest_close"  : latest_close,
        "prev_close"    : prev_close,
        "price_change"  : price_change,
        "pct_change"    : pct_change,
        "period_high"   : period_high,
        "period_low"    : period_low,
        "avg_volume"    : avg_volume,
        "total_return_pct" : total_return,
        "trading_days"  : len(ticker_df),
    }
