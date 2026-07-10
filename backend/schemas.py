"""
============================================================
QuantumSentinel — Pydantic Schemas (Request / Response)
============================================================
All API data contracts are defined here.
FastAPI uses these for automatic validation + Swagger docs.
============================================================
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════
class SignalEnum(str, Enum):
    buy  = "Buy"
    hold = "Hold"
    sell = "Sell"


class PeriodEnum(str, Enum):
    one_month   = "1mo"
    three_month = "3mo"
    six_month   = "6mo"
    one_year    = "1y"
    two_year    = "2y"


class ModelEnum(str, Enum):
    qsvc    = "qsvc"
    svm     = "svm"
    auto    = "auto"


# ════════════════════════════════════════════════════════════
# COMMON
# ════════════════════════════════════════════════════════════
class HealthResponse(BaseModel):
    status    : str  = Field(..., example="ok")
    version   : str  = Field(..., example="1.0.0")
    service   : str  = Field(..., example="QuantumSentinel Backend")
    timestamp : str  = Field(..., example="2024-07-09T10:00:00Z")


class ErrorResponse(BaseModel):
    detail    : str  = Field(..., example="Ticker INVALID not found.")


# ════════════════════════════════════════════════════════════
# STOCK DATA
# ════════════════════════════════════════════════════════════
class OHLCVRecord(BaseModel):
    date   : str   = Field(..., example="2024-07-01")
    open   : float = Field(..., example=182.35)
    high   : float = Field(..., example=184.10)
    low    : float = Field(..., example=181.00)
    close  : float = Field(..., example=183.50)
    volume : int   = Field(..., example=52_300_000)

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-07-01",
                "open": 182.35, "high": 184.10,
                "low": 181.00, "close": 183.50,
                "volume": 52_300_000,
            }
        }


class StockResponse(BaseModel):
    ticker  : str             = Field(..., example="AAPL")
    period  : str             = Field(..., example="3mo")
    rows    : int             = Field(..., example=63)
    data    : List[OHLCVRecord]


class TickerListResponse(BaseModel):
    tickers : List[str]       = Field(..., example=["AAPL", "MSFT", "TSLA"])
    count   : int             = Field(..., example=20)


# ════════════════════════════════════════════════════════════
# SENTIMENT
# ════════════════════════════════════════════════════════════
class SentimentRecord(BaseModel):
    headline   : str   = Field(..., example="Apple beats earnings estimate")
    source     : str   = Field(..., example="Reuters")
    date       : str   = Field(..., example="2024-07-01")
    ticker     : str   = Field(..., example="AAPL")
    label      : str   = Field(..., example="positive")
    positive   : float = Field(..., example=0.87)
    neutral    : float = Field(..., example=0.10)
    negative   : float = Field(..., example=0.03)
    confidence : float = Field(..., example=0.87)
    compound   : float = Field(..., example=0.84)


class SentimentSummary(BaseModel):
    ticker            : str   = Field(..., example="AAPL")
    articles_analyzed : int   = Field(..., example=12)
    avg_compound      : float = Field(..., example=0.42)
    avg_positive      : float = Field(..., example=0.65)
    avg_neutral       : float = Field(..., example=0.25)
    avg_negative      : float = Field(..., example=0.10)
    dominant_label    : str   = Field(..., example="positive")
    headlines         : List[SentimentRecord]


class NewsResponse(BaseModel):
    total    : int              = Field(..., example=50)
    articles : List[SentimentRecord]


# ════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    ticker    : str        = Field(..., example="AAPL")
    model     : ModelEnum  = Field(ModelEnum.auto, example="auto")
    period    : PeriodEnum = Field(PeriodEnum.three_month, example="3mo")

    class Config:
        json_schema_extra = {
            "example": {"ticker": "AAPL", "model": "auto", "period": "3mo"}
        }


class BatchPredictRequest(BaseModel):
    tickers   : List[str]  = Field(..., min_length=1, example=["AAPL", "MSFT", "TSLA"])
    model     : ModelEnum  = Field(ModelEnum.auto, example="auto")
    period    : PeriodEnum = Field(PeriodEnum.three_month, example="3mo")

    class Config:
        json_schema_extra = {
            "example": {
                "tickers": ["AAPL", "MSFT", "TSLA"],
                "model"  : "auto",
                "period" : "3mo",
            }
        }


class ProbabilityBreakdown(BaseModel):
    Sell : float = Field(..., example=0.12)
    Hold : float = Field(..., example=0.25)
    Buy  : float = Field(..., example=0.63)


class PredictResponse(BaseModel):
    ticker         : str                 = Field(..., example="AAPL")
    signal         : int                 = Field(..., example=2, ge=0, le=2)
    signal_name    : str                 = Field(..., example="Buy")
    signal_icon    : str                 = Field(..., example="🟢")
    confidence     : float               = Field(..., example=0.63)
    probabilities  : ProbabilityBreakdown
    raw_features   : Dict[str, float]    = Field(..., example={"rsi_14": 62.3})
    model_used     : str                 = Field(..., example="QSVC (Quantum)")
    period         : str                 = Field(..., example="3mo")


class BatchPredictResponse(BaseModel):
    total    : int               = Field(..., example=3)
    results  : List[PredictResponse]


# ════════════════════════════════════════════════════════════
# MODEL STATUS & TRAINING
# ════════════════════════════════════════════════════════════
class ModelStatusResponse(BaseModel):
    qsvc_loaded   : bool = Field(..., example=True)
    svm_loaded    : bool = Field(..., example=True)
    pca_loaded    : bool = Field(..., example=True)
    scaler_loaded : bool = Field(..., example=True)
    all_ready     : bool = Field(..., example=True)
    message       : str  = Field(..., example="All models are loaded and ready.")


class TrainRequest(BaseModel):
    tickers : Optional[List[str]] = Field(
        None,
        example=["AAPL", "MSFT", "TSLA"],
        description="Tickers to train on. Defaults to config DEFAULT_TICKERS.",
    )
    period  : PeriodEnum = Field(
        PeriodEnum.six_month,
        example="6mo",
        description="Historical data window for training.",
    )


class TrainResponse(BaseModel):
    status       : str   = Field(..., example="started")
    message      : str   = Field(..., example="Training pipeline started in background.")
    tickers      : List[str]
    period       : str


class TrainResultResponse(BaseModel):
    status           : str   = Field(..., example="completed")
    qsvc_accuracy    : float = Field(..., example=0.73)
    svm_accuracy     : float = Field(..., example=0.71)
    training_seconds : float = Field(..., example=42.3)
    message          : str
