"""
============================================================
QuantumSentinel — Predictions Router
============================================================
Endpoints:
  POST /api/predict            — single ticker prediction
  POST /api/predict/batch      — batch predictions
  GET  /api/predict/models     — model status check
  POST /api/predict/train      — trigger training pipeline
============================================================
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

# ── Add project root to path ─────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from config import cfg  # type: ignore[import]
from backend.schemas import (  # type: ignore[import]
    PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse,
    ProbabilityBreakdown, ModelStatusResponse,
    TrainRequest, TrainResponse, TrainResultResponse,
    ModelEnum, PeriodEnum,
)

router = APIRouter(prefix="/predict", tags=["Predictions"])

# ── In-memory training state tracker ─────────────────────────
_train_state: dict = {"status": "idle", "result": None}


# ── Helpers ──────────────────────────────────────────────────
def _load_models():
    from predictor import load_models
    return load_models()


def _run_prediction(ticker: str, period: str, model_pref: ModelEnum) -> PredictResponse:
    from data_collector     import get_combined_stock_df, load_news
    from sentiment_analyzer import safe_analyze_news_df
    from predictor          import load_models, predict

    models   = _load_models()
    stock_df = get_combined_stock_df([ticker], period)
    news_raw = load_news(str(cfg.data.NEWS_CSV_PATH))
    news_df  = safe_analyze_news_df(news_raw)

    use_qsvc = (model_pref == ModelEnum.qsvc) or (model_pref == ModelEnum.auto)
    result   = predict(ticker, models, stock_df, news_df, use_qsvc=use_qsvc)

    return PredictResponse(
        ticker        = result["ticker"],
        signal        = result["signal"],
        signal_name   = result["signal_name"],
        signal_icon   = result["signal_icon"],
        confidence    = result["confidence"],
        probabilities = ProbabilityBreakdown(**result["probabilities"]),
        raw_features  = result["raw_features"],
        model_used    = result["model_used"],
        period        = period,
    )


def _background_train(tickers: list[str], period: str):
    """Run the full training pipeline in a background thread."""
    global _train_state
    _train_state = {"status": "running", "result": None}
    t0 = time.time()

    try:
        from data_collector     import get_combined_stock_df, load_news
        from sentiment_analyzer import safe_analyze_news_df
        from feature_engineer   import build_multi_ticker_dataset
        from quantum_model      import train_all
        import joblib

        stock_df = get_combined_stock_df(tickers, period)
        news_raw = load_news(str(cfg.data.NEWS_CSV_PATH))
        news_df  = safe_analyze_news_df(news_raw)
        X, y, scaler = build_multi_ticker_dataset(stock_df, news_df, tickers)

        cfg.model.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, str(cfg.model.SCALER_PATH))

        results  = train_all(X, y, test_size=cfg.model.TEST_SIZE)
        elapsed  = round(time.time() - t0, 1)

        _train_state = {
            "status": "completed",
            "result": {
                "qsvc_accuracy"    : results["qsvc_accuracy"],
                "svm_accuracy"     : results["svm_accuracy"],
                "training_seconds" : elapsed,
            },
        }
    except Exception as exc:
        _train_state = {"status": "error", "result": {"error": str(exc)}}


# ────────────────────────────────────────────────────────────
# POST /api/predict
# ────────────────────────────────────────────────────────────
@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict Buy / Hold / Sell for a ticker",
    description=(
        "Runs the full QuantumSentinel inference pipeline:\n"
        "1. Fetches live stock data (yfinance)\n"
        "2. Loads & analyses news sentiment (FinBERT)\n"
        "3. Engineers features\n"
        "4. Runs QSVC or classical SVM prediction\n\n"
        "Falls back to rule-based signal if no trained model is available."
    ),
)
async def predict_ticker(body: PredictRequest) -> PredictResponse:
    ticker = body.ticker.upper()
    try:
        return _run_prediction(ticker, body.period.value, body.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


# ────────────────────────────────────────────────────────────
# POST /api/predict/batch
# ────────────────────────────────────────────────────────────
@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch predictions for multiple tickers",
    description="Runs the prediction pipeline for a list of tickers in sequence.",
)
async def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    tickers = [t.upper() for t in body.tickers]
    results = []

    for ticker in tickers:
        try:
            res = _run_prediction(ticker, body.period.value, body.model)
            results.append(res)
        except Exception as exc:
            # Don't fail the entire batch — skip failed tickers
            print(f"[WARN] Prediction failed for {ticker}: {exc}")

    if not results:
        raise HTTPException(
            status_code=500,
            detail="All predictions failed. Check logs for details.",
        )

    return BatchPredictResponse(total=len(results), results=results)


# ────────────────────────────────────────────────────────────
# GET /api/predict/models
# ────────────────────────────────────────────────────────────
@router.get(
    "/models",
    response_model=ModelStatusResponse,
    summary="Check model status",
    description="Checks whether all required model artefacts (QSVC, SVM, PCA, scaler) are present on disk.",
)
async def get_model_status() -> ModelStatusResponse:
    qsvc_ok   = cfg.model.QSVC_PATH.exists()
    svm_ok    = cfg.model.SVM_PATH.exists()
    pca_ok    = cfg.model.PCA_PATH.exists()
    scaler_ok = cfg.model.SCALER_PATH.exists()
    all_ok    = all([qsvc_ok, svm_ok, pca_ok, scaler_ok])

    if all_ok:
        msg = "All models are loaded and ready."
    else:
        missing = [
            name for name, ok in [
                ("QSVC",   qsvc_ok),
                ("SVM",    svm_ok),
                ("PCA",    pca_ok),
                ("scaler", scaler_ok),
            ]
            if not ok
        ]
        msg = f"Missing model artefacts: {', '.join(missing)}. Run /api/predict/train first."

    return ModelStatusResponse(
        qsvc_loaded   = qsvc_ok,
        svm_loaded    = svm_ok,
        pca_loaded    = pca_ok,
        scaler_loaded = scaler_ok,
        all_ready     = all_ok,
        message       = msg,
    )


# ────────────────────────────────────────────────────────────
# POST /api/predict/train
# ────────────────────────────────────────────────────────────
@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Trigger model training",
    description=(
        "Launches the full QuantumSentinel training pipeline in the background:\n"
        "1. Fetches stock data\n"
        "2. Runs FinBERT sentiment analysis\n"
        "3. Engineers features\n"
        "4. Trains QSVC + classical SVM\n"
        "5. Saves all artefacts to /models/\n\n"
        "Returns immediately. Poll `GET /api/predict/train/status` for progress."
    ),
)
async def trigger_training(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TrainResponse:
    global _train_state

    if _train_state["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Training is already running. Check /api/predict/train/status for progress.",
        )

    tickers = body.tickers or cfg.data.DEFAULT_TICKERS
    period  = body.period.value

    background_tasks.add_task(_background_train, tickers, period)

    return TrainResponse(
        status  = "started",
        message = "Training pipeline started in the background. Poll /api/predict/train/status.",
        tickers = tickers,
        period  = period,
    )


# ────────────────────────────────────────────────────────────
# GET /api/predict/train/status
# ────────────────────────────────────────────────────────────
@router.get(
    "/train/status",
    summary="Check training status",
    description="Returns the current state of the background training job.",
)
async def get_train_status() -> dict:
    state = _train_state.copy()

    if state["status"] == "completed" and state["result"]:
        r = state["result"]
        return {
            "status"           : "completed",
            "qsvc_accuracy"    : round(r.get("qsvc_accuracy",    0.0) * 100, 1),
            "svm_accuracy"     : round(r.get("svm_accuracy",     0.0) * 100, 1),
            "training_seconds" : r.get("training_seconds", 0.0),
            "message"          : "Training complete. Models saved to /models/.",
        }

    if state["status"] == "error":
        return {
            "status" : "error",
            "message": state["result"].get("error", "Unknown error."),
        }

    return {
        "status" : state["status"],
        "message": "Training is running..." if state["status"] == "running" else "No training job started.",
    }
