"""
============================================================
QuantumSentinel — FastAPI Backend (main.py)
============================================================
Entry point for the REST API server.

Start the server:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Or use the convenience script:
    run_backend.bat

Swagger UI  → http://localhost:8000/docs
ReDoc       → http://localhost:8000/redoc
OpenAPI JSON→ http://localhost:8000/openapi.json
============================================================
"""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Add project root to path ─────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from config import cfg  # type: ignore[import]

# ── Routers ──────────────────────────────────────────────────
from backend.routers.stocks      import router as stocks_router  # type: ignore[import]
from backend.routers.sentiment   import router as sentiment_router  # type: ignore[import]
from backend.routers.predictions import router as predictions_router  # type: ignore[import]

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level   = cfg.logging.LOG_LEVEL,
    format  = cfg.logging.LOG_FORMAT,
    datefmt = cfg.logging.DATE_FORMAT,
)
logger = logging.getLogger("quantumsentinel.backend")


# ════════════════════════════════════════════════════════════
# APPLICATION
# ════════════════════════════════════════════════════════════
app = FastAPI(
    title       = f"{cfg.app.TITLE} API",
    description = (
        f"## {cfg.app.SUBTITLE}\n\n"
        f"{cfg.app.DESCRIPTION}\n\n"
        "### Key Features\n"
        "- ⚛️ **Quantum SVC** predictions (Buy / Hold / Sell)\n"
        "- 🧠 **FinBERT** financial sentiment analysis\n"
        "- 📈 **Live stock data** via Yahoo Finance\n"
        "- 🔄 **Background model training** with status polling\n\n"
        "### Authentication\n"
        "No authentication is required for local use. "
        "Add an API key middleware for production deployments.\n\n"
        f"GitHub: [{cfg.app.GITHUB_URL}]({cfg.app.GITHUB_URL})"
    ),
    version     = cfg.app.VERSION,
    docs_url    = cfg.backend.DOCS_URL,
    redoc_url   = cfg.backend.REDOC_URL,
    contact     = {"name": "QuantumSentinel", "email": cfg.app.CONTACT},
    license_info= {"name": "MIT"},
    openapi_tags= [
        {
            "name"       : "Health",
            "description": "Server health checks and metadata.",
        },
        {
            "name"       : "Stock Data",
            "description": "Fetch OHLCV price data from Yahoo Finance.",
        },
        {
            "name"       : "Sentiment Analysis",
            "description": "FinBERT financial sentiment for news headlines.",
        },
        {
            "name"       : "Predictions",
            "description": "Buy / Hold / Sell signal generation using QSVC or SVM.",
        },
    ],
)


# ════════════════════════════════════════════════════════════
# MIDDLEWARE
# ════════════════════════════════════════════════════════════
# CORS — allow Streamlit and local development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins     = cfg.backend.CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request logging middleware ───────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.now(timezone.utc)
    response = await call_next(request)
    elapsed  = (datetime.now(timezone.utc) - start).total_seconds() * 1000
    logger.info(
        f"{request.method:6s} {request.url.path:<45s} "
        f"→ {response.status_code}  [{elapsed:.0f}ms]"
    )
    return response


# ════════════════════════════════════════════════════════════
# ROUTERS
# ════════════════════════════════════════════════════════════
API = cfg.backend.API_PREFIX   # "/api"

app.include_router(stocks_router,      prefix=API)
app.include_router(sentiment_router,   prefix=API)
app.include_router(predictions_router, prefix=API)


# ════════════════════════════════════════════════════════════
# HEALTH & ROOT ENDPOINTS
# ════════════════════════════════════════════════════════════
@app.get(
    "/",
    tags=["Health"],
    summary="Root — server info",
    description="Returns basic server metadata and links to API documentation.",
)
async def root() -> dict:
    return {
        "service"    : f"{cfg.app.TITLE} Backend API",
        "version"    : cfg.app.VERSION,
        "status"     : "online",
        "timestamp"  : datetime.now(timezone.utc).isoformat(),
        "docs"       : "/docs",
        "redoc"      : "/redoc",
        "api_prefix" : API,
        "endpoints"  : {
            "tickers"       : f"{API}/stocks/tickers",
            "stock_data"    : f"{API}/stocks/{{ticker}}",
            "stock_summary" : f"{API}/stocks/{{ticker}}/summary",
            "news"          : f"{API}/sentiment/news",
            "sentiment"     : f"{API}/sentiment/{{ticker}}",
            "analyze"       : f"{API}/sentiment/analyze",
            "predict"       : f"{API}/predict",
            "predict_batch" : f"{API}/predict/batch",
            "model_status"  : f"{API}/predict/models",
            "train"         : f"{API}/predict/train",
            "train_status"  : f"{API}/predict/train/status",
        },
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Lightweight health probe for monitoring and load balancers.",
)
async def health() -> dict:
    return {
        "status"    : "ok",
        "version"   : cfg.app.VERSION,
        "service"   : f"{cfg.app.TITLE} Backend",
        "timestamp" : datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/info",
    tags=["Health"],
    summary="System info",
    description="Returns current configuration summary (safe fields only).",
)
async def info() -> dict:
    return {
        "app": {
            "title"   : cfg.app.TITLE,
            "version" : cfg.app.VERSION,
        },
        "data": {
            "default_tickers" : cfg.data.DEFAULT_TICKERS,
            "default_period"  : cfg.data.DEFAULT_PERIOD,
            "buy_threshold"   : cfg.data.BUY_THRESHOLD,
            "sell_threshold"  : cfg.data.SELL_THRESHOLD,
        },
        "model": {
            "n_qubits"      : cfg.model.N_QUBITS,
            "test_size"     : cfg.model.TEST_SIZE,
            "qsvc_exists"   : cfg.model.QSVC_PATH.exists(),
            "svm_exists"    : cfg.model.SVM_PATH.exists(),
        },
        "sentiment": {
            "model_name" : cfg.sentiment.MODEL_NAME,
            "batch_size" : cfg.sentiment.BATCH_SIZE,
        },
        "features": {
            "columns" : cfg.features.FEATURE_COLS,
        },
        "backend": {
            "host"       : cfg.backend.HOST,
            "port"       : cfg.backend.PORT,
            "api_prefix" : cfg.backend.API_PREFIX,
        },
    }


# ════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HANDLER
# ════════════════════════════════════════════════════════════
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail"  : "An internal server error occurred.",
            "type"    : type(exc).__name__,
            "path"    : str(request.url.path),
            "hint"    : "Check server logs for details.",
        },
    )


# ════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN EVENTS
# ════════════════════════════════════════════════════════════
@app.on_event("startup")
async def on_startup():
    cfg.ensure_dirs()
    logger.info("=" * 55)
    logger.info(f"  {cfg.app.TITLE} Backend API v{cfg.app.VERSION}")
    logger.info("=" * 55)
    logger.info(f"  Docs     → http://localhost:{cfg.backend.PORT}/docs")
    logger.info(f"  Health   → http://localhost:{cfg.backend.PORT}/health")
    logger.info(f"  API Base → http://localhost:{cfg.backend.PORT}{API}")
    logger.info("  Status   : Ready ✅")
    logger.info("=" * 55)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info(f"{cfg.app.TITLE} Backend shutting down. Goodbye! 👋")


# ════════════════════════════════════════════════════════════
# DIRECT RUN  (python backend/main.py)
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host      = cfg.backend.HOST,
        port      = cfg.backend.PORT,
        reload    = cfg.backend.RELOAD,
        log_level = cfg.backend.LOG_LEVEL,
    )
