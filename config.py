"""
============================================================
QuantumSentinel — Centralized Configuration
============================================================
Single source of truth for ALL project settings.
Import this module anywhere instead of hardcoding values.

Usage:
    from config import cfg
    print(cfg.DEFAULT_TICKERS)
    print(cfg.MODEL_DIR)
============================================================
"""

import os
from pathlib import Path

# ── Project root (this file lives at the project root) ──────
ROOT_DIR = Path(__file__).resolve().parent


# ════════════════════════════════════════════════════════════
# APPLICATION
# ════════════════════════════════════════════════════════════
class AppConfig:
    TITLE       = "QuantumSentinel"
    SUBTITLE    = "Quantum Analytics for Investor Behaviour & Market Sentiment"
    VERSION     = "1.0.0"
    DESCRIPTION = (
        "An AI-powered investment signal platform combining FinBERT sentiment "
        "analysis with Quantum Support Vector Classifiers (QSVC) to generate "
        "Buy / Hold / Sell recommendations."
    )
    ICON        = "⚛️"
    GITHUB_URL  = "https://github.com/"
    CONTACT     = "support@quantumsentinel.ai"


# ════════════════════════════════════════════════════════════
# TICKERS & DATA
# ════════════════════════════════════════════════════════════
class DataConfig:
    # Default stock universe
    DEFAULT_TICKERS: list[str] = [
        "AAPL", "MSFT", "TSLA", "NVDA",
        "AMZN", "GOOGL", "META", "SPY",
    ]

    # Additional tickers available in the dashboard
    EXTENDED_TICKERS: list[str] = [
        "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "SPY",
        "JPM", "BAC", "GS", "XOM", "CVX", "PFE", "JNJ", "WMT", "NFLX",
        "AMD", "INTC", "QCOM",
    ]

    # yfinance look-back periods
    DEFAULT_PERIOD     = "3mo"    # used for real-time dashboard
    TRAINING_PERIOD    = "6mo"    # used when training models

    # Data paths
    DATA_DIR           = ROOT_DIR / "data"
    NEWS_CSV_PATH      = DATA_DIR / "sample_news.csv"
    PRODUCTS_CSV_PATH  = DATA_DIR / "products.csv"
    FEATURES_CSV_PATH  = DATA_DIR / "sample_features.csv"

    # Label thresholds for Buy / Hold / Sell
    BUY_THRESHOLD      = 0.005    # +0.5% next-day return → Buy
    SELL_THRESHOLD     = -0.005   # -0.5% next-day return → Sell


# ════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════
class ModelConfig:
    MODEL_DIR    = ROOT_DIR / "models"

    # Persisted artefact paths
    QSVC_PATH    = MODEL_DIR / "qsvc_model.pkl"
    SVM_PATH     = MODEL_DIR / "classical_svm.pkl"
    PCA_PATH     = MODEL_DIR / "pca_transform.pkl"
    SCALER_PATH  = MODEL_DIR / "feature_scaler.pkl"

    # Training hyper-parameters
    TEST_SIZE         = 0.25       # train / test split ratio
    RANDOM_STATE      = 42
    N_QUBITS          = 4          # number of qubits in the quantum kernel
    MAX_ITER          = 200        # SVM max iterations

    # Signal mappings
    SIGNAL_MAP  = {0: "Sell", 1: "Hold", 2: "Buy"}
    SIGNAL_ICON = {0: "🔴",   1: "🟡",   2: "🟢"}


# ════════════════════════════════════════════════════════════
# SENTIMENT / NLP
# ════════════════════════════════════════════════════════════
class SentimentConfig:
    # FinBERT — financial BERT model for sentiment classification
    MODEL_NAME   = "ProsusAI/finbert"
    CACHE_DIR    = ROOT_DIR / "models" / "finbert_cache"

    # Inference settings
    BATCH_SIZE   = 16              # headlines per batch
    MAX_LENGTH   = 512             # max tokens (FinBERT cap)

    # Label order as returned by FinBERT
    LABELS       = ["positive", "negative", "neutral"]


# ════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
class FeatureConfig:
    FEATURE_COLS: list[str] = [
        "sentiment_score",   # FinBERT compound score [-1, +1]
        "daily_return",      # (Close_t - Close_{t-1}) / Close_{t-1}
        "volume_norm",       # log-normalised trading volume
        "volatility_5d",     # rolling 5-day std of returns
        "rsi_14",            # 14-day Relative Strength Index
        "ma_ratio",          # Close / 20-day SMA
        "sentiment_pos",     # raw FinBERT positive score
        "sentiment_neg",     # raw FinBERT negative score
    ]

    RSI_PERIOD       = 14
    VOLATILITY_WINDOW = 5
    SMA_WINDOW       = 20


# ════════════════════════════════════════════════════════════
# BACKEND (FastAPI)
# ════════════════════════════════════════════════════════════
class BackendConfig:
    HOST         = "0.0.0.0"
    PORT         = 8000
    RELOAD       = True            # auto-reload on code changes (dev mode)
    LOG_LEVEL    = "info"
    API_PREFIX   = "/api"
    DOCS_URL     = "/docs"
    REDOC_URL    = "/redoc"

    # CORS — add your frontend origin if deploying separately
    CORS_ORIGINS: list[str] = [
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()
    ] or [
        "http://localhost:8501",   # Streamlit default
        "http://localhost:3000",   # React / Next.js dev
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8000",
    ]

    # Cache settings — keep loaded models in memory (seconds)
    MODEL_CACHE_TTL   = 3600       # 1 hour
    STOCK_CACHE_TTL   = 300        # 5 minutes
    SENTIMENT_CACHE_TTL = 600      # 10 minutes


# ════════════════════════════════════════════════════════════
# DASHBOARD (Streamlit)
# ════════════════════════════════════════════════════════════
class DashboardConfig:
    PAGE_TITLE      = "QuantumSentinel"
    PAGE_ICON       = "⚛️"
    LAYOUT          = "wide"
    SIDEBAR_STATE   = "expanded"

    AUTO_REFRESH_SECONDS  = 300    # 5-minute auto-refresh interval
    MAX_NEWS_DISPLAY      = 50     # max headlines in news table
    DEFAULT_CHART_HEIGHT  = 400    # pixels

    # Theme colours
    COLOR_BUY       = "#22c55e"    # green
    COLOR_HOLD      = "#eab308"    # yellow
    COLOR_SELL      = "#ef4444"    # red
    COLOR_PRIMARY   = "#6366f1"    # indigo
    COLOR_BG        = "#0f172a"    # dark navy


# ════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════
class LogConfig:
    LOG_DIR     = ROOT_DIR / "logs"
    LOG_FILE    = LOG_DIR / "quantumsentinel.log"
    LOG_LEVEL   = "INFO"
    LOG_FORMAT  = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    MAX_BYTES   = 10 * 1024 * 1024   # 10 MB per log file
    BACKUP_COUNT = 5


# ════════════════════════════════════════════════════════════
# UNIFIED CONFIG OBJECT
# ════════════════════════════════════════════════════════════
class Config:
    """Unified access point — use `from config import cfg`."""
    app       = AppConfig()
    data      = DataConfig()
    model     = ModelConfig()
    sentiment = SentimentConfig()
    features  = FeatureConfig()
    backend   = BackendConfig()
    dashboard = DashboardConfig()
    logging   = LogConfig()

    # ── Convenience shortcuts ────────────────────────────────
    @property
    def DEFAULT_TICKERS(self) -> list[str]:
        return self.data.DEFAULT_TICKERS

    @property
    def MODEL_DIR(self) -> Path:
        return self.model.MODEL_DIR

    @property
    def DATA_DIR(self) -> Path:
        return self.data.DATA_DIR

    @property
    def FEATURE_COLS(self) -> list[str]:
        return self.features.FEATURE_COLS

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [
            self.data.DATA_DIR,
            self.model.MODEL_DIR,
            self.sentiment.CACHE_DIR,
            self.logging.LOG_DIR,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


# ── Module-level singleton ───────────────────────────────────
cfg = Config()


# ── Quick self-test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  QuantumSentinel — Configuration")
    print("=" * 55)
    print(f"\n  App Title       : {cfg.app.TITLE} v{cfg.app.VERSION}")
    print(f"  Root Dir        : {ROOT_DIR}")
    print(f"  Default Tickers : {cfg.DEFAULT_TICKERS}")
    print(f"  Model Dir       : {cfg.MODEL_DIR}")
    print(f"  News CSV        : {cfg.data.NEWS_CSV_PATH}")
    print(f"  FinBERT Model   : {cfg.sentiment.MODEL_NAME}")
    print(f"  Feature Cols    : {cfg.FEATURE_COLS}")
    print(f"  API Port        : {cfg.backend.PORT}")
    print(f"  Buy Threshold   : {cfg.data.BUY_THRESHOLD}")
    print(f"  Sell Threshold  : {cfg.data.SELL_THRESHOLD}")
    print("\n  ✅  Config loaded successfully!")
