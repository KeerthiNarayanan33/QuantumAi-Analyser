"""
============================================================
QuantumSentinel — Data Collector
============================================================
Fetches:
  1. Stock price data from Yahoo Finance (yfinance)
  2. Financial news headlines (sample CSV + optional NewsAPI)

Output:
  - DataFrame of OHLCV stock data
  - DataFrame of news headlines with ticker & date
============================================================
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import yfinance as yf

# ── Load centralized config ──────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import cfg  # type: ignore[import]

# ── Default settings (sourced from config) ───────────────────
DEFAULT_TICKERS  = cfg.data.DEFAULT_TICKERS
DEFAULT_PERIOD   = cfg.data.DEFAULT_PERIOD
NEWS_CSV_PATH    = str(cfg.data.NEWS_CSV_PATH)


# ────────────────────────────────────────────────────────────
# 1. STOCK PRICE DATA
# ────────────────────────────────────────────────────────────
def fetch_stock_data(tickers: list[str] = DEFAULT_TICKERS,
                     period: str = DEFAULT_PERIOD) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV (Open, High, Low, Close, Volume) data for each ticker.

    Parameters
    ----------
    tickers : list of ticker symbols, e.g. ["AAPL", "TSLA"]
    period  : lookback string accepted by yfinance, e.g. "1mo", "3mo", "1y"

    Returns
    -------
    dict  {ticker: DataFrame with columns [Open, High, Low, Close, Volume]}
    """
    stock_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty:
                print(f"[WARN] No data for {ticker}, skipping.")
                continue
            # Flatten multi-level column index if present and ensure flat string column names
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
            else:
                df.columns = [str(c) for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df["Ticker"] = ticker
            stock_data[ticker] = df
            print(f"[OK]   Fetched {len(df)} rows for {ticker}")
        except Exception as e:
            print(f"[ERR]  {ticker}: {e}")
    return stock_data


def get_combined_stock_df(tickers: list[str] = DEFAULT_TICKERS,
                          period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """
    Fetch all tickers and stack them into a single long-format DataFrame.

    Returns
    -------
    DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume
    """
    raw = fetch_stock_data(tickers, period)
    frames = []
    for ticker, df in raw.items():
        df = df.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
        # Ensure all columns are flat strings (reset_index might introduce tuple names if columns had a name)
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        df["Ticker"] = ticker
        frames.append(df)

    if not frames:
        # Fallback: generate synthetic data so the dashboard never breaks
        print("[WARN] No live data fetched — generating synthetic fallback data.")
        return _generate_synthetic_stock_data(tickers)

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    return combined


# ────────────────────────────────────────────────────────────
# 2. NEWS DATA
# ────────────────────────────────────────────────────────────
def load_news(path: str = NEWS_CSV_PATH) -> pd.DataFrame:
    """
    Load financial news headlines from the bundled CSV sample file.

    Returns
    -------
    DataFrame with columns: headline, source, date, ticker, sector
    """
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        # Strip extra quotes that may appear in CSV
        df["headline"] = df["headline"].str.strip('"').str.strip()
        print(f"[OK]   Loaded {len(df)} news articles from {path}")
        return df
    except FileNotFoundError:
        print(f"[WARN] News CSV not found at {path} — generating synthetic headlines.")
        return _generate_synthetic_news()


def _generate_synthetic_news() -> pd.DataFrame:
    """Fallback: create synthetic news when CSV is unavailable."""
    headlines = [
        ("Apple reports record quarterly revenue", "Reuters", "AAPL", "Technology"),
        ("Fed signals rate cuts, markets rally", "Bloomberg", "SPY", "Finance"),
        ("Tesla deliveries miss targets, shares fall", "CNBC", "TSLA", "Automotive"),
        ("Microsoft Azure revenue grows 28%", "WSJ", "MSFT", "Technology"),
        ("Oil prices surge on Middle East tensions", "Reuters", "XOM", "Energy"),
        ("Nvidia GPU demand soars on AI boom", "CNBC", "NVDA", "Technology"),
        ("Banking sector faces rising credit risk", "FT", "JPM", "Finance"),
        ("Inflation shows cooling trend", "Bloomberg", "SPY", "Finance"),
        ("Meta reports strong user growth", "WSJ", "META", "Technology"),
        ("Pfizer cuts revenue forecast", "CNBC", "PFE", "Healthcare"),
    ]
    dates = pd.date_range(end=datetime.date.today(), periods=len(headlines), freq="D")
    rows = []
    for i, (h, src, tkr, sec) in enumerate(headlines):
        rows.append({"headline": h, "source": src, "date": dates[i],
                     "ticker": tkr, "sector": sec})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# 3. SYNTHETIC STOCK FALLBACK
# ────────────────────────────────────────────────────────────
def _generate_synthetic_stock_data(tickers: list[str]) -> pd.DataFrame:
    """Generate random-walk price data for demo purposes."""
    frames = []
    dates = pd.date_range(end=datetime.date.today(), periods=60, freq="B")
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**31)
        price = 100 + np.random.randn(len(dates)).cumsum()
        price = np.clip(price, 10, None)
        vol   = np.random.randint(1_000_000, 10_000_000, size=len(dates))
        df = pd.DataFrame({
            "Date":   dates,
            "Ticker": ticker,
            "Open":   price * np.random.uniform(0.99, 1.01, len(dates)),
            "High":   price * np.random.uniform(1.00, 1.03, len(dates)),
            "Low":    price * np.random.uniform(0.97, 1.00, len(dates)),
            "Close":  price,
            "Volume": vol,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ────────────────────────────────────────────────────────────
# QUICK TEST
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  QuantumSentinel — Data Collector Test")
    print("=" * 55)

    stocks = get_combined_stock_df(["AAPL", "TSLA"], period="1mo")
    print(f"\nStock data shape : {stocks.shape}")
    print(stocks.tail(3))

    news = load_news()
    print(f"\nNews data shape  : {news.shape}")
    print(news.head(3))
