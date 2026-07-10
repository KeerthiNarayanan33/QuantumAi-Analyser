"""
============================================================
QuantumSentinel — Feature Engineer
============================================================
Builds the feature matrix used by both classical SVM and
the Quantum Support Vector Classifier (QSVC).

Features generated
──────────────────
  1. sentiment_score   : FinBERT compound score [-1, +1]
  2. daily_return      : (Close_t - Close_{t-1}) / Close_{t-1}
  3. volume_norm       : log-normalised trading volume
  4. volatility_5d     : rolling 5-day std of returns
  5. rsi_14            : 14-day Relative Strength Index
  6. ma_ratio          : Close / 20-day SMA  (trend proxy)
  7. sentiment_pos     : raw FinBERT positive score
  8. sentiment_neg     : raw FinBERT negative score

Target (label)
──────────────
  Based on next-day return:
    > +0.5%  → 2  (Buy)
    < -0.5%  → 0  (Sell)
    else     → 1  (Hold)
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ────────────────────────────────────────────────────────────
# PRICE FEATURES
# ────────────────────────────────────────────────────────────
def compute_price_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator columns to a single-ticker OHLCV DataFrame.

    Parameters
    ----------
    stock_df : DataFrame with columns [Date, Open, High, Low, Close, Volume]

    Returns
    -------
    DataFrame with new feature columns appended
    """
    df = stock_df.copy().sort_values("Date").reset_index(drop=True)

    # 1. Daily return
    df["daily_return"] = df["Close"].pct_change()

    # 2. Log-normalised volume  (log avoids scale issues with large integers)
    df["volume_norm"] = np.log1p(df["Volume"])

    # 3. Rolling 5-day volatility  (std of daily returns)
    df["volatility_5d"] = df["daily_return"].rolling(window=5).std()

    # 4. RSI-14  ─────────────────────────────────────────────
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=13, adjust=False).mean()   # Wilder smoothing
    avg_l  = loss.ewm(com=13, adjust=False).mean()
    rs     = avg_g / (avg_l + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 5. Price / 20-day SMA  (>1 = price above trend = bullish)
    sma_20         = df["Close"].rolling(window=20).mean()
    df["ma_ratio"] = df["Close"] / (sma_20 + 1e-9)

    return df


# ────────────────────────────────────────────────────────────
# MERGE STOCK + SENTIMENT
# ────────────────────────────────────────────────────────────
def merge_stock_sentiment(stock_df: pd.DataFrame,
                          sentiment_df: pd.DataFrame,
                          ticker: str) -> pd.DataFrame:
    """
    Join daily stock features with the aggregated daily sentiment
    for that ticker.

    Parameters
    ----------
    stock_df     : long-format stock DataFrame (all tickers)
    sentiment_df : news DataFrame with sentiment columns already added
    ticker       : ticker symbol to filter on

    Returns
    -------
    Merged DataFrame indexed by Date, ready for feature extraction
    """
    # ── Filter to the requested ticker ──────────────────────
    stocks = stock_df[stock_df["Ticker"] == ticker].copy()
    stocks = compute_price_features(stocks)
    stocks["Date"] = pd.to_datetime(stocks["Date"]).dt.normalize()

    # ── Aggregate daily sentiment for this ticker ────────────
    news = sentiment_df.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()

    # Include both ticker-specific news AND market-wide news (SPY)
    relevant = news[news["ticker"].isin([ticker, "SPY"])].copy()

    daily_sent = (
        relevant
        .groupby("date")
        .agg(
            sentiment_score=("compound", "mean"),
            sentiment_pos  =("positive", "mean"),
            sentiment_neg  =("negative", "mean"),
            news_count     =("headline", "count"),
        )
        .reset_index()
        .rename(columns={"date": "Date"})
    )

    # ── Merge on Date (left join keeps all trading days) ─────
    merged = stocks.merge(daily_sent, on="Date", how="left")

    # Forward-fill sentiment on days with no news
    for col in ["sentiment_score", "sentiment_pos", "sentiment_neg", "news_count"]:
        merged[col] = merged[col].ffill().fillna(0.0)

    return merged


# ────────────────────────────────────────────────────────────
# LABEL GENERATION
# ────────────────────────────────────────────────────────────
def generate_labels(df: pd.DataFrame,
                    buy_threshold: float  =  0.005,
                    sell_threshold: float = -0.005) -> pd.DataFrame:
    """
    Create the target column `signal` based on next-day return.

      next_return > +0.5%  → 2  (Buy)
      next_return < -0.5%  → 0  (Sell)
      otherwise            → 1  (Hold)

    Parameters
    ----------
    df             : DataFrame with `daily_return` column
    buy_threshold  : minimum next-day return to label as Buy
    sell_threshold : maximum next-day return to label as Sell

    Returns
    -------
    DataFrame with added `next_return` and `signal` columns
    """
    df = df.copy()
    df["next_return"] = df["daily_return"].shift(-1)   # look ahead 1 day

    conditions = [
        df["next_return"] >  buy_threshold,
        df["next_return"] <  sell_threshold,
    ]
    choices = [2, 0]   # Buy=2, Sell=0
    df["signal"] = np.select(conditions, choices, default=1)   # Hold=1

    return df


# ────────────────────────────────────────────────────────────
# FEATURE MATRIX BUILDER
# ────────────────────────────────────────────────────────────
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

def build_feature_matrix(merged_df: pd.DataFrame,
                         scale: bool = True) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Extract and optionally scale the feature matrix X and label vector y.

    Parameters
    ----------
    merged_df : output of merge_stock_sentiment() + generate_labels()
    scale     : whether to MinMax-scale features to [0, 1]
                (required for QSVC quantum kernel)

    Returns
    -------
    X      : np.ndarray  shape (n_samples, n_features)
    y      : np.ndarray  shape (n_samples,)  — signal labels
    scaler : fitted MinMaxScaler (or None)
    """
    df = generate_labels(merged_df).dropna(subset=FEATURE_COLS + ["signal"])

    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df["signal"].values.astype(int)

    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)

    return X, y, scaler


def build_multi_ticker_dataset(stock_df: pd.DataFrame,
                                sentiment_df: pd.DataFrame,
                                tickers: list[str],
                                scale: bool = True):
    """
    Build a combined dataset from multiple tickers for richer training.

    Returns
    -------
    X, y, scaler
    """
    all_X, all_y = [], []

    for ticker in tickers:
        try:
            merged = merge_stock_sentiment(stock_df, sentiment_df, ticker)
            merged = generate_labels(merged)
            df     = merged.dropna(subset=FEATURE_COLS + ["signal"])
            if len(df) < 10:
                continue
            X_t = df[FEATURE_COLS].values.astype(np.float64)
            y_t = df["signal"].values.astype(int)
            all_X.append(X_t)
            all_y.append(y_t)
        except Exception as e:
            print(f"[WARN] Skipping {ticker}: {e}")

    if not all_X:
        raise ValueError("No valid data produced for any ticker.")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)

    return X, y, scaler


# ────────────────────────────────────────────────────────────
# QUICK TEST
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_collector import get_combined_stock_df, load_news
    from sentiment_analyzer import safe_analyze_news_df

    print("=" * 55)
    print("  QuantumSentinel — Feature Engineering Test")
    print("=" * 55)

    stocks  = get_combined_stock_df(["AAPL", "MSFT"], "2mo")
    news    = load_news()
    news_s  = safe_analyze_news_df(news)

    merged  = merge_stock_sentiment(stocks, news_s, "AAPL")
    X, y, _ = build_feature_matrix(merged)

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Label distribution   : Buy={sum(y==2)}, Hold={sum(y==1)}, Sell={sum(y==0)}")
    print(f"Feature names        : {FEATURE_COLS}")
