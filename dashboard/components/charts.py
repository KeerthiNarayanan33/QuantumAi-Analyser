"""
============================================================
QuantumSentinel — Chart Components  [Pro-Max Redesign]
============================================================
21st-dev Plotly theme:
  • Transparent backgrounds — blends with glassmorphism cards
  • Neon grid lines + glowing traces
  • Gradient fills on area/bar charts
  • Custom hover templates with emoji labels
  • Premium gauge with neon arc
============================================================
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Pro-Max colour palette ─────────────────────────────────
COLOURS = {
    "positive" : "#4ade80",   # neon green
    "neutral"  : "#fbbf24",   # gold
    "negative" : "#f87171",   # hot red
    "buy"      : "#4ade80",
    "hold"     : "#fbbf24",
    "sell"     : "#f87171",
    "accent"   : "#38bdf8",   # electric cyan
    "purple"   : "#a78bfa",   # quantum violet
    "bg"       : "#020817",   # ultra-dark
    "surface"  : "rgba(13,27,42,0.0)",   # transparent for glassmorphism
    "grid"     : "rgba(56,189,248,0.06)",
    "text"     : "#f1f5f9",
    "muted"    : "#475569",
}

# ── Shared layout base ─────────────────────────────────────
def _base_layout(**kwargs) -> dict:
    layout = dict(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(
            family = "Space Grotesk, sans-serif",
            color  = COLOURS["text"],
            size   = 12,
        ),
        margin        = dict(t=50, b=40, l=50, r=20),
        legend        = dict(
            bgcolor     = "rgba(13,27,42,0.8)",
            bordercolor = "rgba(56,189,248,0.2)",
            borderwidth = 1,
            font        = dict(color=COLOURS["text"]),
        ),
        hoverlabel = dict(
            bgcolor     = "rgba(13,27,42,0.95)",
            bordercolor = "rgba(56,189,248,0.3)",
            font        = dict(family="Space Grotesk, sans-serif", color="#f1f5f9"),
        ),
    )
    layout.update(kwargs)
    return layout


# ────────────────────────────────────────────────────────────
# SENTIMENT PIE / DONUT CHART
# ────────────────────────────────────────────────────────────
def sentiment_pie(label_counts: dict, title: str = "Sentiment Distribution") -> go.Figure:
    """
    Premium donut chart with neon segment glows.
    """
    labels     = list(label_counts.keys())
    values     = list(label_counts.values())
    colour_map = [COLOURS.get(l, "#888") for l in labels]

    fig = go.Figure(go.Pie(
        labels        = [l.capitalize() for l in labels],
        values        = values,
        hole          = 0.62,
        marker        = dict(
            colors = colour_map,
            line   = dict(color="rgba(2,8,23,0.8)", width=3),
        ),
        textinfo      = "label+percent",
        textfont      = dict(family="Space Grotesk, sans-serif", size=12, color="#f1f5f9"),
        hovertemplate = "<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        rotation      = 90,
    ))

    total = sum(values)
    fig.update_layout(
        **_base_layout(
            title  = dict(
                text = f"<b>{title}</b>",
                font = dict(size=14, color=COLOURS["text"], family="Space Grotesk, sans-serif"),
                x    = 0.5,
            ),
            margin = dict(t=50, b=10, l=10, r=10),
            legend = dict(
                bgcolor     = "rgba(0,0,0,0)",
                font        = dict(color=COLOURS["muted"]),
                orientation = "h",
                yanchor     = "bottom",
                y           = -0.15,
                xanchor     = "center",
                x           = 0.5,
            ),
            annotations = [dict(
                text      = f"<b style='font-size:18px'>{total}</b><br><span style='color:#475569'>articles</span>",
                x=0.5, y=0.5,
                font      = dict(size=13, color=COLOURS["text"], family="JetBrains Mono, monospace"),
                showarrow = False,
            )],
        )
    )
    return fig


# ────────────────────────────────────────────────────────────
# SENTIMENT TIMELINE
# ────────────────────────────────────────────────────────────
def sentiment_timeline(news_df: pd.DataFrame) -> go.Figure:
    """
    Neon area chart of daily average compound sentiment score.
    """
    if "compound" not in news_df.columns or "date" not in news_df.columns:
        return _empty_chart("No sentiment timeline data available")

    daily = (news_df.groupby("date")["compound"]
                    .mean()
                    .reset_index()
                    .sort_values("date"))

    pos_mask = daily["compound"] >= 0
    fig = go.Figure()

    # Positive fill area
    fig.add_trace(go.Scatter(
        x            = daily["date"],
        y            = daily["compound"].clip(lower=0),
        mode         = "none",
        fill         = "tozeroy",
        fillcolor    = "rgba(74,222,128,0.08)",
        showlegend   = False,
        hoverinfo    = "skip",
    ))

    # Negative fill area
    fig.add_trace(go.Scatter(
        x            = daily["date"],
        y            = daily["compound"].clip(upper=0),
        mode         = "none",
        fill         = "tozeroy",
        fillcolor    = "rgba(248,113,113,0.08)",
        showlegend   = False,
        hoverinfo    = "skip",
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x            = daily["date"],
        y            = daily["compound"],
        mode         = "lines+markers",
        name         = "Avg Sentiment",
        line         = dict(color=COLOURS["accent"], width=2.5, shape="spline"),
        marker       = dict(
            color  = [COLOURS["positive"] if v >= 0 else COLOURS["negative"] for v in daily["compound"]],
            size   = 7,
            line   = dict(width=1.5, color="rgba(2,8,23,0.8)"),
        ),
        hovertemplate = "📅 %{x}<br>Score: <b>%{y:.3f}</b><extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(100,116,139,0.4)", line_width=1)

    fig.update_layout(
        **_base_layout(
            title       = dict(
                text = "<b>Sentiment Timeline</b>",
                font = dict(size=14, color=COLOURS["text"]),
                x    = 0.5,
            ),
            xaxis_title = "Date",
            yaxis_title = "Compound Score",
            yaxis       = dict(
                range       = [-1.1, 1.1],
                gridcolor   = COLOURS["grid"],
                zerolinecolor = "rgba(56,189,248,0.1)",
                tickfont    = dict(family="JetBrains Mono, monospace", color=COLOURS["muted"]),
            ),
            xaxis       = dict(
                gridcolor = COLOURS["grid"],
                tickfont  = dict(family="Space Grotesk, sans-serif", color=COLOURS["muted"]),
            ),
            showlegend  = False,
        )
    )
    return fig


# ────────────────────────────────────────────────────────────
# STOCK PRICE — CANDLESTICK
# ────────────────────────────────────────────────────────────
def stock_price_chart(stock_df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Premium candlestick with neon candles and SMA overlay.
    """
    if stock_df.empty or "Ticker" not in stock_df.columns:
        return _empty_chart(f"No price data available for {ticker}")
    df = stock_df[stock_df["Ticker"] == ticker].copy().sort_values("Date")
    if df.empty:
        return _empty_chart(f"No price data for {ticker}")

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x          = df["Date"],
        open       = df["Open"],
        high       = df["High"],
        low        = df["Low"],
        close      = df["Close"],
        increasing = dict(
            line      = dict(color=COLOURS["buy"], width=1.5),
            fillcolor = "rgba(74,222,128,0.7)",
        ),
        decreasing = dict(
            line      = dict(color=COLOURS["sell"], width=1.5),
            fillcolor = "rgba(248,113,113,0.7)",
        ),
        name       = ticker,
        hovertext  = ticker,
    ))

    # 20-day SMA overlay
    if len(df) >= 20:
        df["SMA20"] = df["Close"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x    = df["Date"],
            y    = df["SMA20"],
            mode = "lines",
            name = "SMA-20",
            line = dict(color=COLOURS["purple"], width=1.5, dash="dot"),
            hovertemplate = "SMA-20: <b>$%{y:.2f}</b><extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(
            title       = dict(
                text = f"<b>{ticker}</b> — Price Trend",
                font = dict(size=14, color=COLOURS["text"]),
                x    = 0.5,
            ),
            xaxis_title = "Date",
            yaxis_title = "Price (USD)",
            yaxis       = dict(
                gridcolor = COLOURS["grid"],
                tickfont  = dict(family="JetBrains Mono, monospace", color=COLOURS["muted"]),
                tickprefix= "$",
            ),
            xaxis       = dict(
                gridcolor              = COLOURS["grid"],
                rangeslider_visible   = False,
                tickfont              = dict(family="Space Grotesk, sans-serif", color=COLOURS["muted"]),
            ),
        )
    )
    return fig


# ────────────────────────────────────────────────────────────
# BUY / HOLD / SELL GAUGE
# ────────────────────────────────────────────────────────────
def signal_gauge(signal_name: str, confidence: float) -> go.Figure:
    """
    Premium neon gauge meter.
    """
    colour_map = {
        "Buy":  COLOURS["buy"],
        "Hold": COLOURS["hold"],
        "Sell": COLOURS["sell"],
    }
    colour = colour_map.get(signal_name, COLOURS["accent"])

    # Convert hex to rgba to support alpha in Plotly indicator bordercolor
    hex_clean = colour.lstrip("#")
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)
    border_colour = f"rgba({r}, {g}, {b}, 0.27)"

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = confidence * 100,
        title  = dict(
            text = f"<b>{signal_name}</b>",
            font = dict(size=20, color=colour, family="Space Grotesk, sans-serif"),
        ),
        number = dict(
            suffix    = "%",
            font      = dict(size=32, color=colour, family="JetBrains Mono, monospace"),
        ),
        gauge  = dict(
            axis    = dict(
                range     = [0, 100],
                tickwidth = 1,
                tickcolor = COLOURS["muted"],
                tickfont  = dict(family="JetBrains Mono, monospace", color=COLOURS["muted"], size=10),
            ),
            bar     = dict(
                color     = colour,
                thickness = 0.28,
                line      = dict(color=colour, width=2),
            ),
            bgcolor      = "rgba(13,27,42,0.6)",
            borderwidth  = 1,
            bordercolor  = border_colour,
            steps        = [
                dict(range=[0,  33],  color="rgba(248,113,113,0.08)"),
                dict(range=[33, 66],  color="rgba(251,191,36,0.08)"),
                dict(range=[66, 100], color="rgba(74,222,128,0.08)"),
            ],
            threshold = dict(
                line      = dict(color=colour, width=3),
                thickness = 0.8,
                value     = confidence * 100,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(color=COLOURS["text"], family="Space Grotesk, sans-serif"),
        margin        = dict(t=50, b=10, l=30, r=30),
        height        = 280,
    )
    return fig


# ────────────────────────────────────────────────────────────
# PROBABILITY BAR CHART
# ────────────────────────────────────────────────────────────
def probability_bar(probabilities: dict) -> go.Figure:
    """
    Horizontal neon bar chart for signal probabilities.
    """
    labels  = ["Sell", "Hold", "Buy"]
    values  = [probabilities.get(l, 0) * 100 for l in labels]
    colours = [COLOURS["sell"], COLOURS["hold"], COLOURS["buy"]]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = labels,
        orientation = "h",
        marker      = dict(
            color = colours,
            line  = dict(width=0),
            opacity = 0.85,
        ),
        text        = [f"{v:.1f}%" for v in values],
        textposition= "outside",
        textfont    = dict(family="JetBrains Mono, monospace", color=COLOURS["text"], size=12),
        hovertemplate = "<b>%{y}</b>: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            xaxis  = dict(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False),
            yaxis  = dict(
                gridcolor = COLOURS["grid"],
                tickfont  = dict(family="Space Grotesk, sans-serif", color=COLOURS["muted"], size=13),
            ),
            margin     = dict(t=10, b=10, l=60, r=60),
            height     = 190,
            showlegend = False,
        )
    )
    return fig


# ────────────────────────────────────────────────────────────
# FEATURE CONTRIBUTION (XAI)
# ────────────────────────────────────────────────────────────
def feature_contribution_chart(contributions_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal waterfall XAI chart with neon bars.
    """
    if contributions_df.empty:
        return _empty_chart("No contribution data")

    df = contributions_df.copy()
    colours = [COLOURS["buy"] if c > 0 else COLOURS["sell"]
               for c in df["contribution"]]

    fig = go.Figure(go.Bar(
        x           = df["contribution"],
        y           = df["description"],
        orientation = "h",
        marker      = dict(color=colours, line=dict(width=0), opacity=0.85),
        text        = [f"{v:+.3f}" for v in df["contribution"]],
        textposition= "outside",
        textfont    = dict(family="JetBrains Mono, monospace", color=COLOURS["text"], size=11),
        hovertemplate = "<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_dash="solid", line_color="rgba(100,116,139,0.3)", line_width=1)

    fig.update_layout(
        **_base_layout(
            title  = dict(
                text = "<b>Feature Contributions (XAI)</b>",
                font = dict(size=14, color=COLOURS["text"]),
                x    = 0.5,
            ),
            xaxis  = dict(
                gridcolor = COLOURS["grid"],
                title     = "Contribution Score",
                tickfont  = dict(family="JetBrains Mono, monospace", color=COLOURS["muted"]),
            ),
            yaxis  = dict(
                autorange = "reversed",
                tickfont  = dict(family="Space Grotesk, sans-serif", color=COLOURS["muted"]),
            ),
            margin = dict(t=50, b=40, l=200, r=80),
            height = 340,
        )
    )
    return fig


# ────────────────────────────────────────────────────────────
# SECTOR SENTIMENT
# ────────────────────────────────────────────────────────────
def sector_sentiment_chart(news_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal neon bar chart of average sentiment by sector.
    """
    if "sector" not in news_df.columns or "compound" not in news_df.columns:
        return _empty_chart("Sector data not available")

    by_sector = (news_df.groupby("sector")["compound"]
                        .mean()
                        .reset_index()
                        .sort_values("compound", ascending=True))

    colours = [COLOURS["positive"] if v >= 0 else COLOURS["negative"]
               for v in by_sector["compound"]]

    fig = go.Figure(go.Bar(
        x           = by_sector["compound"],
        y           = by_sector["sector"],
        orientation = "h",
        marker      = dict(color=colours, line=dict(width=0), opacity=0.85),
        text        = [f"{v:+.3f}" for v in by_sector["compound"]],
        textposition= "outside",
        textfont    = dict(family="JetBrains Mono, monospace", color=COLOURS["text"], size=11),
        hovertemplate = "<b>%{y}</b><br>Avg Sentiment: %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_dash="dot", line_color="rgba(100,116,139,0.3)", line_width=1)

    fig.update_layout(
        **_base_layout(
            title  = dict(
                text = "<b>Sector-wise Sentiment</b>",
                font = dict(size=14, color=COLOURS["text"]),
                x    = 0.5,
            ),
            xaxis  = dict(
                gridcolor = COLOURS["grid"],
                title     = "Avg Compound Score",
                tickfont  = dict(family="JetBrains Mono, monospace", color=COLOURS["muted"]),
            ),
            yaxis  = dict(
                gridcolor = COLOURS["grid"],
                tickfont  = dict(family="Space Grotesk, sans-serif", color=COLOURS["muted"]),
            ),
            margin = dict(t=50, b=40, l=130, r=80),
            height = 380,
        )
    )
    return fig


# ────────────────────────────────────────────────────────────
# MODEL COMPARISON — QSVC vs SVM
# ────────────────────────────────────────────────────────────
def model_comparison_chart(qsvc_acc: float, svm_acc: float) -> go.Figure:
    """
    Grouped neon bar — QSVC vs classical SVM accuracy.
    """
    models  = ["⚛️ QSVC (Quantum)", "🔷 Classical SVM"]
    accs    = [qsvc_acc * 100, svm_acc * 100]
    colours = [COLOURS["accent"], COLOURS["purple"]]

    fig = go.Figure(go.Bar(
        x           = models,
        y           = accs,
        marker      = dict(
            color   = colours,
            opacity = 0.85,
            line    = dict(width=0),
        ),
        text        = [f"{a:.1f}%" for a in accs],
        textposition= "outside",
        textfont    = dict(family="JetBrains Mono, monospace", color=COLOURS["text"], size=13),
        width       = 0.45,
        hovertemplate = "<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            title  = dict(
                text = "<b>QSVC vs Classical SVM — Test Accuracy</b>",
                font = dict(size=14, color=COLOURS["text"]),
                x    = 0.5,
            ),
            yaxis  = dict(
                range     = [0, 115],
                gridcolor = COLOURS["grid"],
                title     = "Accuracy (%)",
                tickfont  = dict(family="JetBrains Mono, monospace", color=COLOURS["muted"]),
                ticksuffix= "%",
            ),
            xaxis  = dict(
                gridcolor = COLOURS["grid"],
                tickfont  = dict(family="Space Grotesk, sans-serif", color=COLOURS["text"], size=13),
            ),
            height     = 320,
            showlegend = False,
        )
    )
    return fig


# ────────────────────────────────────────────────────────────
# VOLUME CHART
# ────────────────────────────────────────────────────────────
def volume_chart(stock_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Neon green/red volume bars."""
    if stock_df.empty or "Ticker" not in stock_df.columns:
        return _empty_chart(f"No volume data available for {ticker}")
    df = stock_df[stock_df["Ticker"] == ticker].copy().sort_values("Date")
    if df.empty or "Volume" not in df.columns:
        return _empty_chart(f"No volume data for {ticker}")

    df["daily_return"] = df["Close"].pct_change()
    bar_colours = [COLOURS["buy"] if r >= 0 else COLOURS["sell"]
                   for r in df["daily_return"].fillna(0)]

    fig = go.Figure(go.Bar(
        x           = df["Date"],
        y           = df["Volume"],
        marker      = dict(color=bar_colours, line=dict(width=0), opacity=0.75),
        name        = "Volume",
        hovertemplate = "📅 %{x}<br>Volume: <b>%{y:,.0f}</b><extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            title  = dict(
                text = f"<b>{ticker}</b> — Volume",
                font = dict(size=13, color=COLOURS["text"]),
                x    = 0.5,
            ),
            yaxis  = dict(
                gridcolor = COLOURS["grid"],
                title     = "Volume",
                tickfont  = dict(family="JetBrains Mono, monospace", color=COLOURS["muted"]),
            ),
            xaxis  = dict(
                gridcolor = COLOURS["grid"],
                tickfont  = dict(family="Space Grotesk, sans-serif", color=COLOURS["muted"]),
            ),
            margin     = dict(t=40, b=30, l=60, r=10),
            height     = 190,
            showlegend = False,
        )
    )
    return fig


# ────────────────────────────────────────────────────────────
# HELPER — Empty chart
# ────────────────────────────────────────────────────────────
def _empty_chart(msg: str) -> go.Figure:
    """Styled empty figure with centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text      = f"<b>{msg}</b>",
        xref      = "paper", yref = "paper",
        x=0.5, y=0.5, showarrow=False,
        font      = dict(size=13, color=COLOURS["muted"], family="Space Grotesk, sans-serif"),
    )
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        xaxis = dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis = dict(showgrid=False, showticklabels=False, zeroline=False),
        height = 260,
    )
    return fig
