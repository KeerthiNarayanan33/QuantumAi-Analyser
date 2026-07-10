"""
============================================================
QuantumSentinel — KPI Card Components  [Pro-Max Redesign]
============================================================
21st-dev design language:
  • Glassmorphism cards with backdrop-filter blur
  • Neon glow borders + gradient top-line accents
  • Space Grotesk + JetBrains Mono typography
  • Animated signal badges with pulse rings
  • Shimmer confidence bars
============================================================
"""


def kpi_card(title: str,
             value: str,
             subtitle: str = "",
             icon: str     = "📊",
             colour: str   = "#38bdf8",
             glow: bool    = True) -> str:
    """
    Glassmorphism KPI card with neon accent border and glow.

    Parameters
    ----------
    title    : e.g. "Market Sentiment"
    value    : e.g. "Positive"
    subtitle : e.g. "Based on 50 articles"
    icon     : emoji icon
    colour   : hex accent colour
    glow     : whether to add a neon glow effect

    Returns
    -------
    str — HTML string (pass to st.markdown with unsafe_allow_html=True)
    """
    glow_style = f"box-shadow: 0 0 40px {colour}1a, 0 4px 24px rgba(0,0,0,0.4);" if glow else "box-shadow: 0 4px 24px rgba(0,0,0,0.4);"

    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(13,27,42,0.85) 0%, rgba(2,8,23,0.9) 100%);
        border: 1px solid {colour}33;
        border-radius: 20px;
        padding: 22px 18px 18px;
        text-align: center;
        {glow_style}
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 158px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
        cursor: default;
    ">
        <!-- Neon top-line accent -->
        <div style="
            position: absolute;
            top: 0; left: 10%; right: 10%;
            height: 1.5px;
            background: linear-gradient(90deg, transparent, {colour}, transparent);
            border-radius: 999px;
        "></div>

        <!-- Subtle mesh background spot -->
        <div style="
            position: absolute;
            top: -30px; right: -30px;
            width: 80px; height: 80px;
            background: radial-gradient(circle, {colour}18 0%, transparent 70%);
            border-radius: 50%;
        "></div>

        <div style="font-size: 28px; margin-bottom: 8px; filter: drop-shadow(0 0 10px {colour}88); position: relative; z-index: 1;">{icon}</div>
        <div style="
            color: #475569;
            font-size: 9.5px;
            font-weight: 700;
            letter-spacing: 1.8px;
            text-transform: uppercase;
            margin-bottom: 8px;
            font-family: 'Space Grotesk', sans-serif;
            position: relative; z-index: 1;
        ">{title}</div>
        <div style="
            color: {colour};
            font-size: 22px;
            font-weight: 800;
            line-height: 1;
            font-family: 'JetBrains Mono', monospace;
            text-shadow: 0 0 25px {colour}66;
            position: relative; z-index: 1;
        ">{value}</div>
        <div style="
            color: #334155;
            font-size: 11px;
            margin-top: 7px;
            font-family: 'Space Grotesk', sans-serif;
            position: relative; z-index: 1;
        ">{subtitle}</div>
    </div>
    """


def signal_badge(signal_name: str) -> str:
    """Large animated glassmorphism badge for Buy / Hold / Sell signal."""
    colours = {
        "Buy":  ("#4ade80", "rgba(74,222,128,0.12)", "#4ade8033"),
        "Hold": ("#fbbf24", "rgba(251,191,36,0.12)",  "#fbbf2433"),
        "Sell": ("#f87171", "rgba(248,113,113,0.12)", "#f8717133"),
    }
    icons = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}
    ring_colours = {"Buy": "#4ade80", "Hold": "#fbbf24", "Sell": "#f87171"}

    colour, bg, border = colours.get(signal_name, ("#38bdf8", "rgba(56,189,248,0.12)", "#38bdf833"))
    icon               = icons.get(signal_name, "⚪")
    ring_col           = ring_colours.get(signal_name, "#38bdf8")

    return f"""
    <style>
    @keyframes qs-pulse-ring {{
        0%   {{ box-shadow: 0 0 0 0 {ring_col}55, 0 0 40px {ring_col}33; }}
        50%  {{ box-shadow: 0 0 0 18px {ring_col}00, 0 0 60px {ring_col}55; }}
        100% {{ box-shadow: 0 0 0 0 {ring_col}00, 0 0 40px {ring_col}33; }}
    }}
    @keyframes qs-float {{
        0%, 100% {{ transform: translateY(0px); }}
        50%       {{ transform: translateY(-4px); }}
    }}
    </style>
    <div style="
        background: {bg};
        border: 1.5px solid {border};
        border-radius: 24px;
        padding: 28px 20px;
        text-align: center;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        animation: qs-pulse-ring 2.5s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    ">
        <!-- Neon top line -->
        <div style="
            position: absolute;
            top: 0; left: 15%; right: 15%;
            height: 2px;
            background: linear-gradient(90deg, transparent, {colour}, transparent);
        "></div>

        <div style="font-size: 52px; animation: qs-float 3s ease-in-out infinite;">{icon}</div>
        <div style="
            color: {colour};
            font-size: 38px;
            font-weight: 900;
            letter-spacing: 5px;
            margin-top: 10px;
            font-family: 'Space Grotesk', sans-serif;
            text-shadow: 0 0 30px {colour}88;
        ">{signal_name.upper()}</div>
    </div>
    """


def confidence_bar(confidence: float, signal_name: str) -> str:
    """Premium gradient confidence bar with shimmer animation."""
    colours = {
        "Buy":  ("#4ade80", "#22c55e"),
        "Hold": ("#fbbf24", "#f59e0b"),
        "Sell": ("#f87171", "#ef4444"),
    }
    c1, c2 = colours.get(signal_name, ("#38bdf8", "#0ea5e9"))
    pct     = round(confidence * 100, 1)

    return f"""
    <style>
    @keyframes qs-shimmer {{
        0%   {{ background-position: -200% center; }}
        100% {{ background-position: 200% center; }}
    }}
    </style>
    <div style="margin: 14px 0;">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        ">
            <span style="
                color: #475569;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 1px;
                text-transform: uppercase;
                font-family: 'Space Grotesk', sans-serif;
            ">Model Confidence</span>
            <span style="
                color: {c1};
                font-weight: 800;
                font-size: 15px;
                font-family: 'JetBrains Mono', monospace;
                text-shadow: 0 0 15px {c1}66;
            ">{pct}%</span>
        </div>
        <div style="
            background: rgba(13,27,42,0.8);
            border-radius: 999px;
            height: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
        ">
            <div style="
                background: linear-gradient(90deg, {c2}, {c1}, {c2});
                background-size: 200% auto;
                width: {pct}%;
                height: 100%;
                border-radius: 999px;
                animation: qs-shimmer 2.5s linear infinite;
                box-shadow: 0 0 10px {c1}88;
            "></div>
        </div>
    </div>
    """


def page_header(title: str, subtitle: str = "") -> str:
    """Cinematic page header with animated mesh gradient and glow text."""
    return f"""
    <style>
    @keyframes qs-mesh-shift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    </style>
    <div style="
        background:
            radial-gradient(ellipse at 0% 50%, rgba(56,189,248,0.12) 0%, transparent 60%),
            radial-gradient(ellipse at 100% 50%, rgba(167,139,250,0.10) 0%, transparent 60%),
            linear-gradient(135deg, rgba(13,27,42,0.95) 0%, rgba(2,8,23,0.98) 100%);
        border-bottom: 1px solid rgba(56,189,248,0.12);
        padding: 30px 0 22px 0;
        margin-bottom: 28px;
        text-align: center;
        position: relative;
        overflow: hidden;
        border-radius: 0 0 16px 16px;
    ">
        <!-- Animated glow orbs -->
        <div style="
            position: absolute;
            top: -20px; left: 10%;
            width: 150px; height: 150px;
            background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
            border-radius: 50%;
            animation: qs-mesh-shift 8s ease infinite;
        "></div>
        <div style="
            position: absolute;
            top: -20px; right: 10%;
            width: 150px; height: 150px;
            background: radial-gradient(circle, rgba(167,139,250,0.08) 0%, transparent 70%);
            border-radius: 50%;
            animation: qs-mesh-shift 8s ease infinite reverse;
        "></div>

        <div style="
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #a78bfa, #4ade80, #38bdf8);
            background-size: 300% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: qs-mesh-shift 5s linear infinite;
            letter-spacing: -0.5px;
            font-family: 'Space Grotesk', sans-serif;
            position: relative; z-index: 1;
        ">{title}</div>
        <div style="
            color: #475569;
            font-size: 12.5px;
            margin-top: 6px;
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.3px;
            position: relative; z-index: 1;
        ">{subtitle}</div>
    </div>
    """


def section_divider(label: str = "") -> str:
    """Neon glow section divider with gradient label."""
    if label:
        return f"""
        <div style="
            display: flex;
            align-items: center;
            margin: 28px 0 18px 0;
            gap: 12px;
        ">
            <div style="flex:1; height:1px; background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2));"></div>
            <div style="
                color: #38bdf8;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 2px;
                text-transform: uppercase;
                white-space: nowrap;
                font-family: 'Space Grotesk', sans-serif;
                text-shadow: 0 0 12px rgba(56,189,248,0.5);
                padding: 4px 12px;
                border: 1px solid rgba(56,189,248,0.2);
                border-radius: 999px;
                background: rgba(56,189,248,0.05);
            ">{label}</div>
            <div style="flex:1; height:1px; background: linear-gradient(90deg, rgba(56,189,248,0.2), transparent);"></div>
        </div>
        """
    return '<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(56,189,248,0.15), transparent); margin:20px 0;"></div>'
