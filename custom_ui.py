"""
custom_ui.py
============
Module UI/UX thuần túy — chỉ chứa CSS, Animation, Metric Cards,
Banner trạng thái, hàm vẽ biểu đồ nâng cao và Telegram Alert.

NGUYÊN TẮC: File này KHÔNG chứa bất kỳ logic model/data nào.
Mọi dữ liệu đều được truyền vào qua tham số hàm.
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================
# 1. CSS & THEME TOÀN CỤC
# ============================================================
def inject_css():
    """
    Inject CSS nền trắng, font hiện đại, animation mượt mà.
    Gọi một lần duy nhất ở đầu app.py.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background: #f8fafc; color: #0f172a; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.04);
    }
    section[data-testid="stSidebar"] * { color: #0f172a !important; }
    section[data-testid="stSidebar"] .stTextInput input {
        background: #f8fafc;
        border: 1.5px solid #cbd5e1;
        border-radius: 8px;
        font-size: 14px;
        transition: border 0.2s;
    }
    section[data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #e2e8f0;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        font-size: 14px;
        color: #64748b;
        padding: 8px 20px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .metric-card.danger::before  { background: linear-gradient(90deg,#ef4444,#f97316); }
    .metric-card.safe::before    { background: linear-gradient(90deg,#22c55e,#10b981); }
    .metric-card.warning::before { background: linear-gradient(90deg,#f59e0b,#eab308); }
    .metric-card.neutral::before { background: linear-gradient(90deg,#3b82f6,#6366f1); }
    .metric-label { font-size: 11px; font-weight: 600; color: #94a3b8;
                    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #0f172a; line-height: 1; }
    .metric-sub   { font-size: 12px; color: #64748b; margin-top: 6px; }

    /* ── Banner cảnh báo ── */
    .banner {
        border-radius: 10px;
        padding: 14px 20px;
        margin: 12px 0;
        font-size: 14px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
        animation: slideIn 0.4s ease;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .banner.danger  { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
    .banner.safe    { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
    .banner.warning { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
    .banner.info    { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; }

    /* ── Recommendation Box ── */
    .rec-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    }
    .rec-box h3 { font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 12px; }
    .rec-box p  { font-size: 14px; color: #475569; line-height: 1.7; }

    /* ── Backtest KPI ── */
    .kpi-row { display: flex; gap: 12px; margin: 12px 0; flex-wrap: wrap; }
    .kpi-box {
        flex: 1; min-width: 140px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .kpi-label { font-size: 11px; color: #94a3b8; text-transform: uppercase;
                 letter-spacing: 0.6px; margin-bottom: 6px; }
    .kpi-value { font-size: 22px; font-weight: 700; }
    .kpi-up   { color: #16a34a; }
    .kpi-down { color: #dc2626; }
    .kpi-neu  { color: #2563eb; }

    /* ── Divider ── */
    .section-divider {
        height: 1px; background: #e2e8f0;
        margin: 20px 0; border: none;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 2. HEADER / PAGE TITLE
# ============================================================
def render_header(symbol: str, latest_close: float, change_pct: float):
    """
    Render tiêu đề trang với tên mã, giá và % thay đổi.

    Args:
        symbol: Mã cổ phiếu.
        latest_close: Giá đóng cửa gần nhất.
        change_pct: Phần trăm thay đổi so với phiên trước.
    """
    color = "#16a34a" if change_pct >= 0 else "#dc2626"
    arrow = "▲" if change_pct >= 0 else "▼"
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between;
                background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:20px 28px; margin-bottom:16px;
                box-shadow:0 1px 6px rgba(0,0,0,0.05);">
        <div>
            <div style="font-size:28px; font-weight:700; color:#0f172a;">{symbol}</div>
            <div style="font-size:13px; color:#64748b; margin-top:2px;">
                VN30 · ProjectADY_StockDB · Cập nhật {datetime.now().strftime('%H:%M %d/%m/%Y')}
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:32px; font-weight:700; color:#0f172a;">
                {latest_close:,.0f}
            </div>
            <div style="font-size:15px; font-weight:600; color:{color};">
                {arrow} {abs(change_pct):.2f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 3. METRIC CARDS
# ============================================================
def render_metric_cards(prob: float, rsi: float, vol_ratio: float, divergence: bool):
    """
    Hiển thị 4 thẻ chỉ số chính.

    Args:
        prob: Xác suất Bull Trap (0.0 - 1.0).
        rsi: Giá trị RSI(14).
        vol_ratio: Tỷ lệ Volume / Volume MA20.
        divergence: Có phân kỳ OBV hay không.
    """
    if prob > 0.5:
        card_cls, risk_txt = "danger",  "Rủi ro cao"
    elif prob > 0.3:
        card_cls, risk_txt = "warning", "Cần theo dõi"
    else:
        card_cls, risk_txt = "safe",    "Tín hiệu tốt"

    rsi_cls = "danger" if rsi > 70 else ("safe" if rsi < 30 else "neutral")
    vol_cls = "danger" if vol_ratio < 0.7 else ("warning" if vol_ratio < 1.0 else "safe")
    div_cls = "danger" if divergence else "safe"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card {card_cls}">
            <div class="metric-label">Xac suat Bull Trap</div>
            <div class="metric-value">{prob:.1%}</div>
            <div class="metric-sub">{risk_txt}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        rsi_txt = "Qua mua" if rsi > 70 else ("Qua ban" if rsi < 30 else "Trung lap")
        st.markdown(f"""<div class="metric-card {rsi_cls}">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value">{rsi:.1f}</div>
            <div class="metric-sub">{rsi_txt}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        vol_txt = "Khoi luong yeu" if vol_ratio < 0.7 else ("Binh thuong" if vol_ratio < 1.0 else "Manh")
        st.markdown(f"""<div class="metric-card {vol_cls}">
            <div class="metric-label">Volume Ratio</div>
            <div class="metric-value">{vol_ratio:.2f}x</div>
            <div class="metric-sub">{vol_txt}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        div_txt = "Dong tien rut ra" if divergence else "On dinh"
        st.markdown(f"""<div class="metric-card {div_cls}">
            <div class="metric-label">Phan ky OBV</div>
            <div class="metric-value">{"Co" if divergence else "Khong"}</div>
            <div class="metric-sub">{div_txt}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)


# ============================================================
# 4. BANNER TRẠNG THÁI
# ============================================================
def render_status_banner(prob: float, symbol: str, market_trend: str):
    """
    Hiển thị banner cảnh báo dựa trên xác suất và xu hướng thị trường.

    Args:
        prob: Xác suất Bull Trap.
        symbol: Mã cổ phiếu.
        market_trend: 'Uptrend' | 'Downtrend' | 'Sideway'.
    """
    if prob > 0.65:
        cls, icon, msg = "danger", "CANH BAO",\
            f"{symbol} co xac suat Bull Trap rat cao ({prob:.0%}). " \
            f"Thi truong dang {market_trend}. Khuyen nghi KHONG MUA them, xem xet chot loi."
    elif prob > 0.5:
        cls, icon, msg = "warning", "CHU Y",\
            f"{symbol} co dau hieu Bull Trap ({prob:.0%}). " \
            f"Xu huong thi truong: {market_trend}. Theo doi chat che."
    elif prob > 0.3:
        cls, icon, msg = "info", "THEO DOI",\
            f"{symbol} chua co tin hieu ro rang ({prob:.0%}). " \
            f"Thi truong {market_trend}. Quan sat them 1-2 phien."
    else:
        cls, icon, msg = "safe", "AN TOAN",\
            f"{symbol} co xac suat Breakout that cao ({1-prob:.0%}). " \
            f"Thi truong {market_trend}. Co the xem xet vi the Long."

    st.markdown(f"""
    <div class="banner {cls}">
        <strong>[{icon}]</strong> {msg}
    </div>""", unsafe_allow_html=True)


# ============================================================
# 5. BIỂU ĐỒ NÂNG CAO (Candlestick + S/R + Fibonacci + OBV)
# ============================================================
def _compute_support_resistance(df: pd.DataFrame, n_levels: int = 3) -> tuple:
    """
    Tính các mức Kháng cự / Hỗ trợ động dựa trên local extrema.

    Args:
        df: DataFrame OHLCV.
        n_levels: Số mức S/R cần tìm.

    Returns:
        tuple: (support_levels: list, resistance_levels: list)
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    window = max(5, len(df) // 20)

    resistance_candidates = []
    support_candidates    = []

    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window: i + window]):
            resistance_candidates.append(highs[i])
        if lows[i] == min(lows[i - window: i + window]):
            support_candidates.append(lows[i])

    # Cluster gần nhau thành 1 mức
    def cluster(levels: list, tol_pct: float = 0.015) -> list:
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - clustered[-1]) / clustered[-1] > tol_pct:
                clustered.append(lvl)
        return clustered

    res_levels = cluster(resistance_candidates)[-n_levels:]
    sup_levels = cluster(support_candidates)[:n_levels]
    return sup_levels, res_levels


def _compute_fibonacci(df: pd.DataFrame) -> dict:
    """
    Tính các mức Fibonacci Retracement dựa trên Max/Min khung hiển thị.

    Args:
        df: DataFrame OHLCV.

    Returns:
        dict: {level_name: price_value}
    """
    high = df["High"].max()
    low  = df["Low"].min()
    diff = high - low
    ratios = {
        "0.0%":   high,
        "23.6%":  high - 0.236 * diff,
        "38.2%":  high - 0.382 * diff,
        "50.0%":  high - 0.500 * diff,
        "61.8%":  high - 0.618 * diff,
        "78.6%":  high - 0.786 * diff,
        "100.0%": low,
    }
    return ratios


def build_advanced_chart(df: pd.DataFrame, symbol: str, show_fib: bool = True,
                          show_sr: bool = True) -> go.Figure:
    """
    Vẽ biểu đồ tương tác nâng cao:
    - Subplot 1: Candlestick + MA20 + S/R động + Fibonacci + marker Bull Trap.
    - Subplot 2: OBV + OBV_MA20 + điểm phân kỳ.

    Args:
        df: DataFrame đã có features và cột Bull_Trap_Prob, Prediction.
        symbol: Mã cổ phiếu.
        show_fib: Hiển thị Fibonacci hay không.
        show_sr: Hiển thị S/R hay không.

    Returns:
        go.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.68, 0.32],
        subplot_titles=[
            f"<b>{symbol}</b>  —  Gia & Tin hieu",
            "OBV  (On-Balance Volume)",
        ],
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Gia",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        increasing_fillcolor="#16a34a",  decreasing_fillcolor="#dc2626",
        whiskerwidth=0.3,
    ), row=1, col=1)

    # ── MA20 ──
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA20"],
        name="MA20",
        line=dict(color="#2563eb", width=2, dash="dot"),
        opacity=0.85,
    ), row=1, col=1)

    # ── S/R Động ──
    if show_sr and "High" in df.columns:
        sup_levels, res_levels = _compute_support_resistance(df)
        x_range = [df.index[0], df.index[-1]]

        for lvl in sup_levels:
            fig.add_shape(type="line", x0=x_range[0], x1=x_range[1],
                          y0=lvl, y1=lvl, row=1, col=1,
                          line=dict(color="#16a34a", width=1.2, dash="dot"))
            fig.add_annotation(x=x_range[-1], y=lvl, text=f"S {lvl:,.0f}",
                               font=dict(size=10, color="#16a34a"),
                               showarrow=False, xanchor="left", row=1, col=1)

        for lvl in res_levels:
            fig.add_shape(type="line", x0=x_range[0], x1=x_range[1],
                          y0=lvl, y1=lvl, row=1, col=1,
                          line=dict(color="#dc2626", width=1.2, dash="dot"))
            fig.add_annotation(x=x_range[-1], y=lvl, text=f"R {lvl:,.0f}",
                               font=dict(size=10, color="#dc2626"),
                               showarrow=False, xanchor="left", row=1, col=1)

    # ── Fibonacci Retracement ──
    if show_fib:
        fib_levels = _compute_fibonacci(df)
        fib_colors = {
            "23.6%": "#a855f7", "38.2%": "#3b82f6",
            "50.0%": "#f59e0b", "61.8%": "#ef4444", "78.6%": "#ec4899",
        }
        for label, price in fib_levels.items():
            if label in fib_colors:
                fig.add_hline(y=price, line_dash="dash",
                              line_color=fib_colors[label], opacity=0.45,
                              annotation_text=f"Fib {label}",
                              annotation_font_size=9,
                              annotation_font_color=fib_colors[label],
                              row=1, col=1)

    # ── Marker Bull Trap (xác suất > 50%) ──
    if "Bull_Trap_Prob" in df.columns:
        trap_mask = df["Bull_Trap_Prob"] > 0.5
        if trap_mask.any():
            t = df[trap_mask]
            fig.add_trace(go.Scatter(
                x=t.index, y=t["Low"] * 0.98,
                mode="markers", name="Bull Trap",
                marker=dict(symbol="triangle-up", color="#ef4444",
                            size=13, line=dict(color="#ffffff", width=1.5)),
                hovertemplate="<b>%{x}</b><br>Xac suat: %{customdata:.1%}<extra></extra>",
                customdata=t["Bull_Trap_Prob"],
            ), row=1, col=1)

        safe_mask = (df.get("Prediction", pd.Series(0, index=df.index)) == 0) \
                    & (df["Bull_Trap_Prob"] <= 0.3) & df["MA20"].notna()
        if safe_mask.any():
            s = df[safe_mask]
            fig.add_trace(go.Scatter(
                x=s.index, y=s["Low"] * 0.98,
                mode="markers", name="An Toan",
                marker=dict(symbol="triangle-up", color="#16a34a",
                            size=11, opacity=0.8, line=dict(color="#ffffff", width=1)),
                hovertemplate="<b>%{x}</b><br>Xac suat: %{customdata:.1%}<extra></extra>",
                customdata=s["Bull_Trap_Prob"],
            ), row=1, col=1)

    # ── OBV ──
    if "OBV" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["OBV"], name="OBV",
            line=dict(color="#2563eb", width=1.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        ), row=2, col=1)

        if "OBV_MA20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["OBV_MA20"], name="OBV MA20",
                line=dict(color="#d97706", width=1.5, dash="dash"),
            ), row=2, col=1)

        if "OBV_Divergence" in df.columns:
            div_mask = df["OBV_Divergence"] == 1
            if div_mask.any():
                dv = df[div_mask]
                fig.add_trace(go.Scatter(
                    x=dv.index, y=dv["OBV"], mode="markers",
                    name="Phan ky OBV",
                    marker=dict(symbol="circle", color="#ef4444",
                                size=7, line=dict(color="#fff", width=1)),
                ), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#0f172a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.95)", bordercolor="#e2e8f0", borderwidth=1),
        height=720,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(t=60, b=20, l=70, r=80),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor="#f1f5f9", showgrid=True, zeroline=False,
                     linecolor="#e2e8f0", tickfont=dict(color="#64748b", size=11))
    fig.update_yaxes(gridcolor="#f1f5f9", showgrid=True, zeroline=False,
                     linecolor="#e2e8f0", tickfont=dict(color="#64748b", size=11))
    return fig


# ============================================================
# 6. MARKET CONTEXT (VNIndex)
# ============================================================
def analyze_market_context(df_vnindex: pd.DataFrame) -> dict:
    """
    Phân tích xu hướng thị trường từ dữ liệu VNIndex.

    Args:
        df_vnindex: DataFrame OHLCV của VNIndex (index = Date).

    Returns:
        dict: {trend, method, description, color}
    """
    if df_vnindex is None or len(df_vnindex) < 20:
        return {"trend": "Khong xac dinh", "method": "Quan sat",
                "description": "Khong du du lieu VNIndex.", "color": "#64748b"}

    ma20 = df_vnindex["Close"].rolling(20).mean()
    ma50 = df_vnindex["Close"].rolling(50).mean()
    last_close = df_vnindex["Close"].iloc[-1]
    last_ma20  = ma20.iloc[-1]
    last_ma50  = ma50.iloc[-1] if len(ma50.dropna()) > 0 else last_ma20

    # Tính slope MA20 (10 phiên gần nhất)
    slope = (ma20.iloc[-1] - ma20.iloc[-10]) / ma20.iloc[-10] * 100 if len(ma20.dropna()) >= 10 else 0

    if last_close > last_ma20 > last_ma50 and slope > 0.5:
        return {
            "trend": "Uptrend",
            "method": "Momentum / Breakout",
            "description": "VNIndex dang trong xu huong tang. Uu tien mua breakout, dat cut-loss duoi MA20.",
            "color": "#16a34a",
        }
    elif last_close < last_ma20 < last_ma50 and slope < -0.5:
        return {
            "trend": "Downtrend",
            "method": "Phong thu / Giam ti trong",
            "description": "VNIndex dang giam. Han che mo vi the moi. Uu tien bao ve von.",
            "color": "#dc2626",
        }
    else:
        return {
            "trend": "Sideway",
            "method": "Mean Reversion / Mua vung ho tro",
            "description": "VNIndex di ngang. Giao dich theo bien do: mua gan ho tro, ban gan khang cu.",
            "color": "#d97706",
        }


def render_market_context(context: dict):
    """
    Hiển thị Market Context box.

    Args:
        context: Dict kết quả từ analyze_market_context().
    """
    color = context["color"]
    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:20px 24px; margin:12px 0; border-left:4px solid {color};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:11px; font-weight:600; color:#94a3b8;
                            text-transform:uppercase; letter-spacing:0.8px;">
                    Xu Huong Thi Truong (VNIndex)
                </div>
                <div style="font-size:22px; font-weight:700; color:{color}; margin:6px 0;">
                    {context['trend']}
                </div>
                <div style="font-size:13px; color:#0f172a; font-weight:600;">
                    Phuong phap: {context['method']}
                </div>
            </div>
        </div>
        <div style="font-size:13px; color:#475569; margin-top:10px; line-height:1.6;">
            {context['description']}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 7. KHUYẾN NGHỊ & STORYTELLING
# ============================================================
def generate_recommendation(symbol: str, prob: float, rsi: float,
                              vol_ratio: float, divergence: bool,
                              market_trend: str, latest_close: float,
                              ma20: float) -> dict:
    """
    Tự động tạo khuyến nghị và narrative dựa trên tín hiệu kỹ thuật.

    Args:
        symbol: Mã cổ phiếu.
        prob: Xác suất Bull Trap.
        rsi: RSI(14).
        vol_ratio: Tỷ lệ Volume.
        divergence: Phân kỳ OBV.
        market_trend: Xu hướng thị trường.
        latest_close: Giá đóng cửa.
        ma20: Giá trị MA20.

    Returns:
        dict: {action, confidence, story, color}
    """
    # Tính điểm cảnh báo tổng hợp
    score = 0
    signals = []

    if prob > 0.5:
        score += 3
        signals.append(f"Mo hinh AI phat hien xac suat bay gia cao ({prob:.0%})")
    if rsi > 70:
        score += 2
        signals.append(f"RSI({rsi:.0f}) o vung qua mua — ap luc ban manh")
    if vol_ratio < 0.7:
        score += 2
        signals.append(f"Khoi luong suy yeu ({vol_ratio:.2f}x) — thieu dong luc")
    if divergence:
        score += 2
        signals.append("Phan ky OBV — dong tien dang rut ra ngam")
    if latest_close > ma20 * 1.05:
        score += 1
        signals.append(f"Gia vot xa MA20 ({((latest_close/ma20-1)*100):.1f}%) — rui ro dieu chinh")
    if market_trend == "Downtrend":
        score += 1

    # Quyết định
    if score >= 6:
        action, color = "BAN / TRANH MUA", "#dc2626"
        confidence = "Cao"
        intro = f"Ky thuat {symbol} dang phat di nhieu tin hieu canh bao nghiem trong."
        conclusion = "Khuyen nghi KHONG mo vi the moi. Neu dang nam, xem xet chot loi tu 30-50%."
    elif score >= 4:
        action, color = "THEO DOI / CHO", "#d97706"
        confidence = "Trung binh"
        intro = f"{symbol} co mot so dau hieu tieu cuc can quan sat them."
        conclusion = "Chua nen hành dong. Doi xac nhan them 1-2 phien giao dich."
    elif score >= 2:
        action, color = "NAM GIU", "#2563eb"
        confidence = "Trung binh"
        intro = f"{symbol} dang trong trang thai trung tinh voi mot vai rui ro nho."
        conclusion = "Neu da co vi the, co the nam giu voi cut-loss duoi MA20."
    else:
        action, color = "CO THE MUA", "#16a34a"
        confidence = "Kha cao"
        intro = f"{symbol} hien thi nhieu tin hieu ky thuat tich cuc."
        conclusion = "Breakout co xac suat thanh cong cao. Quan ly rui ro voi SL duoi 3%."

    # Narrative
    signal_text = "; ".join(signals) if signals else "Chua co tin hieu dang ke"
    story = (
        f"{intro} Cu the, cac tin hieu phat hien bao gom: {signal_text}. "
        f"Thi truong tong the dang o trang thai {market_trend}, "
        f"{'ho tro' if market_trend == 'Uptrend' else 'gay them ap luc'} "
        f"cho co phieu nay. {conclusion}"
    )

    return {"action": action, "confidence": confidence,
            "story": story, "color": color, "score": score}


def render_recommendation(rec: dict):
    """
    Hiển thị khuyến nghị và narrative.

    Args:
        rec: Dict từ generate_recommendation().
    """
    st.markdown(f"""
    <div class="rec-box">
        <div style="display:flex; justify-content:space-between; align-items:center;
                    margin-bottom:14px;">
            <h3 style="margin:0;">Khuyen Nghi Giao Dich</h3>
            <div style="background:{rec['color']}15; color:{rec['color']};
                        font-weight:700; font-size:15px;
                        border:1.5px solid {rec['color']}40;
                        border-radius:8px; padding:6px 18px;">
                {rec['action']}
            </div>
        </div>
        <div style="font-size:12px; color:#94a3b8; margin-bottom:10px;">
            Do tin cay: <strong style="color:#0f172a;">{rec['confidence']}</strong>
            &nbsp;|&nbsp; Diem canh bao: <strong style="color:#0f172a;">{rec['score']}/9</strong>
        </div>
        <p style="font-size:14px; color:#475569; line-height:1.75; margin:0;">
            {rec['story']}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 8. BACKTEST KPIs
# ============================================================
def compute_backtest_kpis(df: pd.DataFrame) -> dict:
    """
    Tính toán và so sánh hiệu suất AI Strategy vs Buy & Hold.

    AI Strategy: Thoát khi model dự đoán Bull Trap (prob > 0.5),
                 vào lại khi prob < 0.3.
    Buy & Hold: Giữ suốt toàn bộ khung thời gian.

    Args:
        df: DataFrame đã có Bull_Trap_Prob, Close.

    Returns:
        dict chứa các KPI so sánh.
    """
    if "Bull_Trap_Prob" not in df.columns or len(df) < 10:
        return {}

    prices = df["Close"].values
    probs  = df["Bull_Trap_Prob"].values

    # Buy & Hold
    bh_return = (prices[-1] - prices[0]) / prices[0] * 100

    # AI Strategy — simulate
    position  = True
    entry     = prices[0]
    ai_cash   = 1.0
    ai_shares = 1.0 / prices[0]
    trades    = []
    wins      = 0

    for i in range(1, len(prices)):
        if position and probs[i] > 0.5:
            # Thoát vị thế
            proceeds = ai_shares * prices[i]
            pnl      = (prices[i] - entry) / entry * 100
            if pnl > 0:
                wins += 1
            trades.append(pnl)
            ai_cash   = proceeds
            ai_shares = 0
            position  = False
        elif not position and probs[i] < 0.3:
            # Vào lại
            entry     = prices[i]
            ai_shares = ai_cash / prices[i]
            ai_cash   = 0
            position  = True

    # Giá trị cuối
    final_value = ai_cash + ai_shares * prices[-1]
    ai_return   = (final_value - 1.0) * 100

    alpha      = ai_return - bh_return
    win_rate   = (wins / len(trades) * 100) if trades else 0
    n_trades   = len(trades)
    avg_trade  = (sum(trades) / n_trades) if trades else 0

    return {
        "bh_return":  round(bh_return, 2),
        "ai_return":  round(ai_return, 2),
        "alpha":      round(alpha, 2),
        "win_rate":   round(win_rate, 1),
        "n_trades":   n_trades,
        "avg_trade":  round(avg_trade, 2),
    }


def render_backtest_kpis(kpis: dict):
    """
    Hiển thị KPI backtest dạng thẻ so sánh.

    Args:
        kpis: Dict từ compute_backtest_kpis().
    """
    if not kpis:
        st.info("Khong du du lieu de tinh backtest.")
        return

    def cls(v): return "kpi-up" if v > 0 else ("kpi-down" if v < 0 else "kpi-neu")
    def sign(v): return f"+{v}" if v > 0 else str(v)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-box">
            <div class="kpi-label">AI Strategy Return</div>
            <div class="kpi-value {cls(kpis['ai_return'])}">{sign(kpis['ai_return'])}%</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Buy & Hold Return</div>
            <div class="kpi-value {cls(kpis['bh_return'])}">{sign(kpis['bh_return'])}%</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Alpha (AI - B&H)</div>
            <div class="kpi-value {cls(kpis['alpha'])}">{sign(kpis['alpha'])}%</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Win Rate</div>
            <div class="kpi-value kpi-neu">{kpis['win_rate']}%</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">So giao dich</div>
            <div class="kpi-value kpi-neu">{kpis['n_trades']}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">TB moi giao dich</div>
            <div class="kpi-value {cls(kpis['avg_trade'])}">{sign(kpis['avg_trade'])}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_backtest_chart(df: pd.DataFrame) -> go.Figure:
    """
    Vẽ biểu đồ so sánh đường lãi/lỗ AI Strategy vs Buy & Hold.

    Args:
        df: DataFrame có Close và Bull_Trap_Prob.

    Returns:
        go.Figure
    """
    prices = df["Close"].values
    probs  = df.get("Bull_Trap_Prob", pd.Series(0, index=df.index)).values
    dates  = df.index

    # Buy & Hold equity curve
    bh_equity = prices / prices[0] * 100

    # AI equity curve
    position  = True
    ai_equity = [100.0]
    ai_value  = 100.0
    ai_shares = 1.0 / prices[0]
    ai_cash   = 0.0

    for i in range(1, len(prices)):
        if position and probs[i] > 0.5:
            ai_cash   = ai_shares * prices[i]
            ai_shares = 0
            position  = False
        elif not position and probs[i] < 0.3:
            ai_shares = ai_cash / prices[i]
            ai_cash   = 0
            position  = True
        current = ai_cash + ai_shares * prices[i]
        ai_equity.append(current / (100 / 100) if prices[0] == 0 else current * 100 / prices[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=bh_equity,
        name="Buy & Hold", mode="lines",
        line=dict(color="#94a3b8", width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(148,163,184,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=ai_equity,
        name="AI Strategy", mode="lines",
        line=dict(color="#2563eb", width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color="#e2e8f0")

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
        font=dict(family="Inter, Arial, sans-serif", color="#0f172a"),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=40, b=20, l=60, r=20),
        yaxis_title="Gia tri danh muc (goc = 100)",
        transition=dict(duration=300),
    )
    fig.update_xaxes(gridcolor="#f1f5f9")
    fig.update_yaxes(gridcolor="#f1f5f9")
    return fig


# ============================================================
# 9. TELEGRAM ALERT
# ============================================================
def send_telegram_alert(bot_token: str, chat_id: str,
                         symbol: str, prob: float,
                         rec_action: str, market_trend: str) -> bool:
    """
    Gửi cảnh báo Telegram khi xác suất Bull Trap vượt ngưỡng.
    Dùng session_state để tránh spam — mỗi mã chỉ gửi 1 lần/phiên.

    Args:
        bot_token: Token Telegram Bot.
        chat_id: Chat ID nhận tin nhắn.
        symbol: Mã cổ phiếu.
        prob: Xác suất Bull Trap.
        rec_action: Khuyến nghị (BAN / THEO DOI...).
        market_trend: Xu hướng thị trường.

    Returns:
        bool: True nếu gửi thành công.
    """
    # Kiểm tra đã gửi trong phiên này chưa
    alert_key = f"telegram_sent_{symbol}"
    if st.session_state.get(alert_key, False):
        return False

    message = (
        f"CANH BAO BULL TRAP\n"
        f"Ma: {symbol}\n"
        f"Xac suat Bay gia: {prob:.1%}\n"
        f"Khuyen nghi: {rec_action}\n"
        f"Thi truong: {market_trend}\n"
        f"Thoi gian: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}\n"
        f"[Bull Trap Detector - VN30]"
    )
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
        if resp.status_code == 200:
            st.session_state[alert_key] = True  # Đánh dấu đã gửi
            return True
        return False
    except Exception:
        return False


def render_telegram_section(symbol: str, prob: float,
                              rec_action: str, market_trend: str):
    """
    Hiển thị form cấu hình và nút gửi Telegram Alert.

    Args:
        symbol, prob, rec_action, market_trend: Thông tin để gửi alert.
    """
    with st.expander("Cau hinh Telegram Alert"):
        col1, col2 = st.columns(2)
        with col1:
            bot_token = st.text_input("Bot Token", type="password",
                                       placeholder="110201543:AAHdqT...")
        with col2:
            chat_id = st.text_input("Chat ID", placeholder="-100123456789")

        threshold = st.slider("Nguong canh bao (%)", 30, 80, 50, 5)
        auto_send = st.checkbox("Tu dong gui khi vuot nguong", value=False)

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            manual_send = st.button("Gui thu ngay", use_container_width=True)

        if bot_token and chat_id:
            should_send = (auto_send and prob * 100 >= threshold) or manual_send
            if should_send:
                ok = send_telegram_alert(bot_token, chat_id,
                                          symbol, prob, rec_action, market_trend)
                with col_status:
                    if ok:
                        st.success("Da gui canh bao Telegram thanh cong!")
                    elif st.session_state.get(f"telegram_sent_{symbol}", False):
                        st.info("Da gui trong phien nay — tranh spam.")
                    else:
                        st.error("Gui that bai. Kiem tra lai Bot Token / Chat ID.")
        else:
            with col_status:
                st.caption("Nhap Bot Token va Chat ID de kich hoat.")


# ============================================================
# 10. TRADE DEMO — DANH MỤC ẢO VỐN 100 TỶ
# ============================================================

INITIAL_CAPITAL = 100_000_000_000  # 100 tỷ VNĐ
TRADE_LOG_KEY   = "trade_demo_log"
PORTFOLIO_KEY   = "trade_demo_portfolio"
CAPITAL_KEY     = "trade_demo_capital"


# File lưu trạng thái demo (persist qua refresh)
DEMO_SAVE_FILE = "trade_demo_state.json"


def _save_demo_state():
    """Lưu trạng thái trade demo ra file JSON để persist qua refresh."""
    import json
    state = {
        "capital":   st.session_state.get(CAPITAL_KEY, INITIAL_CAPITAL),
        "portfolio": st.session_state.get(PORTFOLIO_KEY, {}),
        "log":       st.session_state.get(TRADE_LOG_KEY, []),
    }
    try:
        with open(DEMO_SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_demo_state():
    """Load trạng thái trade demo từ file JSON (gọi khi init)."""
    import json
    if not os.path.exists(DEMO_SAVE_FILE):
        return None
    try:
        with open(DEMO_SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def init_trade_demo():
    """
    Khởi tạo session_state cho Trade Demo.
    Ưu tiên load từ file JSON để tồn tại qua refresh.
    Vốn ban đầu: 100 tỷ VNĐ.
    """
    if CAPITAL_KEY not in st.session_state:
        saved = _load_demo_state()
        if saved:
            st.session_state[CAPITAL_KEY]   = saved.get("capital",   INITIAL_CAPITAL)
            st.session_state[PORTFOLIO_KEY] = saved.get("portfolio", {})
            st.session_state[TRADE_LOG_KEY] = saved.get("log",       [])
        else:
            st.session_state[CAPITAL_KEY]   = INITIAL_CAPITAL
            st.session_state[PORTFOLIO_KEY] = {}
            st.session_state[TRADE_LOG_KEY] = []


def _get_portfolio() -> dict:
    """Trả về danh mục hiện tại từ session_state."""
    return st.session_state.get(PORTFOLIO_KEY, {})


def _get_capital() -> float:
    """Trả về tiền mặt hiện có."""
    return st.session_state.get(CAPITAL_KEY, INITIAL_CAPITAL)


def _get_log() -> list:
    """Trả về lịch sử giao dịch."""
    return st.session_state.get(TRADE_LOG_KEY, [])


def execute_buy(symbol: str, price: float, qty: int) -> dict:
    """
    Thực hiện lệnh MUA ảo.

    Args:
        symbol: Mã cổ phiếu.
        price: Giá mua (VNĐ/cp).
        qty: Số lượng cổ phiếu (đơn vị: cp, bội số 100).

    Returns:
        dict: {success, message, cost}
    """
    if qty <= 0 or qty % 100 != 0:
        return {"success": False, "message": "So luong phai la boi so 100 va lon hon 0.", "cost": 0}

    cost    = price * qty
    capital = _get_capital()

    if cost > capital:
        return {
            "success": False,
            "message": f"Khong du von. Can {cost:,.0f} | Co {capital:,.0f} VND",
            "cost": 0,
        }

    # Cập nhật vốn
    st.session_state[CAPITAL_KEY] = capital - cost

    # Cập nhật danh mục (tính giá vốn bình quân)
    portfolio = _get_portfolio()
    if symbol in portfolio:
        old_qty  = portfolio[symbol]["qty"]
        old_cost = portfolio[symbol]["avg_cost"]
        new_qty  = old_qty + qty
        new_cost = (old_qty * old_cost + qty * price) / new_qty
        portfolio[symbol]["qty"]      = new_qty
        portfolio[symbol]["avg_cost"] = new_cost
    else:
        portfolio[symbol] = {
            "qty":       qty,
            "avg_cost":  price,
            "first_buy": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
    st.session_state[PORTFOLIO_KEY] = portfolio

    # Ghi log
    log_entry = {
        "time":   datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "action": "MUA",
        "symbol": symbol,
        "qty":    qty,
        "price":  price,
        "value":  cost,
        "note":   f"Mua {qty:,} cp @ {price:,.0f}",
    }
    st.session_state[TRADE_LOG_KEY].append(log_entry)
    _save_demo_state()  # Persist qua refresh

    return {
        "success": True,
        "message": f"Da mua {qty:,} cp {symbol} @ {price:,.0f} | Tong: {cost:,.0f} VND",
        "cost": cost,
    }


def execute_sell(symbol: str, price: float, qty: int) -> dict:
    """
    Thực hiện lệnh BÁN ảo.

    Args:
        symbol: Mã cổ phiếu.
        price: Giá bán (VNĐ/cp).
        qty: Số lượng cổ phiếu.

    Returns:
        dict: {success, message, pnl, pnl_pct}
    """
    portfolio = _get_portfolio()

    if symbol not in portfolio:
        return {"success": False, "message": f"Khong co {symbol} trong danh muc.", "pnl": 0, "pnl_pct": 0}

    held_qty = portfolio[symbol]["qty"]
    if qty <= 0 or qty % 100 != 0:
        return {"success": False, "message": "So luong phai la boi so 100.", "pnl": 0, "pnl_pct": 0}
    if qty > held_qty:
        return {"success": False, "message": f"Chi co {held_qty:,} cp, khong du de ban {qty:,} cp.", "pnl": 0, "pnl_pct": 0}

    avg_cost = portfolio[symbol]["avg_cost"]
    proceeds = price * qty
    pnl      = (price - avg_cost) * qty
    pnl_pct  = (price - avg_cost) / avg_cost * 100

    # Cập nhật vốn
    st.session_state[CAPITAL_KEY] = _get_capital() + proceeds

    # Cập nhật danh mục
    new_qty = held_qty - qty
    if new_qty == 0:
        del portfolio[symbol]
    else:
        portfolio[symbol]["qty"] = new_qty
    st.session_state[PORTFOLIO_KEY] = portfolio

    # Ghi log
    log_entry = {
        "time":   datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "action": "BAN",
        "symbol": symbol,
        "qty":    qty,
        "price":  price,
        "value":  proceeds,
        "note":   f"Ban {qty:,} cp @ {price:,.0f} | PnL: {pnl:+,.0f} ({pnl_pct:+.1f}%)",
    }
    st.session_state[TRADE_LOG_KEY].append(log_entry)
    _save_demo_state()  # Persist qua refresh

    return {
        "success": True,
        "message": f"Da ban {qty:,} cp {symbol} @ {price:,.0f}",
        "pnl":     pnl,
        "pnl_pct": pnl_pct,
    }


def reset_demo():
    """Reset toàn bộ danh mục về trạng thái ban đầu và xóa file lưu."""
    import json
    st.session_state[CAPITAL_KEY]   = INITIAL_CAPITAL
    st.session_state[PORTFOLIO_KEY] = {}
    st.session_state[TRADE_LOG_KEY] = []
    try:
        if os.path.exists(DEMO_SAVE_FILE):
            os.remove(DEMO_SAVE_FILE)
    except Exception:
        pass


def compute_portfolio_summary(current_prices: dict) -> dict:
    """
    Tính tổng quan danh mục với giá thị trường hiện tại.

    Args:
        current_prices: {symbol: current_price} giá thị trường.

    Returns:
        dict: Tổng hợp NAV, PnL, tỷ trọng từng mã.
    """
    portfolio    = _get_portfolio()
    capital      = _get_capital()
    rows         = []
    total_market = 0.0
    total_cost   = 0.0

    for sym, info in portfolio.items():
        cur_price  = current_prices.get(sym, info["avg_cost"])
        market_val = cur_price * info["qty"]
        cost_val   = info["avg_cost"] * info["qty"]
        pnl        = market_val - cost_val
        pnl_pct    = pnl / cost_val * 100 if cost_val > 0 else 0

        total_market += market_val
        total_cost   += cost_val

        rows.append({
            "symbol":     sym,
            "qty":        info["qty"],
            "avg_cost":   info["avg_cost"],
            "cur_price":  cur_price,
            "market_val": market_val,
            "cost_val":   cost_val,
            "pnl":        pnl,
            "pnl_pct":    pnl_pct,
            "first_buy":  info.get("first_buy", ""),
        })

    nav          = capital + total_market
    total_pnl    = total_market - total_cost
    total_return = (nav - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {
        "rows":         rows,
        "cash":         capital,
        "total_market": total_market,
        "total_cost":   total_cost,
        "total_pnl":    total_pnl,
        "nav":          nav,
        "total_return": total_return,
        "initial":      INITIAL_CAPITAL,
    }


def render_trade_demo(symbol: str, current_price: float, prob: float):
    """
    Render toàn bộ Tab Trade Demo:
    - Tổng quan NAV / PnL / Cash
    - Lệnh Mua / Bán
    - Bảng danh mục
    - Lịch sử giao dịch
    - Biểu đồ tỷ trọng

    Args:
        symbol: Mã đang xem.
        current_price: Giá hiện tại của mã đang xem.
        prob: Xác suất Bull Trap (để hiện gợi ý).
    """
    import plotly.express as px

    init_trade_demo()

    # Giá thị trường cho tất cả mã trong danh mục
    portfolio     = _get_portfolio()
    current_prices = {sym: current_price if sym == symbol
                      else info["avg_cost"]   # Dùng giá vốn nếu không có giá thật
                      for sym, info in portfolio.items()}
    current_prices[symbol] = current_price

    summary = compute_portfolio_summary(current_prices)

    # ── TỔNG QUAN NAV ──────────────────────────────────────
    ret_color = "#16a34a" if summary["total_return"] >= 0 else "#dc2626"
    pnl_color = "#16a34a" if summary["total_pnl"]   >= 0 else "#dc2626"
    sign      = "+" if summary["total_pnl"] >= 0 else ""

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px;">
        <div class="metric-card neutral">
            <div class="metric-label">Von dau tu ban dau</div>
            <div class="metric-value" style="font-size:18px;">
                {INITIAL_CAPITAL/1e9:.0f} ty
            </div>
            <div class="metric-sub">100,000,000,000 VND</div>
        </div>
        <div class="metric-card neutral">
            <div class="metric-label">NAV Hien tai</div>
            <div class="metric-value" style="font-size:18px; color:#2563eb;">
                {summary['nav']/1e9:.2f} ty
            </div>
            <div class="metric-sub">{summary['nav']:,.0f} VND</div>
        </div>
        <div class="metric-card {'safe' if summary['total_pnl'] >= 0 else 'danger'}">
            <div class="metric-label">Loi/Lo tong</div>
            <div class="metric-value" style="font-size:18px; color:{pnl_color};">
                {sign}{summary['total_pnl']/1e6:,.0f}M
            </div>
            <div class="metric-sub">{sign}{summary['total_return']:.2f}% so von ban dau</div>
        </div>
        <div class="metric-card neutral">
            <div class="metric-label">Tien mat con lai</div>
            <div class="metric-value" style="font-size:18px;">
                {summary['cash']/1e9:.2f} ty
            </div>
            <div class="metric-sub">{summary['cash']/summary['nav']*100:.1f}% NAV</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── LỆNH GIAO DỊCH ─────────────────────────────────────
    st.markdown("#### Dat lenh giao dich")

    # Gợi ý dựa trên AI
    if prob > 0.5:
        st.markdown(f"""<div class="banner danger">
            AI canh bao: {symbol} co xac suat Bull Trap {prob:.0%}.
            Nen xem xet BAN bot hoac tranh MUA them.
        </div>""", unsafe_allow_html=True)
    elif prob < 0.3:
        st.markdown(f"""<div class="banner safe">
            AI: {symbol} co tin hieu tich cuc ({1-prob:.0%} Breakout that).
            Co the xem xet MUA.
        </div>""", unsafe_allow_html=True)

    col_info, col_buy, col_sell = st.columns([1, 1, 1])

    with col_info:
        st.markdown(f"""
        <div class="rec-box" style="padding:16px;">
            <div class="metric-label">Ma dang xem</div>
            <div style="font-size:22px; font-weight:700; color:#0f172a;">{symbol}</div>
            <div style="font-size:13px; color:#64748b; margin-top:4px;">
                Gia hien tai: <strong>{current_price:,.0f}</strong> VND
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:8px;">
                Tien mat: {summary['cash']/1e9:.2f} ty VND<br>
                Max mua: ~{int(summary['cash'] / current_price / 100) * 100:,} cp
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_buy:
        st.markdown("**Lenh MUA**")
        buy_price = st.number_input(
            "Gia mua (VND)", value=float(round(current_price)),
            step=100.0, format="%.0f", key="buy_price"
        )
        buy_qty = st.number_input(
            "So luong (boi so 100)", value=100,
            step=100, min_value=100, key="buy_qty"
        )
        buy_total = buy_price * buy_qty
        st.caption(f"Tong tien: {buy_total:,.0f} VND ({buy_total/1e6:.1f}M)")

        if st.button("MUA", use_container_width=True, type="primary", key="btn_buy"):
            result = execute_buy(symbol, buy_price, buy_qty)
            if result["success"]:
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result["message"])

    with col_sell:
        st.markdown("**Lenh BAN**")
        held = portfolio.get(symbol, {}).get("qty", 0)
        avg  = portfolio.get(symbol, {}).get("avg_cost", current_price)

        sell_price = st.number_input(
            "Gia ban (VND)", value=float(round(current_price)),
            step=100.0, format="%.0f", key="sell_price"
        )
        sell_qty = st.number_input(
            "So luong (boi so 100)", value=min(100, held) if held >= 100 else 100,
            step=100, min_value=100, key="sell_qty"
        )
        if held > 0:
            est_pnl = (sell_price - avg) * sell_qty
            pnl_sign = "+" if est_pnl >= 0 else ""
            pnl_col  = "#16a34a" if est_pnl >= 0 else "#dc2626"
            st.caption(f"Dang nam: {held:,} cp | Gia von: {avg:,.0f}")
            st.markdown(
                f"<span style='font-size:12px;color:{pnl_col};'>"
                f"Du kien L/L: {pnl_sign}{est_pnl:,.0f} VND</span>",
                unsafe_allow_html=True
            )
        else:
            st.caption(f"Khong co {symbol} trong danh muc")

        if st.button("BAN", use_container_width=True, key="btn_sell"):
            result = execute_sell(symbol, sell_price, sell_qty)
            if result["success"]:
                pnl_str = f"{result['pnl']:+,.0f} VND ({result['pnl_pct']:+.1f}%)"
                if result["pnl"] >= 0:
                    st.success(f"{result['message']} | Loi: {pnl_str}")
                else:
                    st.warning(f"{result['message']} | Lo: {pnl_str}")
                st.rerun()
            else:
                st.error(result["message"])

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── BẢNG DANH MỤC ──────────────────────────────────────
    st.markdown("#### Danh muc hien tai")

    if not summary["rows"]:
        st.info("Danh muc trong. Dat lenh MUA de bat dau giao dich demo.")
    else:
        rows_display = []
        for r in summary["rows"]:
            pnl_sign = "+" if r["pnl"] >= 0 else ""
            rows_display.append({
                "Ma CP":        r["symbol"],
                "So luong":     f"{r['qty']:,}",
                "Gia von":      f"{r['avg_cost']:,.0f}",
                "Gia TT":       f"{r['cur_price']:,.0f}",
                "Gia tri TT":   f"{r['market_val']/1e6:,.1f}M",
                "L/L":          f"{pnl_sign}{r['pnl']/1e6:,.2f}M",
                "L/L %":        f"{pnl_sign}{r['pnl_pct']:.2f}%",
                "Mua tu":       r["first_buy"],
            })
        df_port = pd.DataFrame(rows_display)
        st.dataframe(df_port, use_container_width=True, hide_index=True)

        # Biểu đồ tỷ trọng danh mục theo NAV thực tế
        if len(summary["rows"]) > 0:
            nav_total  = summary["nav"]
            pie_labels = []
            pie_values = []
            pie_texts  = []

            for r in summary["rows"]:
                pct = r["market_val"] / nav_total * 100 if nav_total > 0 else 0
                pie_labels.append(r["symbol"])
                pie_values.append(r["market_val"])
                pie_texts.append(
                    f"{r['symbol']}<br>{pct:.1f}%<br>{r['market_val']/1e6:.0f}M"
                )

            # Tiền mặt
            cash_pct = summary["cash"] / nav_total * 100 if nav_total > 0 else 100
            pie_labels.append("Tien mat")
            pie_values.append(summary["cash"])
            pie_texts.append(
                f"Tien mat<br>{cash_pct:.1f}%<br>{summary['cash']/1e9:.2f}ty"
            )

            colors = ["#3b82f6","#10b981","#f59e0b","#ef4444",
                      "#8b5cf6","#06b6d4","#f97316","#84cc16",
                      "#ec4899","#94a3b8"]

            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_values,
                text=pie_texts,
                textinfo="text",
                textposition="inside",
                hole=0.5,
                marker=dict(
                    colors=colors[:len(pie_labels)],
                    line=dict(color="#ffffff", width=2),
                ),
                hovertemplate="<b>%{label}</b><br>Gia tri: %{value:,.0f} VND<br>Ty trong: %{percent}<extra></extra>",
            ))

            # Annotation NAV ở giữa donut
            fig_pie.add_annotation(
                text=f"NAV<br><b>{nav_total/1e9:.2f}ty</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color="#0f172a", family="Inter, Arial"),
                align="center",
            )

            fig_pie.update_layout(
                title=dict(
                    text="Co cau danh muc theo NAV",
                    font=dict(size=14, color="#0f172a"),
                    x=0,
                ),
                template="plotly_white",
                paper_bgcolor="#ffffff",
                font=dict(family="Inter, Arial, sans-serif", color="#0f172a"),
                height=380,
                margin=dict(t=50, b=60, l=10, r=10),
                legend=dict(
                    orientation="h", yanchor="top", y=-0.08,
                    xanchor="center", x=0.5,
                    font=dict(size=11),
                ),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── LỊCH SỬ GIAO DỊCH ──────────────────────────────────
    col_hist, col_reset = st.columns([4, 1])
    with col_hist:
        st.markdown("#### Lich su giao dich")
    with col_reset:
        if st.button("Reset Demo", key="btn_reset", use_container_width=True):
            reset_demo()
            st.success("Da reset ve von ban dau 100 ty!")
            st.rerun()

    log = _get_log()
    if not log:
        st.caption("Chua co giao dich nao.")
    else:
        df_log = pd.DataFrame(reversed(log))  # Mới nhất lên trên
        df_log = df_log.rename(columns={
            "time": "Thoi gian", "action": "Hanh dong",
            "symbol": "Ma CP",  "qty": "So luong",
            "price": "Gia",     "value": "Gia tri",
            "note": "Chi tiet",
        })
        df_log["So luong"] = df_log["So luong"].apply(lambda x: f"{x:,}")
        df_log["Gia"]      = df_log["Gia"].apply(lambda x: f"{x:,.0f}")
        df_log["Gia tri"]  = df_log["Gia tri"].apply(lambda x: f"{x/1e6:,.1f}M")

        # Tô màu hàng Mua/Bán
        def highlight_action(row):
            color = "#f0fdf4" if row["Hanh dong"] == "MUA" else "#fef2f2"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_log[["Thoi gian", "Hanh dong", "Ma CP",
                    "So luong", "Gia", "Gia tri", "Chi tiet"]].style.apply(
                highlight_action, axis=1
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Tổng kết P&L đã thực hiện
        realized = sum(
            (e["price"] - next(
                (x["price"] for x in log
                 if x["action"] == "MUA" and x["symbol"] == e["symbol"]), e["price"]
            )) * e["qty"]
            for e in log if e["action"] == "BAN"
        )
