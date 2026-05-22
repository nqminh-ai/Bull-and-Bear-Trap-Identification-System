"""
app.py
======
Streamlit Dashboard — điều phối dữ liệu, KHÔNG chứa UI/UX logic.
Import custom_ui.py để hiển thị, import logic từ ml_pipeline.py.

Kiến trúc:
    app.py          -> Điều phối (Controller)
    custom_ui.py    -> Giao diện (View)
    ml_pipeline.py  -> Logic model/data (Model)

NGUYÊN TẮC: Không chỉnh sửa compute_rsi, compute_obv,
            compute_features, predict_bull_trap.
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyodbc
import streamlit as st

# Import toàn bộ UI từ custom_ui.py
import custom_ui as ui

warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="Bull Trap Detector | VN30",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS toàn cục (chỉ gọi 1 lần)
ui.inject_css()

# ============================================================
# CẤU HÌNH DATABASE & MODEL
# ============================================================
DB_SERVER   = "NGUYEN_MINH"
DB_DATABASE = "ProjectADY_StockDB"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"
MODEL_DIR   = "models"

FEATURE_COLUMNS = [
    "RSI_14", "MA20", "Volume_Ratio", "Upper_Shadow_Ratio",
    "Body_Range_Ratio", "Is_Doji", "OBV", "OBV_MA20",
    "OBV_Divergence", "Price_vs_MA20",
]


# ============================================================
# HÀM KẾT NỐI DATABASE
# ============================================================
def get_connection():
    """Tạo kết nối pyodbc đến SQL Server. Returns pyodbc.Connection."""
    conn_str = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol: str, lookback_days: int = 180) -> pd.DataFrame:
    """
    Lấy dữ liệu OHLCV từ dbo.daily_prices (cache 5 phút).

    Args:
        symbol: Mã cổ phiếu (upper-case).
        lookback_days: Số ngày lịch sử.

    Returns:
        pd.DataFrame OHLCV với index Date.
    """
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query = f"""
        SELECT
            [time]   AS TradingDate,
            [open]   AS OpenPrice,
            [high]   AS HighPrice,
            [low]    AS LowPrice,
            [close]  AS ClosePrice,
            [volume] AS TotalVolume
        FROM dbo.daily_prices
        WHERE [ticker] = '{symbol}'
          AND [time] >= '{cutoff}'
        ORDER BY [time] ASC
    """
    conn = get_connection()
    df   = pd.read_sql(query, conn)
    conn.close()

    df = df.rename(columns={
        "TradingDate": "Date", "OpenPrice": "Open",
        "HighPrice": "High",   "LowPrice": "Low",
        "ClosePrice": "Close", "TotalVolume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


@st.cache_resource(show_spinner=False)
def load_model(symbol: str):
    """Load model .pkl. Returns model hoặc None."""
    path = os.path.join(MODEL_DIR, f"{symbol}_bull_trap.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# LOGIC XỬ LÝ (GỌI TỪ ml_pipeline — KHÔNG CHỈNH SỬA)
# ============================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Tính RSI. Returns pd.Series. [GIỮ NGUYÊN]"""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Tính OBV. Returns pd.Series. [GIỮ NGUYÊN]"""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toàn bộ features Bull Trap. Returns pd.DataFrame. [GIỮ NGUYÊN]"""
    r = df.copy()
    r["RSI_14"]             = compute_rsi(r["Close"])
    r["MA20"]               = r["Close"].rolling(20).mean()
    r["Price_vs_MA20"]      = r["Close"] - r["MA20"]
    r["Volume_MA20"]        = r["Volume"].rolling(20).mean()
    r["Volume_Ratio"]       = r["Volume"] / r["Volume_MA20"].replace(0, np.nan)
    r["Upper_Shadow"]       = r["High"] - r[["Open", "Close"]].max(axis=1)
    r["Body"]               = (r["Close"] - r["Open"]).abs()
    r["Upper_Shadow_Ratio"] = r["Upper_Shadow"] / r["Body"].replace(0, 1e-6)
    r["Range"]              = r["High"] - r["Low"]
    r["Body_Range_Ratio"]   = r["Body"] / r["Range"].replace(0, np.nan)
    r["Is_Doji"]            = (r["Body_Range_Ratio"] < 0.1).astype(int)
    r["OBV"]                = compute_obv(r)
    r["OBV_MA20"]           = r["OBV"].rolling(20).mean()
    price_up                = r["Close"] > r["MA20"]
    money_out               = r["OBV"] < r["OBV_MA20"]
    r["OBV_Divergence"]     = (price_up & money_out).astype(int)
    return r


def predict_bull_trap(df_features: pd.DataFrame, model) -> pd.DataFrame:
    """
    Dự đoán xác suất Bull Trap. [GIỮ NGUYÊN]
    Xử lý IndexError khi model chỉ có 1 nhãn.
    """
    result     = df_features.copy()
    valid_mask = result[FEATURE_COLUMNS].notna().all(axis=1)
    X          = result.loc[valid_mask, FEATURE_COLUMNS]

    predictions   = np.zeros(len(result))
    probabilities = np.zeros(len(result))

    if len(X) > 0 and model is not None:
        try:
            preds = model.predict(X)
            try:
                probas     = model.predict_proba(X)
                proba_trap = probas[:, 1] if probas.shape[1] >= 2 \
                             else (preds == 1).astype(float)
            except (IndexError, AttributeError):
                proba_trap = (preds == 1).astype(float)
            predictions[valid_mask]   = preds
            probabilities[valid_mask] = proba_trap
        except Exception:
            pass

    result["Prediction"]     = predictions
    result["Bull_Trap_Prob"] = probabilities
    return result


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar() -> tuple:
    """
    Render sidebar và trả về tham số người dùng.

    Returns:
        tuple: (symbol, lookback_days, show_fib, show_sr,
                tg_token, tg_chat_id, tg_threshold)
    """
    with st.sidebar:
        st.markdown("### Bull Trap Detector")
        st.markdown("---")

        raw = st.text_input(
            "Ma co phieu",
            value="VNM",
            placeholder="VD: VNM, VPB, HPG...",
        )
        symbol = raw.strip().upper()

        st.markdown("---")
        lookback_days = st.slider("So ngay lich su", 60, 365, 180, 30)

        st.markdown("---")
        st.markdown("**Tuy chon bieu do**")
        show_fib = st.checkbox("Fibonacci Retracement", value=True)
        show_sr  = st.checkbox("Khang cu / Ho tro (S/R)", value=True)

        st.markdown("---")
        st.markdown("**Telegram Alert**")
        tg_token    = st.text_input("Bot Token", type="password", placeholder="Bot token...")
        tg_chat_id  = st.text_input("Chat ID",   placeholder="-100xxx")
        tg_threshold = st.slider("Nguong canh bao (%)", 30, 80, 50, 5)

        st.markdown("---")
        st.caption(f"Cap nhat: {datetime.now().strftime('%H:%M %d/%m/%Y')}")

    return symbol, lookback_days, show_fib, show_sr, tg_token, tg_chat_id, tg_threshold


# ============================================================
# APP CHÍNH
# ============================================================
def main():
    """Hàm điều phối chính — chỉ xử lý luồng dữ liệu, gọi UI từ custom_ui."""

    # Sidebar
    symbol, lookback_days, show_fib, show_sr, \
        tg_token, tg_chat_id, tg_threshold = render_sidebar()

    if not symbol:
        st.warning("Vui long nhap ma co phieu.")
        return

    # Load model
    model = load_model(symbol)
    if model is None:
        st.warning(f"Chua co model cho {symbol}. Chay ml_pipeline.py truoc.")

    # Lấy dữ liệu chính
    with st.spinner(f"Dang tai du lieu {symbol}..."):
        try:
            df_raw = fetch_stock_data(symbol, lookback_days=lookback_days)
        except Exception as e:
            st.error(f"Loi ket noi SQL Server: {e}")
            return

    if df_raw.empty:
        st.error(f"Khong co du lieu cho {symbol}.")
        return

    # Lấy dữ liệu VNIndex cho Market Context
    try:
        df_vni = fetch_stock_data("VNINDEX", lookback_days=lookback_days)
    except Exception:
        df_vni = None

    # Tính features & dự đoán
    df_features  = compute_features(df_raw)
    df_predicted = predict_bull_trap(df_features, model)

    # Lấy giá trị phiên cuối
    latest = df_predicted.dropna(subset=["RSI_14", "Bull_Trap_Prob"]).iloc[-1]
    prob        = float(latest["Bull_Trap_Prob"])
    rsi         = float(latest["RSI_14"])
    vol_ratio   = float(latest["Volume_Ratio"])
    divergence  = bool(latest["OBV_Divergence"])
    last_close  = float(latest["Close"])
    ma20_val    = float(latest["MA20"]) if not pd.isna(latest["MA20"]) else last_close

    prev_close  = float(df_predicted["Close"].iloc[-2]) \
                  if len(df_predicted) >= 2 else last_close
    change_pct  = (last_close - prev_close) / prev_close * 100

    # Market context
    context      = ui.analyze_market_context(df_vni)
    market_trend = context["trend"]

    # Khuyến nghị
    rec = ui.generate_recommendation(
        symbol, prob, rsi, vol_ratio, divergence,
        market_trend, last_close, ma20_val
    )

    # ── HEADER ──
    ui.render_header(symbol, last_close, change_pct)

    # ── STATUS BANNER ──
    ui.render_status_banner(prob, symbol, market_trend)

    # ── METRIC CARDS ──
    ui.render_metric_cards(prob, rsi, vol_ratio, divergence)

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Bieu do ky thuat",
        "Phan tich & Khuyen nghi",
        "Backtest",
        "Trade Demo  (100 ty)",
    ])

    # ── TAB 1: BIỂU ĐỒ ──
    with tab1:
        fig = ui.build_advanced_chart(df_predicted, symbol,
                                      show_fib=show_fib, show_sr=show_sr)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Du lieu chi tiet (20 phien gan nhat)"):
            show_cols = [c for c in [
                "Open", "High", "Low", "Close", "Volume",
                "RSI_14", "MA20", "Volume_Ratio",
                "Upper_Shadow_Ratio", "OBV_Divergence",
                "Bull_Trap_Prob", "Prediction",
            ] if c in df_predicted.columns]
            st.dataframe(
                df_predicted[show_cols].tail(20).style.format({
                    "Open": "{:.2f}", "High": "{:.2f}",
                    "Low": "{:.2f}",  "Close": "{:.2f}",
                    "RSI_14": "{:.1f}", "MA20": "{:.2f}",
                    "Volume_Ratio": "{:.2f}",
                    "Upper_Shadow_Ratio": "{:.2f}",
                    "Bull_Trap_Prob": "{:.1%}",
                }).background_gradient(
                    subset=["Bull_Trap_Prob"], cmap="RdYlGn_r"
                ),
                use_container_width=True,
            )

    # ── TAB 2: PHÂN TÍCH ──
    with tab2:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            ui.render_market_context(context)

        with col_right:
            ui.render_recommendation(rec)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Telegram
        ui.render_telegram_section(
            symbol, prob, rec["action"], market_trend
        )
        # Gửi tự động nếu có config và vượt ngưỡng
        if tg_token and tg_chat_id and prob * 100 >= tg_threshold:
            sent = ui.send_telegram_alert(
                tg_token, tg_chat_id,
                symbol, prob, rec["action"], market_trend
            )
            if sent:
                st.toast(f"Da gui canh bao Telegram cho {symbol}!", icon="")

    # ── TAB 3: BACKTEST ──
    with tab3:
        st.markdown("#### So sanh hieu suat: AI Strategy vs Buy & Hold")
        st.caption(
            "AI Strategy: thoat vi the khi Bull_Trap_Prob > 0.5, "
            "vao lai khi < 0.3. Khong tinh phi giao dich."
        )

        kpis = ui.compute_backtest_kpis(df_predicted)
        ui.render_backtest_kpis(kpis)

        st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

        fig_bt = ui.build_backtest_chart(df_predicted)
        st.plotly_chart(fig_bt, use_container_width=True)

    # ── TAB 4: TRADE DEMO ──
    with tab4:
        ui.render_trade_demo(
            symbol=symbol,
            current_price=last_close,
            prob=prob,
        )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
