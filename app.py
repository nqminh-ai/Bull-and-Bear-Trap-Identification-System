"""
app.py — Bull & Bear Trap Detector
===================================
Controller: điều phối dữ liệu, gọi UI từ custom_ui.py,
logic từ ml_pipeline.py và analytics.py.

Tabs:
  1. Bieu do ky thuat
  2. Phan tich & Khuyen nghi
  3. Scanner VN30
  4. Backtest
  5. Trade Demo
  6. Model Performance
"""

import os, pickle, warnings, time, json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyodbc
import streamlit as st

import custom_ui as ui
import analytics  as ana
import realtime   as rt

warnings.filterwarnings("ignore")

# ── Trang ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bull & Bear Trap | VN30",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

# ── Config ───────────────────────────────────────────────────
DB_SERVER   = "NGUYEN_MINH"
DB_DATABASE = "ProjectADY_StockDB"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"
MODEL_DIR   = "models"

FEATURE_COLUMNS = [
    "RSI_14","MA20","Volume_Ratio","Upper_Shadow_Ratio",
    "Body_Range_Ratio","Is_Doji","OBV","OBV_MA20",
    "OBV_Divergence","Price_vs_MA20",
    "Lower_Shadow_Ratio","RSI_Weekly",
    "Is_Shooting_Star","Is_Bearish_Engulf","Is_Evening_Star",
]

VN30_SYMBOLS = [
    "ACB","BCM","BID","BVH","CTG","DXG","FPT","GAS","GVR",
    "HDB","HPG","MBB","MSN","MWG","NVL","PDR","PLX","POW",
    "SAB","SHB","SSI","STB","TCB","TPB","VCB","VHM","VIB",
    "VIC","VJC","VNM","VPB",
]


# ── DB ───────────────────────────────────────────────────────
def get_connection():
    conn_str = (
        f"DRIVER={{{DB_DRIVER}}};SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol: str, lookback_days: int = 180) -> pd.DataFrame:
    """Lấy OHLCV từ DB. Returns pd.DataFrame."""
    cutoff = (datetime.now()-timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query  = f"""
        SELECT [time] AS TradingDate,[open] AS OpenPrice,
               [high] AS HighPrice, [low]  AS LowPrice,
               [close] AS ClosePrice,[volume] AS TotalVolume
        FROM dbo.daily_prices
        WHERE [ticker]='{symbol}' AND [time]>='{cutoff}'
        ORDER BY [time] ASC
    """
    conn = get_connection()
    df   = pd.read_sql(query, conn); conn.close()
    df   = df.rename(columns={
        "TradingDate":"Date","OpenPrice":"Open","HighPrice":"High",
        "LowPrice":"Low","ClosePrice":"Close","TotalVolume":"Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


# ── Model loaders ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bundle(symbol: str, trap_type: str = "bull") -> dict:
    """
    Load ensemble bundle {models, metrics}.
    Hỗ trợ cả format cũ (.pkl đơn) và mới (bundle dict).
    Returns dict hoặc None.
    """
    path = os.path.join(MODEL_DIR, f"{symbol}_{trap_type}_trap.pkl")
    # Fallback: format cũ chỉ có bull
    if not os.path.exists(path) and trap_type == "bull":
        path = os.path.join(MODEL_DIR, f"{symbol}_bull_trap.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Format cũ: trực tiếp là model pipeline
    if not isinstance(data, dict):
        return {"models": {"xgb": data}, "metrics": {}}
    return data


# ── Feature Engineering (GIỮ NGUYÊN) ─────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def compute_obv(df):
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toàn bộ features. GIỮ NGUYÊN. Returns pd.DataFrame."""
    r = df.copy()
    r["RSI_14"]             = compute_rsi(r["Close"])
    r["RSI_Weekly"]         = compute_rsi(
        r["Close"].resample("W").last().reindex(r.index, method="ffill"), 14
    )
    r["MA20"]               = r["Close"].rolling(20).mean()
    r["Price_vs_MA20"]      = r["Close"] - r["MA20"]
    r["Volume_MA20"]        = r["Volume"].rolling(20).mean()
    r["Volume_Ratio"]       = r["Volume"] / r["Volume_MA20"].replace(0, np.nan)
    r["Upper_Shadow"]       = r["High"] - r[["Open","Close"]].max(axis=1)
    r["Lower_Shadow"]       = r[["Open","Close"]].min(axis=1) - r["Low"]
    r["Body"]               = (r["Close"]-r["Open"]).abs()
    r["Range"]              = r["High"] - r["Low"]
    r["Upper_Shadow_Ratio"] = r["Upper_Shadow"] / r["Body"].replace(0,1e-6)
    r["Lower_Shadow_Ratio"] = r["Lower_Shadow"] / r["Body"].replace(0,1e-6)
    r["Body_Range_Ratio"]   = r["Body"] / r["Range"].replace(0, np.nan)
    r["Is_Doji"]            = (r["Body_Range_Ratio"] < 0.1).astype(int)
    r["OBV"]                = compute_obv(r)
    r["OBV_MA20"]           = r["OBV"].rolling(20).mean()
    price_up                = r["Close"] > r["MA20"]
    money_out               = r["OBV"]   < r["OBV_MA20"]
    r["OBV_Divergence"]     = (price_up & money_out).astype(int)
    # Patterns
    body   = r["Body"]
    upper  = r["Upper_Shadow"]
    rng    = r["Range"]
    r["Is_Shooting_Star"]  = (
        (upper >= 2*body.replace(0,1e-6)) & (body <= 0.3*rng.replace(0,1e-6)) &
        (r["Close"] > r["MA20"])
    ).astype(int)
    r["Is_Bearish_Engulf"] = (
        (r["Close"] < r["Open"]) & (r["Close"].shift(1) > r["Open"].shift(1)) &
        (r["Open"] >= r["Close"].shift(1)) & (r["Close"] <= r["Open"].shift(1))
    ).astype(int)
    b1 = (r["Close"].shift(2) - r["Open"].shift(2))
    b2 = (r["Close"].shift(1) - r["Open"].shift(1)).abs()
    b3 = r["Open"] - r["Close"]
    rng2 = (r["High"].shift(1) - r["Low"].shift(1)).replace(0, 1e-6)
    r["Is_Evening_Star"] = (
        (b1 > 0) & (b2/rng2 < 0.25) & (b3 > 0) & (b3 > 0.5*b1.abs())
    ).astype(int)
    return r


def predict_bull_trap(df_features: pd.DataFrame, bundle) -> pd.DataFrame:
    """
    Ensemble predict. GIỮ NGUYÊN. Returns pd.DataFrame.
    Xử lý cả format cũ (single model) và mới (bundle).
    """
    result     = df_features.copy()
    feat_cols  = [c for c in FEATURE_COLUMNS if c in result.columns]
    valid_mask = result[feat_cols].notna().all(axis=1)
    X          = result.loc[valid_mask, feat_cols].fillna(0)

    probs = np.zeros(len(result))
    preds = np.zeros(len(result))

    if len(X) > 0 and bundle is not None:
        try:
            models = bundle.get("models", {}) if isinstance(bundle, dict) else {"xgb": bundle}
            if not models:
                return result.assign(Prediction=0, Bull_Trap_Prob=0.0)

            prob_arr = ana._ensemble_predict(models, X) \
                       if hasattr(ana, "_ensemble_predict") \
                       else _local_ensemble_predict(models, X)

            probs[valid_mask] = prob_arr
            preds[valid_mask] = (prob_arr >= 0.35).astype(int)  # Ngưỡng 0.35 tăng recall
        except Exception:
            pass

    result["Prediction"]     = preds
    result["Bull_Trap_Prob"] = probs
    return result


def _local_ensemble_predict(models: dict, X: pd.DataFrame) -> np.ndarray:
    """Weighted average probability. Returns np.ndarray."""
    weights  = {"xgb": 0.4, "rf": 0.3, "lgb": 0.3}
    if "lgb" not in models:
        weights = {"xgb": 0.55, "rf": 0.45}
    total_w  = sum(weights.get(k, 0.33) for k in models)
    prob_sum = np.zeros(len(X))
    for name, model in models.items():
        w = weights.get(name, 0.33)
        try:
            p = model.predict_proba(X)
            prob_sum += w * (p[:,1] if p.shape[1] >= 2 else (model.predict(X)==1).astype(float))
        except Exception:
            prob_sum += w * (model.predict(X)==1).astype(float)
    return prob_sum / total_w if total_w > 0 else prob_sum


# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### Bull & Bear Trap")
        st.markdown("---")
        symbol = st.text_input("Ma co phieu", value="VNM",
                                placeholder="VD: VNM, HPG, FPT...").strip().upper()
        st.markdown("---")
        lookback = st.slider("So ngay lich su", 60, 365, 180, 30)
        st.markdown("---")
        st.markdown("**Tuy chon bieu do**")
        show_fib = st.checkbox("Fibonacci Retracement", True)
        show_sr  = st.checkbox("S/R dong", True)
        show_vpvr= st.checkbox("Volume Profile (VPVR)", True)
        st.markdown("---")
        st.markdown("**Telegram Alert**")
        tg_token    = st.text_input("Bot Token", type="password")
        tg_chat_id  = st.text_input("Chat ID", placeholder="-100xxx")
        tg_threshold= st.slider("Nguong (%)", 30, 80, 50, 5)
        st.markdown("---")
        st.markdown("**Realtime Data**")
        ssi_id     = st.text_input("SSI Consumer ID",     type="password", placeholder="Tuy chon")
        ssi_secret = st.text_input("SSI Consumer Secret", type="password", placeholder="Tuy chon")
        auto_refresh = st.checkbox("Tu dong cap nhat (60s)", value=True)
        st.caption(f"Cap nhat: {datetime.now().strftime('%H:%M %d/%m/%Y')}")
    return symbol, lookback, show_fib, show_sr, show_vpvr, tg_token, tg_chat_id, tg_threshold, ssi_id, ssi_secret, auto_refresh


# ── Main ──────────────────────────────────────────────────────
def main():
    symbol, lookback, show_fib, show_sr, show_vpvr, \
        tg_token, tg_chat_id, tg_threshold, ssi_id, ssi_secret, auto_refresh = render_sidebar()

    if not symbol:
        st.warning("Nhap ma co phieu.")
        return

    # Load data & models
    bull_bundle = load_bundle(symbol, "bull")
    bear_bundle = load_bundle(symbol, "bear")

    if bull_bundle is None:
        st.warning(f"Chua co model Bull cho {symbol}. Chay ml_pipeline.py truoc.")

    with st.spinner(f"Dang tai {symbol}..."):
        try:
            df_raw = fetch_stock_data(symbol, lookback)
        except Exception as e:
            st.error(f"Loi DB: {e}"); return

    if df_raw.empty:
        st.error(f"Khong co du lieu cho {symbol}."); return

    # Features & predict (truoc de co last_close lam fallback cho realtime)
    df_feat = compute_features(df_raw)
    df_bull = predict_bull_trap(df_feat, bull_bundle)
    df_bear = predict_bull_trap(df_feat, bear_bundle)
    df_bull["Bear_Trap_Prob"] = df_bear["Bull_Trap_Prob"]

    latest_bull = df_bull.dropna(subset=["Bull_Trap_Prob"])
    if latest_bull.empty:
        st.error("Khong du du lieu de tinh toan."); return

    latest    = latest_bull.iloc[-1]
    bull_prob = float(latest["Bull_Trap_Prob"])
    bear_prob = float(latest.get("Bear_Trap_Prob", 0))
    rsi       = float(latest.get("RSI_14", 50))
    vol_ratio = float(latest.get("Volume_Ratio", 1))
    divergence= bool(latest.get("OBV_Divergence", 0))
    last_close= float(latest["Close"])  # Dinh nghia truoc khi dung o realtime block
    ma20_val  = float(latest["MA20"]) if not pd.isna(latest["MA20"]) else last_close

    # Realtime price (sau khi da co last_close lam fallback)
    if "ssi_client" not in st.session_state or (ssi_id and ssi_secret):
        st.session_state["ssi_client"] = (
            rt.SSIDataFeed(ssi_id, ssi_secret) if (ssi_id and ssi_secret) else None
        )
    ssi_client = st.session_state.get("ssi_client")

    if auto_refresh and rt.is_market_open():
        updater_key = "rt_updater"
        if updater_key not in st.session_state:
            updater = rt.RealtimeUpdater([symbol], ssi_client)
            updater.start()
            st.session_state[updater_key] = updater
    elif not rt.is_market_open():
        if "rt_updater" in st.session_state:
            st.session_state["rt_updater"].stop()
            del st.session_state["rt_updater"]

    price_data = rt.get_realtime_price(symbol, ssi_client)
    rt_price   = price_data.get("price", last_close) if price_data.get("price", 0) > 0 else last_close
    df_raw_rt  = rt.render_realtime_chart_overlay(df_raw, price_data)

    # MTF confidence
    mtf       = ana.compute_mtf_confidence(df_feat, bull_prob)
    adj_prob  = mtf["adjusted_prob"]

    prev_close = float(df_bull["Close"].iloc[-2]) if len(df_bull) >= 2 else last_close
    change_pct = (last_close - prev_close) / prev_close * 100

    # Market context
    try:
        df_vni   = fetch_stock_data("VNINDEX", lookback)
    except Exception:
        df_vni = None
    context      = ui.analyze_market_context(df_vni)
    market_trend = context["trend"]

    # Recommendation
    rec = ui.generate_recommendation(
        symbol, adj_prob, rsi, vol_ratio, divergence,
        market_trend, last_close, ma20_val
    )

    # Log prediction
    ana.log_prediction(symbol, "bull",
                        datetime.now().strftime("%Y-%m-%d"),
                        last_close, bull_prob)

    # ── Realtime Ticker ──
    rt.render_realtime_ticker(symbol, price_data)
    if auto_refresh and rt.is_market_open():
        st.markdown(
            "<div style='font-size:11px;color:#94a3b8;margin-bottom:8px;'>"
            f"Tu dong lam moi moi {rt.UPDATE_INTERVAL}s trong gio GD. "
            f"Phien hien tai: <strong>{rt.get_session_phase()}</strong></div>",
            unsafe_allow_html=True
        )
        # Auto-rerun trong giờ GD
        time.sleep(0)
        st.rerun() if "last_rt_refresh" not in st.session_state else None
        st.session_state["last_rt_refresh"] = datetime.now()

    # ── Header ──
    ui.render_header(symbol, rt_price, change_pct)
    ui.render_status_banner(adj_prob, symbol, market_trend)

    # MTF badge
    if mtf["adjustment"] != "Giữ nguyên":
        color = "#1e40af" if adj_prob > bull_prob else "#166534"
        st.markdown(f"""
        <div class="banner info" style="font-size:12px;">
            <strong>Multi-Timeframe:</strong> {mtf['adjustment']}
            &nbsp;|&nbsp; Weekly RSI: {mtf['weekly_rsi']}
            &nbsp;|&nbsp; Weekly: {mtf['weekly_trend']}
            &nbsp;|&nbsp; Xac suat dieu chinh: <strong>{adj_prob:.1%}</strong>
        </div>""", unsafe_allow_html=True)

    # Dual prob row
    col_b, col_bear, col_r, col_v = st.columns(4)
    with col_b:
        c = "danger" if adj_prob > 0.5 else ("warning" if adj_prob > 0.3 else "safe")
        st.markdown(f"""<div class="metric-card {c}">
            <div class="metric-label">Bull Trap (adj)</div>
            <div class="metric-value">{adj_prob:.1%}</div>
            <div class="metric-sub">Raw: {bull_prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with col_bear:
        c2 = "warning" if bear_prob > 0.4 else "safe"
        st.markdown(f"""<div class="metric-card {c2}">
            <div class="metric-label">Bear Trap</div>
            <div class="metric-value">{bear_prob:.1%}</div>
            <div class="metric-sub">{"Co the dao chieu!" if bear_prob > 0.4 else "Binh thuong"}</div>
        </div>""", unsafe_allow_html=True)
    with col_r:
        c3 = "danger" if rsi > 70 else ("safe" if rsi < 30 else "neutral")
        st.markdown(f"""<div class="metric-card {c3}">
            <div class="metric-label">RSI(14) / W-RSI</div>
            <div class="metric-value">{rsi:.1f}</div>
            <div class="metric-sub">Weekly: {mtf.get('weekly_rsi','—')}</div>
        </div>""", unsafe_allow_html=True)
    with col_v:
        c4 = "danger" if divergence else "safe"
        st.markdown(f"""<div class="metric-card {c4}">
            <div class="metric-label">OBV Divergence</div>
            <div class="metric-value">{"Co" if divergence else "Khong"}</div>
            <div class="metric-sub">Vol: {vol_ratio:.2f}x</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Bieu do",
        "Phan tich",
        "Scanner VN30",
        "Backtest",
        "Trade Demo",
        "Model Performance",
    ])

    # ── TAB 1: BIỂU ĐỒ ──────────────────────────────────────
    with tab1:
        df_feat_rt = compute_features(df_raw_rt)
        df_bull_rt = predict_bull_trap(df_feat_rt, bull_bundle)
        df_bull_rt['Bear_Trap_Prob'] = predict_bull_trap(df_feat_rt, bear_bundle)['Bull_Trap_Prob']
        fig = ui.build_advanced_chart(df_bull_rt, symbol,
                                      show_fib=show_fib, show_sr=show_sr)
        st.plotly_chart(fig, use_container_width=True)

        # VPVR
        if show_vpvr:
            st.markdown("#### Volume Profile (VPVR)")
            vp     = ana.compute_volume_profile(df_raw, n_bins=30)
            poc_vah= ana.get_poc_and_vah_val(vp)
            ui.render_vpvr(vp, df_raw, poc_vah)

        # Pattern summary
        patterns = []
        if latest.get("Is_Shooting_Star",  0): patterns.append("Shooting Star")
        if latest.get("Is_Bearish_Engulf", 0): patterns.append("Bearish Engulfing")
        if latest.get("Is_Evening_Star",   0): patterns.append("Evening Star")
        if patterns:
            st.markdown(f"""<div class="banner warning">
                Pattern nhan dien hom nay: <strong>{', '.join(patterns)}</strong>
                — Xac nhan tin hieu Bear.
            </div>""", unsafe_allow_html=True)

        # R/R Calculator
        st.markdown("#### Risk / Reward Calculator")
        entry_input = st.number_input(
            "Gia vao lenh (Entry)", value=float(round(last_close)),
            step=100.0, format="%.0f", key="rr_entry"
        )
        rr = ana.calculate_risk_reward(df_feat, entry_input)
        ui.render_risk_reward(rr)

        rt.render_price_alert_config(symbol, rt_price, adj_prob)

        with st.expander("Du lieu chi tiet (20 phien)"):
            show_cols = [c for c in [
                "Open","High","Low","Close","Volume","RSI_14","MA20",
                "Volume_Ratio","OBV_Divergence",
                "Is_Shooting_Star","Is_Bearish_Engulf","Is_Evening_Star",
                "Bull_Trap_Prob","Bear_Trap_Prob","Prediction",
            ] if c in df_bull.columns]
            st.dataframe(
                df_bull[show_cols].tail(20).style.format({
                    c: "{:.2f}" for c in ["Open","High","Low","Close","MA20"]
                } | {"RSI_14":"{:.1f}","Volume_Ratio":"{:.2f}",
                     "Bull_Trap_Prob":"{:.1%}","Bear_Trap_Prob":"{:.1%}"}
                ).background_gradient(subset=["Bull_Trap_Prob"], cmap="RdYlGn_r"),
                use_container_width=True,
            )

    # ── TAB 2: PHÂN TÍCH ──────────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        with c1: ui.render_market_context(context)
        with c2: ui.render_recommendation(rec)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Foreign Flow
        st.markdown("#### Foreign Flow (Dong tien khoi ngoai)")
        df_ff = ana.fetch_foreign_flow(symbol, lookback)
        ui.render_foreign_flow(df_ff, symbol)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        ui.render_telegram_section(symbol, adj_prob, rec["action"], market_trend)

        if tg_token and tg_chat_id and adj_prob * 100 >= tg_threshold:
            if ui.send_telegram_alert(tg_token, tg_chat_id, symbol,
                                       adj_prob, rec["action"], market_trend):
                st.toast(f"Telegram alert: {symbol}!")

    # ── TAB 3: SCANNER VN30 ───────────────────────────────────
    with tab3:
        st.markdown("#### Multi-Stock Scanner — VN30")
        st.caption("Quet toan bo VN30, xep hang theo xac suat Bull Trap giam dan.")

        # Bảng giá realtime tất cả VN30
        st.markdown("#### Bang gia Realtime VN30")
        if st.button("Lay gia Realtime", key="btn_rt_scan"):
            with st.spinner("Dang lay gia..."):
                all_prices = rt.get_batch_realtime(VN30_SYMBOLS, ssi_client)
                st.session_state["rt_prices"] = all_prices
        if "rt_prices" in st.session_state and st.session_state["rt_prices"]:
            df_rt_table = rt.render_realtime_scanner_row(st.session_state["rt_prices"])
            def color_pct(val):
                try:
                    v = float(val.replace("%","").replace("+",""))
                    return "color:#16a34a;font-weight:600;" if v > 0 else ("color:#dc2626;font-weight:600;" if v < 0 else "")
                except: return ""
            st.dataframe(df_rt_table.style.applymap(color_pct, subset=["%"]),
                         use_container_width=True, hide_index=True, height=400)
        st.markdown("---")
        st.markdown("#### AI Risk Scanner")
        if st.button("Quet AI ngay", type="primary", key="btn_scan"):
            with st.spinner("Dang quet 30+ ma..."):
                df_scan = ana.scan_vn30(
                    symbols=VN30_SYMBOLS,
                    model_loader_fn=load_bundle,
                    feature_fn=compute_features,
                    predict_fn=predict_bull_trap,
                    lookback_days=60,
                )
            st.session_state["scan_result"] = df_scan

        if "scan_result" in st.session_state:
            ui.render_scanner_table(st.session_state["scan_result"])

    # ── TAB 4: BACKTEST ───────────────────────────────────────
    with tab4:
        st.markdown("#### So sanh: AI Strategy vs Buy & Hold")
        kpis   = ui.compute_backtest_kpis(df_bull)
        ui.render_backtest_kpis(kpis)
        fig_bt = ui.build_backtest_chart(df_bull)
        st.plotly_chart(fig_bt, use_container_width=True)

    # ── TAB 5: TRADE DEMO ─────────────────────────────────────
    with tab5:
        ui.render_trade_demo(symbol=symbol, current_price=last_close, prob=adj_prob)

    # ── TAB 6: MODEL PERFORMANCE ──────────────────────────────
    with tab6:
        st.markdown("#### Model Performance Tracker")
        # Verify predictions cũ
        updated = ana.verify_predictions(forward_days=5)
        if updated > 0:
            st.success(f"Da verify {updated} du doan moi.")

        perf = ana.compute_rolling_performance(window=20)
        ui.render_model_performance(perf, symbol)

        # Model metadata
        meta_path = os.path.join(MODEL_DIR, f"{symbol}_bull_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            st.markdown("**Thong tin model hien tai:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("AUC (train)", f"{meta.get('auc_roc',0):.3f}")
            col2.metric("F1 (trap)",   f"{meta.get('f1_trap',0):.3f}")
            col3.metric("Trained at",  meta.get("trained_at","N/A")[:10])


import json  # Đảm bảo import cho meta_path section
if __name__ == "__main__":
    main()
