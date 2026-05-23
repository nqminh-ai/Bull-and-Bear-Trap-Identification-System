"""
analytics.py
============
Module phân tích nâng cao — tách biệt hoàn toàn khỏi UI:
  1. Model Performance Tracker
  2. Multi-Stock Scanner (VN30)
  3. Multi-Timeframe Confirmation
  4. Volume Profile (VPVR)
  5. Foreign Flow Analysis
  6. Risk/Reward Calculator
  7. Pattern Recognition (đã tích hợp vào ml_pipeline)

NGUYÊN TẮC: Không dùng print(). Mọi hàm phải return.
"""

import os
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyodbc

warnings.filterwarnings("ignore")

# ── Cấu hình ─────────────────────────────────────────────────
DB_SERVER   = "NGUYEN_MINH"
DB_DATABASE = "ProjectADY_StockDB"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"
PERF_DIR    = "performance"
os.makedirs(PERF_DIR, exist_ok=True)


def get_connection():
    """Kết nối pyodbc. Returns pyodbc.Connection."""
    conn_str = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


def fetch_ohlcv(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Lấy OHLCV từ DB. Returns pd.DataFrame OHLCV với index Date.
    """
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query  = f"""
        SELECT [time] AS TradingDate, [open] AS OpenPrice,
               [high] AS HighPrice,  [low]  AS LowPrice,
               [close] AS ClosePrice,[volume] AS TotalVolume
        FROM dbo.daily_prices
        WHERE [ticker] = '{symbol}' AND [time] >= '{cutoff}'
        ORDER BY [time] ASC
    """
    conn = get_connection()
    df   = pd.read_sql(query, conn)
    conn.close()
    df = df.rename(columns={
        "TradingDate":"Date","OpenPrice":"Open","HighPrice":"High",
        "LowPrice":"Low","ClosePrice":"Close","TotalVolume":"Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


# ============================================================
# 1. MODEL PERFORMANCE TRACKER
# ============================================================
PRED_LOG_FILE = os.path.join(PERF_DIR, "prediction_log.json")


def log_prediction(symbol: str, trap_type: str, date: str,
                    price: float, prob: float):
    """
    Ghi lại dự đoán để so sánh với thực tế sau 5 phiên.

    Args:
        symbol: Mã cổ phiếu.
        trap_type: 'bull' hoặc 'bear'.
        date: Ngày dự đoán (YYYY-MM-DD).
        price: Giá tại thời điểm dự đoán.
        prob: Xác suất trap.
    """
    entry = {
        "symbol":    symbol,
        "trap_type": trap_type,
        "date":      date,
        "price":     price,
        "prob":      prob,
        "predicted": 1 if prob >= 0.5 else 0,
        "actual":    None,   # Điền sau 5 phiên
        "verified":  False,
        "logged_at": datetime.now().isoformat(),
    }
    log = _load_pred_log()
    log.append(entry)
    _save_pred_log(log)


def _load_pred_log() -> list:
    """Load prediction log từ file JSON."""
    if not os.path.exists(PRED_LOG_FILE):
        return []
    try:
        with open(PRED_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_pred_log(log: list):
    """Lưu prediction log ra file JSON."""
    with open(PRED_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def verify_predictions(forward_days: int = 5) -> int:
    """
    So sánh dự đoán cũ với thực tế (giá T+5).
    Cập nhật cột 'actual' và 'verified' trong log.

    Returns:
        int: Số lượng dự đoán vừa được verify.
    """
    log     = _load_pred_log()
    updated = 0

    for entry in log:
        if entry.get("verified"):
            continue

        pred_date = datetime.strptime(entry["date"], "%Y-%m-%d")
        target_date = pred_date + timedelta(days=forward_days + 2)  # +2 buffer weekends

        if datetime.now() < target_date:
            continue  # Chưa đến ngày verify

        try:
            df = fetch_ohlcv(entry["symbol"], lookback_days=30)
            future_rows = df[df.index > pred_date]

            if len(future_rows) < forward_days:
                continue

            future_close = future_rows.iloc[forward_days - 1]["Close"]
            pred_price   = entry["price"]

            if entry["trap_type"] == "bull":
                # Bull Trap thật: giá T+5 < giá dự đoán
                actual = 1 if future_close < pred_price else 0
            else:
                # Bear Trap thật: giá T+5 > giá dự đoán
                actual = 1 if future_close > pred_price else 0

            entry["actual"]   = actual
            entry["verified"] = True
            updated += 1

        except Exception:
            continue

    if updated > 0:
        _save_pred_log(log)

    return updated


def compute_rolling_performance(window: int = 20) -> dict:
    """
    Tính rolling accuracy, precision, recall từ log đã verified.

    Args:
        window: Số dự đoán gần nhất để tính rolling metrics.

    Returns:
        dict: {accuracy, precision, recall, f1, n_verified, recent_log}
    """
    log      = _load_pred_log()
    verified = [e for e in log if e.get("verified") and e.get("actual") is not None]

    if not verified:
        return {"accuracy": None, "precision": None, "recall": None,
                "f1": None, "n_verified": 0, "recent_log": []}

    recent = verified[-window:]
    y_true = [e["actual"]    for e in recent]
    y_pred = [e["predicted"] for e in recent]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    accuracy  = (tp + tn) / len(recent) if recent else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy":   round(accuracy,  4),
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "n_verified": len(verified),
        "recent_log": recent,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ============================================================
# 2. MULTI-STOCK SCANNER
# ============================================================
def scan_vn30(symbols: list, model_loader_fn, feature_fn,
              predict_fn, lookback_days: int = 90) -> pd.DataFrame:
    """
    Quét toàn bộ danh sách mã, trả về bảng xếp hạng rủi ro.

    Args:
        symbols: Danh sách mã cần quét.
        model_loader_fn: Hàm load model (symbol, trap_type) -> bundle.
        feature_fn: Hàm tính features (df) -> df_features.
        predict_fn: Hàm dự đoán (df_features, bundle) -> df_predicted.
        lookback_days: Ngày lịch sử.

    Returns:
        pd.DataFrame bảng xếp hạng sort theo Bull_Trap_Prob giảm dần.
    """
    rows = []

    for symbol in symbols:
        try:
            df_raw  = fetch_ohlcv(symbol, lookback_days)
            if len(df_raw) < 25:
                continue

            df_feat = feature_fn(df_raw)
            bundle  = model_loader_fn(symbol, "bull")
            df_pred = predict_fn(df_feat, bundle)

            valid = df_pred.dropna(subset=["Bull_Trap_Prob"]).tail(1)
            if valid.empty:
                continue

            latest   = valid.iloc[-1]
            bear_bnd = model_loader_fn(symbol, "bear")
            bear_df  = predict_fn(df_feat, bear_bnd)
            bear_prob = float(bear_df["Bull_Trap_Prob"].dropna().iloc[-1]) \
                        if not bear_df["Bull_Trap_Prob"].dropna().empty else 0.0

            rows.append({
                "Symbol":        symbol,
                "Close":         round(float(latest["Close"]), 2),
                "Bull_Trap_%":   round(float(latest["Bull_Trap_Prob"]) * 100, 1),
                "Bear_Trap_%":   round(bear_prob * 100, 1),
                "RSI":           round(float(latest.get("RSI_14", 0)), 1),
                "Volume_Ratio":  round(float(latest.get("Volume_Ratio", 1)), 2),
                "OBV_Div":       bool(latest.get("OBV_Divergence", 0)),
                "Pattern":       _summarize_patterns(latest),
                "Signal":        _risk_label(float(latest["Bull_Trap_Prob"]), bear_prob),
            })

        except Exception:
            continue

    df_scan = pd.DataFrame(rows)
    if not df_scan.empty:
        df_scan = df_scan.sort_values("Bull_Trap_%", ascending=False).reset_index(drop=True)
    return df_scan


def _summarize_patterns(row) -> str:
    """Tóm tắt pattern nến thành string. Returns str."""
    patterns = []
    if row.get("Is_Shooting_Star",   0): patterns.append("Shooting Star")
    if row.get("Is_Bearish_Engulf",  0): patterns.append("Bear Engulf")
    if row.get("Is_Evening_Star",    0): patterns.append("Evening Star")
    return ", ".join(patterns) if patterns else "—"


def _risk_label(bull_prob: float, bear_prob: float) -> str:
    """Phân loại tín hiệu tổng hợp. Returns str."""
    if bull_prob > 0.65:   return "SELL / AVOID"
    if bull_prob > 0.5:    return "CAUTION"
    if bear_prob > 0.5:    return "REVERSAL?"
    if bull_prob < 0.3:    return "BUY SIGNAL"
    return "NEUTRAL"


# ============================================================
# 3. MULTI-TIMEFRAME CONFIRMATION
# ============================================================
def compute_mtf_confidence(df_daily: pd.DataFrame,
                            bull_prob: float) -> dict:
    """
    Điều chỉnh confidence Bull Trap dựa trên Weekly timeframe.

    Logic:
    - Weekly RSI > 70 + Bull Trap daily → tăng confidence (x1.15)
    - Weekly RSI < 50 + Uptrend weekly → giảm confidence (x0.85)
    - Ngược lại: giữ nguyên

    Args:
        df_daily: DataFrame daily đã có features.
        bull_prob: Xác suất Bull Trap daily.

    Returns:
        dict: {adjusted_prob, weekly_rsi, weekly_trend, adjustment}
    """
    # Weekly OHLCV
    weekly = df_daily["Close"].resample("W").last().dropna()
    if len(weekly) < 15:
        return {"adjusted_prob": bull_prob, "weekly_rsi": None,
                "weekly_trend": "Unknown", "adjustment": "None"}

    # Weekly RSI
    delta  = weekly.diff()
    gain   = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    w_rsi  = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])

    # Weekly MA20
    w_ma20 = float(weekly.rolling(20).mean().iloc[-1]) if len(weekly) >= 20 else float(weekly.mean())
    w_last = float(weekly.iloc[-1])
    w_trend = "Uptrend" if w_last > w_ma20 else "Downtrend"

    # Điều chỉnh
    adj_prob   = bull_prob
    adjustment = "Giữ nguyên"

    if w_rsi > 70 and bull_prob > 0.4:
        adj_prob   = min(0.98, bull_prob * 1.15)
        adjustment = f"Tăng +15% (Weekly RSI={w_rsi:.0f} quá mua)"
    elif w_rsi < 50 and w_trend == "Uptrend" and bull_prob < 0.6:
        adj_prob   = max(0.02, bull_prob * 0.85)
        adjustment = f"Giảm -15% (Weekly Uptrend mạnh, RSI={w_rsi:.0f})"
    elif w_rsi > 65:
        adj_prob   = min(0.95, bull_prob * 1.08)
        adjustment = f"Tăng nhẹ +8% (Weekly RSI={w_rsi:.0f})"

    return {
        "adjusted_prob": round(adj_prob, 4),
        "weekly_rsi":    round(w_rsi, 1),
        "weekly_trend":  w_trend,
        "adjustment":    adjustment,
        "weekly_ma20":   round(w_ma20, 2),
    }


# ============================================================
# 4. VOLUME PROFILE (VPVR)
# ============================================================
def compute_volume_profile(df: pd.DataFrame,
                            n_bins: int = 30) -> pd.DataFrame:
    """
    Tính Volume Profile (VPVR) — phân bổ volume theo dải giá.

    Args:
        df: DataFrame OHLCV.
        n_bins: Số dải giá.

    Returns:
        pd.DataFrame với cột [price_level, volume, pct, is_hvn, is_lvn]
        HVN = High Volume Node, LVN = Low Volume Node.
    """
    price_min = df["Low"].min()
    price_max = df["High"].max()
    bins      = np.linspace(price_min, price_max, n_bins + 1)
    levels    = (bins[:-1] + bins[1:]) / 2

    vol_by_level = np.zeros(n_bins)

    for _, row in df.iterrows():
        # Phân bổ volume theo typical price trong range [Low, High]
        row_bins = np.digitize([row["Low"], row["High"]], bins)
        lo_bin   = max(0, row_bins[0] - 1)
        hi_bin   = min(n_bins - 1, row_bins[1] - 1)
        span     = hi_bin - lo_bin + 1
        vol_by_level[lo_bin:hi_bin + 1] += row["Volume"] / span

    total  = vol_by_level.sum()
    pct    = vol_by_level / total * 100 if total > 0 else vol_by_level
    mean_v = vol_by_level.mean()
    std_v  = vol_by_level.std()

    result = pd.DataFrame({
        "price_level": levels,
        "volume":      vol_by_level,
        "pct":         pct,
        "is_hvn":      vol_by_level > (mean_v + 0.5 * std_v),  # High Volume Node
        "is_lvn":      vol_by_level < (mean_v - 0.5 * std_v),  # Low Volume Node
    })
    return result.sort_values("price_level")


def get_poc_and_vah_val(vp: pd.DataFrame) -> dict:
    """
    Tính POC (Point of Control), VAH, VAL từ Volume Profile.

    Returns:
        dict: {poc, vah, val} — các mức giá quan trọng.
    """
    poc   = float(vp.loc[vp["volume"].idxmax(), "price_level"])
    total = vp["volume"].sum()

    # Value Area = 70% tổng volume quanh POC
    sorted_vp = vp.sort_values("volume", ascending=False)
    cumvol    = 0.0
    va_levels = []
    for _, row in sorted_vp.iterrows():
        if cumvol >= 0.70 * total:
            break
        va_levels.append(row["price_level"])
        cumvol += row["volume"]

    vah = max(va_levels) if va_levels else poc
    val = min(va_levels) if va_levels else poc

    return {"poc": round(poc, 2), "vah": round(vah, 2), "val": round(val, 2)}


# ============================================================
# 5. FOREIGN FLOW ANALYSIS
# ============================================================
def fetch_foreign_flow(symbol: str, lookback_days: int = 90) -> pd.DataFrame:
    """
    Lấy dữ liệu dòng tiền khối ngoại từ DB (nếu có bảng foreign_flow).
    Trả về DataFrame rỗng nếu bảng không tồn tại.

    Returns:
        pd.DataFrame với cột [Date, ForeignBuy, ForeignSell, NetForeign].
    """
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Kiểm tra bảng có tồn tại không
    check_query = """
        SELECT COUNT(*) as cnt
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME IN ('foreign_flow', 'ForeignFlow', 'foreign_trading')
    """
    try:
        conn  = get_connection()
        check = pd.read_sql(check_query, conn)
        if check["cnt"].iloc[0] == 0:
            conn.close()
            return pd.DataFrame()

        # Thử các tên bảng phổ biến
        for table in ["foreign_flow", "ForeignFlow", "foreign_trading"]:
            try:
                query = f"""
                    SELECT [time] AS Date,
                           foreign_buy  AS ForeignBuy,
                           foreign_sell AS ForeignSell,
                           (foreign_buy - foreign_sell) AS NetForeign
                    FROM dbo.{table}
                    WHERE ticker = '{symbol}' AND [time] >= '{cutoff}'
                    ORDER BY [time] ASC
                """
                df = pd.read_sql(query, conn)
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    conn.close()
                    return df.set_index("Date").sort_index()
            except Exception:
                continue

        conn.close()
        return pd.DataFrame()

    except Exception:
        return pd.DataFrame()


# ============================================================
# 6. RISK/REWARD CALCULATOR
# ============================================================
def calculate_risk_reward(df: pd.DataFrame, entry_price: float,
                           risk_pct: float = 2.0) -> dict:
    """
    Tính Stop-Loss, Take-Profit và tỷ lệ R:R tự động.

    Stop-Loss: dưới MA20 hoặc Support gần nhất (lấy min).
    Take-Profit: Resistance gần nhất phía trên entry.

    Args:
        df: DataFrame đã có features (cần MA20, High, Low).
        entry_price: Giá vào lệnh.
        risk_pct: % rủi ro tối đa (mặc định 2%).

    Returns:
        dict: {stop_loss, take_profit, rr_ratio, max_loss, max_gain,
               position_size (với vốn 100 tỷ), sl_source, tp_source}
    """
    ma20 = float(df["MA20"].dropna().iloc[-1]) if "MA20" in df.columns else entry_price

    # Tìm Support & Resistance
    window = max(5, len(df) // 20)
    highs  = df["High"].values
    lows   = df["Low"].values

    sup_levels = []
    res_levels = []
    for i in range(window, len(df) - window):
        if lows[i]  == min(lows[i  - window: i + window]): sup_levels.append(lows[i])
        if highs[i] == max(highs[i - window: i + window]): res_levels.append(highs[i])

    # Support gần nhất phía dưới entry
    sups_below = [s for s in sup_levels if s < entry_price]
    sl_support = max(sups_below) if sups_below else entry_price * 0.97
    sl_ma20    = ma20 * 0.995  # 0.5% dưới MA20

    stop_loss  = min(sl_support, sl_ma20)
    sl_source  = "MA20" if sl_ma20 <= sl_support else "Support"

    # Resistance gần nhất phía trên entry
    res_above  = [r for r in res_levels if r > entry_price]
    take_profit = min(res_above) if res_above else entry_price * 1.06
    tp_source   = "Resistance" if res_above else "Default +6%"

    # Metrics
    risk_per_share   = entry_price - stop_loss
    reward_per_share = take_profit  - entry_price
    rr_ratio         = reward_per_share / risk_per_share if risk_per_share > 0 else 0

    # Position sizing (1% vốn mỗi giao dịch = 1 tỷ)
    capital_per_trade = 1_000_000_000   # 1 tỷ/lệnh
    max_loss_amount   = capital_per_trade * (risk_pct / 100)
    position_size     = int(max_loss_amount / risk_per_share / 100) * 100 \
                        if risk_per_share > 0 else 0

    return {
        "entry_price":    round(entry_price, 2),
        "stop_loss":      round(stop_loss,   2),
        "take_profit":    round(take_profit,  2),
        "sl_source":      sl_source,
        "tp_source":      tp_source,
        "risk_per_share": round(risk_per_share,   2),
        "reward_per_share": round(reward_per_share, 2),
        "rr_ratio":       round(rr_ratio,     2),
        "sl_pct":         round(risk_per_share   / entry_price * 100, 2),
        "tp_pct":         round(reward_per_share / entry_price * 100, 2),
        "position_size":  position_size,
        "est_max_loss":   round(position_size * risk_per_share,   0),
        "est_max_gain":   round(position_size * reward_per_share, 0),
    }
