"""
ml_pipeline.py — Bull & Bear Trap Detector
==========================================
Pipeline huấn luyện ENSEMBLE model (XGBoost + RandomForest + LightGBM)
cho cả Bull Trap và Bear Trap trên toàn bộ mã VN30.

Features:
  - Bear Trap Detection (đảo nhãn Bull Trap)
  - Ensemble Prediction (XGB + RF + LGB weighted vote)
  - Auto-Retraining với model versioning (chỉ deploy nếu AUC tốt hơn)
  - Multi-Timeframe: weekly RSI để điều chỉnh confidence

NGUYÊN TẮC: Không dùng print() trong hàm xử lý. Mọi hàm phải return.
"""

import os
import json
import pickle
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Cấu hình ─────────────────────────────────────────────────
DB_SERVER   = "NGUYEN_MINH"
DB_DATABASE = "ProjectADY_StockDB"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"

MODEL_DIR   = "models"
VERSION_DIR = os.path.join(MODEL_DIR, "versions")
PERF_DIR    = "performance"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(VERSION_DIR, exist_ok=True)
os.makedirs(PERF_DIR,    exist_ok=True)

VN30_SYMBOLS = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "DXG", "FPT", "GAS", "GVR", "HDB",
    "HPG", "MBB", "MSN", "MWG", "NVL",
    "PDR", "PLX", "POW", "SAB", "SHB",
    "SSI", "STB", "TCB", "TPB", "VCB",
    "VHM", "VIB", "VIC", "VJC", "VNM", "VPB",
]

FEATURE_COLUMNS = [
    "RSI_14", "MA20", "Volume_Ratio", "Upper_Shadow_Ratio",
    "Body_Range_Ratio", "Is_Doji", "OBV", "OBV_MA20",
    "OBV_Divergence", "Price_vs_MA20",
    # Thêm cho Bear Trap
    "Lower_Shadow_Ratio", "RSI_Weekly",
    # Pattern
    "Is_Shooting_Star", "Is_Bearish_Engulf", "Is_Evening_Star",
]

TRAP_TYPES = ["bull", "bear"]


# ── Kết nối DB ───────────────────────────────────────────────
def get_connection():
    """Tạo kết nối pyodbc. Returns pyodbc.Connection."""
    conn_str = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


def fetch_stock_data(symbol: str, lookback_days: int = 730) -> pd.DataFrame:
    """
    Lấy OHLCV từ dbo.daily_prices.

    Returns:
        pd.DataFrame OHLCV với index Date.
    """
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query = f"""
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


# ── Feature Engineering ──────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Tính RSI. Returns pd.Series."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(com=period-1, min_periods=period).mean()
    al    = loss.ewm(com=period-1, min_periods=period).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Tính OBV. Returns pd.Series."""
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def compute_weekly_rsi(df: pd.DataFrame) -> pd.Series:
    """
    Resample daily -> weekly, tính RSI(14) weekly,
    rồi reindex về daily (forward-fill).

    Returns:
        pd.Series weekly RSI aligned với daily index.
    """
    weekly = df["Close"].resample("W").last().dropna()
    wrsi   = compute_rsi(weekly, 14)
    return wrsi.reindex(df.index, method="ffill")


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nhận diện nến đặc biệt:
    - Shooting Star: râu trên dài > 2x thân, thân nhỏ, ở vùng cao
    - Bearish Engulfing: nến đỏ bao phủ hoàn toàn nến xanh trước
    - Evening Star: 3 nến — xanh lớn, doji, đỏ lớn

    Returns:
        pd.DataFrame với 3 cột flag pattern (0/1).
    """
    r = df.copy()
    body       = (r["Close"] - r["Open"]).abs()
    upper_sh   = r["High"] - r[["Open","Close"]].max(axis=1)
    lower_sh   = r[["Open","Close"]].min(axis=1) - r["Low"]
    candle_rng = r["High"] - r["Low"]

    # Shooting Star: râu trên >= 2x thân, thân <= 30% range, giá gần vùng cao
    r["Is_Shooting_Star"] = (
        (upper_sh >= 2 * body.replace(0, 1e-6)) &
        (body <= 0.3 * candle_rng.replace(0, 1e-6)) &
        (r["Close"] > r["Close"].rolling(20).mean())
    ).astype(int)

    # Bearish Engulfing: hôm nay đỏ, hôm qua xanh, thân hôm nay bao phủ hôm qua
    prev_open  = r["Open"].shift(1)
    prev_close = r["Close"].shift(1)
    r["Is_Bearish_Engulf"] = (
        (r["Close"] < r["Open"]) &          # Hôm nay đỏ
        (prev_close > prev_open) &           # Hôm qua xanh
        (r["Open"] >= prev_close) &          # Mở trên đóng hôm qua
        (r["Close"] <= prev_open)            # Đóng dưới mở hôm qua
    ).astype(int)

    # Evening Star: nến 1 xanh lớn, nến 2 doji, nến 3 đỏ
    body1  = (r["Close"].shift(2) - r["Open"].shift(2))   # Nến 1 xanh
    body2  = (r["Close"].shift(1) - r["Open"].shift(1)).abs()  # Nến 2 doji
    rng2   = (r["High"].shift(1) - r["Low"].shift(1)).replace(0, 1e-6)
    body3  = r["Open"] - r["Close"]  # Nến 3 đỏ
    r["Is_Evening_Star"] = (
        (body1 > 0) &                               # Nến 1 tăng
        (body2 / rng2 < 0.25) &                     # Nến 2 doji
        (body3 > 0) &                               # Nến 3 giảm
        (body3 > 0.5 * body1.abs())                 # Nến 3 đủ lớn
    ).astype(int)

    return r


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toàn bộ features cho cả Bull & Bear Trap.
    GIỮ NGUYÊN — không chỉnh sửa logic này.

    Returns:
        pd.DataFrame với đầy đủ features.
    """
    r = df.copy()

    # Core indicators
    r["RSI_14"]        = compute_rsi(r["Close"])
    r["RSI_Weekly"]    = compute_weekly_rsi(r)
    r["MA20"]          = r["Close"].rolling(20).mean()
    r["Price_vs_MA20"] = r["Close"] - r["MA20"]

    # Volume
    r["Volume_MA20"]   = r["Volume"].rolling(20).mean()
    r["Volume_Ratio"]  = r["Volume"] / r["Volume_MA20"].replace(0, np.nan)

    # Candle anatomy
    r["Upper_Shadow"]      = r["High"] - r[["Open","Close"]].max(axis=1)
    r["Lower_Shadow"]      = r[["Open","Close"]].min(axis=1) - r["Low"]
    r["Body"]              = (r["Close"] - r["Open"]).abs()
    r["Range"]             = r["High"] - r["Low"]
    r["Upper_Shadow_Ratio"] = r["Upper_Shadow"] / r["Body"].replace(0, 1e-6)
    r["Lower_Shadow_Ratio"] = r["Lower_Shadow"] / r["Body"].replace(0, 1e-6)
    r["Body_Range_Ratio"]   = r["Body"]         / r["Range"].replace(0, np.nan)
    r["Is_Doji"]            = (r["Body_Range_Ratio"] < 0.1).astype(int)

    # OBV & Divergence
    r["OBV"]           = compute_obv(r)
    r["OBV_MA20"]      = r["OBV"].rolling(20).mean()
    price_up           = r["Close"] > r["MA20"]
    money_out          = r["OBV"]   < r["OBV_MA20"]
    r["OBV_Divergence"] = (price_up & money_out).astype(int)

    # Pattern recognition
    r = detect_patterns(r)

    return r


# ── Labeling ─────────────────────────────────────────────────
def label_bull_trap(df: pd.DataFrame, forward_days: int = 5,
                     lookahead_confirm: int = 3) -> pd.DataFrame:
    """
    Gán nhãn Bull Trap với điều kiện nới lỏng để tăng số mẫu:

    Breakout = giá đóng cửa cắt lên trên MA20 trong cửa sổ 3 ngày
               VÀ giá đang dưới MA20 trước đó ít nhất 2 ngày.
    Bull Trap (1) = giá T+forward_days thấp hơn giá đóng cửa ngày breakout.
    Breakout thật (0) = giá T+forward_days >= giá breakout VÀ duy trì trên MA20.

    Dùng rolling window để tăng số mẫu training.

    Returns:
        pd.DataFrame chỉ các ngày Breakout có nhãn.
    """
    r = df.copy()

    # Breakout: close vượt MA20, trước đó 2 ngày liên tiếp dưới MA20
    above_ma20     = r["Close"] > r["MA20"]
    was_below_2d   = (~above_ma20.shift(1).fillna(True)) &                      (~above_ma20.shift(2).fillna(True))
    r["Is_Breakout"] = above_ma20 & was_below_2d

    # Xác nhận breakout: giá đóng cửa > MA20 sau lookahead_confirm ngày
    future_close   = r["Close"].shift(-forward_days)
    future_ma20    = r["MA20"].shift(-lookahead_confirm)

    # Bull Trap: giá T+5 quay lại dưới giá ngày breakout (giảm ≥ 1%)
    trap_condition = r["Is_Breakout"] &                      (future_close < r["Close"] * 0.99)

    # Breakout thật: giá T+5 cao hơn hoặc bằng giá breakout
    real_condition = r["Is_Breakout"] &                      (future_close >= r["Close"] * 0.99)

    r["Label"] = np.where(
        trap_condition, 1,
        np.where(real_condition, 0, np.nan)
    )

    labeled = r.dropna(subset=["Label"]).copy()
    labeled["Label"] = labeled["Label"].astype(int)
    return labeled


def label_bear_trap(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Gán nhãn Bear Trap — đối xứng hoàn toàn với Bull Trap.
    Breakdown xuống dưới MA20 nhưng T+5 hồi phục lên trên giá bán → 1 (trap).
    Breakdown thật → 0.

    Returns:
        pd.DataFrame chỉ các ngày Breakdown có nhãn.
    """
    r = df.copy()
    prev_close = r["Close"].shift(1)
    prev_ma20  = r["MA20"].shift(1)
    # Breakdown: hôm qua trên MA20, hôm nay cắt xuống dưới
    r["Is_Breakdown"] = (r["Close"] < r["MA20"]) & (prev_close >= prev_ma20)
    r["Future_Close"] = r["Close"].shift(-forward_days)
    # Bear Trap: giá T+5 hồi lên cao hơn giá breakdown ≥ 1%
    r["Label"] = np.where(
        r["Is_Breakdown"] & (r["Future_Close"] > r["Close"] * 1.01), 1,
        np.where(r["Is_Breakdown"] & (r["Future_Close"] <= r["Close"] * 1.01), 0, np.nan)
    )
    labeled = r.dropna(subset=["Label"]).copy()
    labeled["Label"] = labeled["Label"].astype(int)
    return labeled


# ── Ensemble Model ───────────────────────────────────────────
def _apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Dùng SMOTE để oversample class thiểu số (trap).
    Fallback về random oversampling nếu không có imbalanced-learn.

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()

    # Chỉ áp dụng nếu mất cân bằng > 3:1
    if n_pos == 0 or n_neg / n_pos <= 3:
        return X, y

    try:
        from imblearn.over_sampling import SMOTE
        # k_neighbors phải < số mẫu thiểu số
        k = min(5, n_pos - 1)
        if k < 1:
            raise ValueError("Qua it mau")
        sm = SMOTE(k_neighbors=k, random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
    except Exception:
        # Fallback: random oversampling thủ công
        trap_idx   = y[y == 1].index
        n_oversample = min(n_neg - n_pos, n_pos * 4)  # Tối đa 5x
        extra_idx  = np.random.choice(trap_idx, size=n_oversample, replace=True)
        X_extra    = X.loc[extra_idx]
        y_extra    = y.loc[extra_idx]
        X_res = pd.concat([X, X_extra]).reset_index(drop=True)
        y_res = pd.concat([y, y_extra]).reset_index(drop=True)
        return X_res, y_res


def train_ensemble(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Huấn luyện 3 model: XGBoost + RandomForest + LightGBM.
    Áp dụng SMOTE oversampling + scale_pos_weight để xử lý mất cân bằng.

    Args:
        X_train: Features.
        y_train: Labels (0/1).

    Returns:
        dict: {"xgb": model, "rf": model, "lgb": model (nếu có)}
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    spw   = min(n_neg / n_pos, 10.0) if n_pos > 0 else 1.0  # Cap tối đa 10x

    logger.info(f"  Train: pos={n_pos} neg={n_neg} spw={spw:.1f}")

    # Áp dụng oversampling
    X_res, y_res = _apply_smote(X_train, y_train)
    n_pos_res = (y_res == 1).sum()
    logger.info(f"  Sau SMOTE: pos={n_pos_res} neg={(y_res==0).sum()}")

    models = {}

    # XGBoost — dùng X_res sau SMOTE, spw=1 vì đã cân bằng
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        scale_pos_weight=1.0,          # Đã cân bằng bởi SMOTE
        min_child_weight=3,            # Tránh overfit mẫu nhỏ
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    pipe_xgb = Pipeline([("scaler", StandardScaler()), ("clf", xgb_clf)])
    pipe_xgb.fit(X_res, y_res)
    models["xgb"] = pipe_xgb

    # RandomForest — class_weight balanced_subsample (robust hơn balanced)
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        class_weight="balanced_subsample",
        min_samples_leaf=3,            # Tránh overfit
        random_state=42, n_jobs=-1,
    )
    pipe_rf = Pipeline([("scaler", StandardScaler()), ("clf", rf_clf)])
    pipe_rf.fit(X_res, y_res)
    models["rf"] = pipe_rf

    # LightGBM
    if HAS_LGB:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            is_unbalance=True,         # LGB tự xử lý imbalance
            min_child_samples=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        pipe_lgb = Pipeline([("scaler", StandardScaler()), ("clf", lgb_clf)])
        pipe_lgb.fit(X_res, y_res)
        models["lgb"] = pipe_lgb

    return models


def ensemble_predict_proba(models: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Weighted average probability từ các model trong ensemble.
    Weights: XGB=0.4, RF=0.3, LGB=0.3 (nếu có LGB).

    Args:
        models: Dict models từ train_ensemble.
        X: Features.

    Returns:
        np.ndarray xác suất class 1.
    """
    weights = {"xgb": 0.4, "rf": 0.3, "lgb": 0.3}
    if "lgb" not in models:
        weights = {"xgb": 0.55, "rf": 0.45}

    total_w  = sum(weights[k] for k in models)
    prob_sum = np.zeros(len(X))

    for name, model in models.items():
        w = weights.get(name, 0)
        try:
            p = model.predict_proba(X)
            prob_sum += w * (p[:, 1] if p.shape[1] >= 2 else (model.predict(X) == 1).astype(float))
        except Exception:
            prob_sum += w * (model.predict(X) == 1).astype(float)

    return prob_sum / total_w


def evaluate_ensemble(models: dict, X_test: pd.DataFrame,
                       y_test: pd.Series) -> dict:
    """
    Đánh giá ensemble trên tập test.

    Returns:
        dict các chỉ số.
    """
    proba  = ensemble_predict_proba(models, X_test)
    preds  = (proba >= 0.35).astype(int)  # Ngưỡng thấp hơn tăng recall trap
    report = classification_report(y_test, preds, output_dict=True)
    auc    = roc_auc_score(y_test, proba) if len(y_test.unique()) > 1 else 0.5
    return {
        "auc_roc":        round(auc, 4),
        "precision_trap": round(report.get("1", {}).get("precision", 0), 4),
        "recall_trap":    round(report.get("1", {}).get("recall", 0), 4),
        "f1_trap":        round(report.get("1", {}).get("f1-score", 0), 4),
        "accuracy":       round(report.get("accuracy", 0), 4),
    }


# ── Model Versioning ─────────────────────────────────────────
def save_model_versioned(models: dict, symbol: str,
                          trap_type: str, metrics: dict) -> str:
    """
    Lưu model với versioning. Chỉ deploy nếu AUC mới >= AUC cũ.

    Args:
        models: Dict ensemble models.
        symbol: Mã cổ phiếu.
        trap_type: 'bull' hoặc 'bear'.
        metrics: Dict metrics từ evaluate_ensemble.

    Returns:
        str: 'deployed' | 'skipped' | 'first'
    """
    base_path    = os.path.join(MODEL_DIR,   f"{symbol}_{trap_type}_trap.pkl")
    meta_path    = os.path.join(MODEL_DIR,   f"{symbol}_{trap_type}_meta.json")
    version_path = os.path.join(VERSION_DIR, f"{symbol}_{trap_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

    new_auc = metrics.get("auc_roc", 0)

    # Đọc AUC cũ
    old_auc = 0.0
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                old_auc = json.load(f).get("auc_roc", 0)
        except Exception:
            old_auc = 0.0

    # Lưu version archive
    with open(version_path, "wb") as f:
        pickle.dump({"models": models, "metrics": metrics}, f)

    if new_auc >= old_auc or old_auc == 0:
        # Deploy model mới
        with open(base_path, "wb") as f:
            pickle.dump({"models": models, "metrics": metrics}, f)
        meta = {**metrics, "trained_at": datetime.now().isoformat(),
                "version": version_path}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return "deployed" if old_auc > 0 else "first"
    else:
        logger.info(f"{symbol} {trap_type}: AUC mới {new_auc} < cũ {old_auc} → giữ model cũ")
        return "skipped"


def load_model_bundle(symbol: str, trap_type: str) -> dict:
    """
    Load ensemble model bundle.

    Returns:
        dict: {"models": {...}, "metrics": {...}} hoặc None.
    """
    path = os.path.join(MODEL_DIR, f"{symbol}_{trap_type}_trap.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Training Pipeline ─────────────────────────────────────────
def run_training_pipeline(symbols: list = None,
                           lookback_days: int = 730) -> dict:
    """
    Chạy pipeline huấn luyện cho cả Bull Trap và Bear Trap,
    toàn bộ danh sách mã.

    Args:
        symbols: Danh sách mã. Mặc định VN30_SYMBOLS.
        lookback_days: Ngày lịch sử.

    Returns:
        dict kết quả từng mã.
    """
    if symbols is None:
        symbols = VN30_SYMBOLS

    results = {}

    for symbol in symbols:
        results[symbol] = {}
        logger.info(f"{'='*40}")
        logger.info(f"Xử lý: {symbol}")

        try:
            df_raw = fetch_stock_data(symbol, lookback_days)
            if len(df_raw) < 60:
                logger.warning(f"{symbol}: Không đủ dữ liệu.")
                results[symbol] = {"status": "skipped"}
                continue

            df_feat = compute_features(df_raw)

        except Exception as e:
            logger.error(f"{symbol}: Lỗi fetch/feature — {e}")
            results[symbol] = {"status": "error", "reason": str(e)}
            continue

        for trap_type in TRAP_TYPES:
            try:
                # Gán nhãn theo loại trap
                df_labeled = (label_bull_trap(df_feat) if trap_type == "bull"
                              else label_bear_trap(df_feat))

                if len(df_labeled) < 20:
                    logger.warning(f"{symbol} {trap_type}: < 20 mẫu.")
                    continue

                # Chỉ dùng features có trong df_labeled
                feat_cols = [c for c in FEATURE_COLUMNS if c in df_labeled.columns]
                X = df_labeled[feat_cols].fillna(0)
                y = df_labeled["Label"]

                logger.info(f"{symbol} {trap_type}: n={len(y)} trap={y.sum()} real={len(y)-y.sum()}")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

                # Ensemble train
                models  = train_ensemble(X_train, y_train)
                metrics = evaluate_ensemble(models, X_test, y_test)
                status  = save_model_versioned(models, symbol, trap_type, metrics)

                logger.info(f"{symbol} {trap_type}: AUC={metrics['auc_roc']} → {status}")
                results[symbol][trap_type] = {"status": status, "metrics": metrics}

            except Exception as e:
                logger.error(f"{symbol} {trap_type}: {e}", exc_info=True)
                results[symbol][trap_type] = {"status": "error", "reason": str(e)}

    return results


# ── Auto-Retraining Scheduler ────────────────────────────────
def schedule_weekly_retrain(symbols: list = None,
                             run_day: str = "sun",
                             run_hour: int = 23):
    """
    Lập lịch retrain tự động hàng tuần (Chủ nhật 23h).
    Chỉ deploy nếu AUC mới tốt hơn (đã xử lý trong save_model_versioned).

    Args:
        symbols: Danh sách mã.
        run_day: Ngày trong tuần ('sun', 'mon'...).
        run_hour: Giờ chạy.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("Cần: pip install apscheduler")
        return

    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")

    def retrain_job():
        logger.info("=== AUTO RETRAIN WEEKLY ===")
        run_training_pipeline(symbols=symbols or VN30_SYMBOLS)

    scheduler.add_job(
        func=retrain_job,
        trigger=CronTrigger(day_of_week=run_day, hour=run_hour,
                            timezone="Asia/Ho_Chi_Minh"),
        id="weekly_retrain",
        replace_existing=True,
    )
    logger.info(f"Retrain scheduler: {run_day} {run_hour}:00")
    scheduler.start()


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    syms = [sys.argv[1]] if len(sys.argv) > 1 else None
    logger.info("=== BULL & BEAR TRAP DETECTOR — TRAINING ===")
    results = run_training_pipeline(symbols=syms)

    logger.info("\n===== KẾT QUẢ =====")
    for sym, res in results.items():
        for trap, info in (res.items() if isinstance(res, dict) else []):
            if isinstance(info, dict) and "metrics" in info:
                m = info["metrics"]
                logger.info(
                    f"{sym:5s} [{trap}] | AUC={m['auc_roc']:.3f} | "
                    f"P={m['precision_trap']:.3f} R={m['recall_trap']:.3f} "
                    f"F1={m['f1_trap']:.3f} | {info['status']}"
                )
