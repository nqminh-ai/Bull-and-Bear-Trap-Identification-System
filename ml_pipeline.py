"""
ml_pipeline.py
==============
Pipeline huấn luyện mô hình phát hiện "Bẫy giá Bull Trap" cho các mã VN30.
Kết nối: SQL Server NGUYEN_MINH | Database: ProjectADY_StockDB | Bảng: dbo.daily_prices
Nguyên tắc: Tuyệt đối không dùng print() trong hàm xử lý, mọi hàm phải return giá trị.
"""

import os
import logging
import warnings
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyodbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# CẤU HÌNH KẾT NỐI DATABASE
# ============================================================
DB_SERVER   = "NGUYEN_MINH"
DB_DATABASE = "ProjectADY_StockDB"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"

# Danh sách mã giao dịch (VN30 + mã bổ sung)
VN30_SYMBOLS = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "DXG", "FPT", "GAS", "GVR", "HDB",
    "HPG", "MBB", "MSN", "MWG", "NVL",
    "PDR", "PLX", "POW", "SAB", "SHB",
    "SSI", "STB", "TCB", "TPB", "VCB",
    "VHM", "VIB", "VIC", "VJC", "VNM", "VPB",
]

# Thư mục lưu model
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Cột features sử dụng cho model
FEATURE_COLUMNS = [
    "RSI_14",
    "MA20",
    "Volume_Ratio",
    "Upper_Shadow_Ratio",
    "Body_Range_Ratio",
    "Is_Doji",
    "OBV",
    "OBV_MA20",
    "OBV_Divergence",
    "Price_vs_MA20",
]


# ============================================================
# HÀM KẾT NỐI DATABASE (pyodbc thuần - không dùng SQLAlchemy)
# ============================================================
def get_connection():
    """
    Tạo kết nối pyodbc trực tiếp đến SQL Server bằng Windows Authentication.

    Returns:
        pyodbc.Connection
    """
    conn_str = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


def fetch_stock_data(symbol: str, lookback_days: int = 730) -> pd.DataFrame:
    """
    Lấy dữ liệu OHLCV từ bảng dbo.daily_prices.

    Cấu trúc bảng thật:
        time    -> Date
        open    -> Open
        high    -> High
        low     -> Low
        close   -> Close
        volume  -> Volume
        ticker  -> Symbol

    Args:
        symbol: Mã cổ phiếu (VD: 'VNM').
        lookback_days: Số ngày lịch sử cần lấy.

    Returns:
        pd.DataFrame với index Date và cột [Open, High, Low, Close, Volume].
    """
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Dùng f-string query - tránh lỗi 8180 parameterized query với pyodbc
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
          AND [time] >= '{cutoff_date}'
        ORDER BY [time] ASC
    """
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()

    df = df.rename(columns={"TradingDate":"Date","OpenPrice":"Open","HighPrice":"High","LowPrice":"Low","ClosePrice":"Close","TotalVolume":"Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


# ============================================================
# HÀM FEATURE ENGINEERING
# ============================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Tính chỉ báo RSI (Relative Strength Index).

    Args:
        series: Chuỗi giá đóng cửa.
        period: Số kỳ tính RSI (mặc định 14).

    Returns:
        pd.Series chứa giá trị RSI.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    Tính chỉ báo OBV (On-Balance Volume) tiêu chuẩn.

    Args:
        df: DataFrame với cột Close và Volume.

    Returns:
        pd.Series chứa giá trị OBV.
    """
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toàn bộ features nhận diện hành vi "cá mập" và phân kỳ dòng tiền.

    Args:
        df: DataFrame OHLCV với index là Date.

    Returns:
        pd.DataFrame chứa tất cả features đã tính (bỏ NaN).
    """
    result = df.copy()

    # --- Trend & Momentum ---
    result["RSI_14"] = compute_rsi(result["Close"], period=14)
    result["MA20"] = result["Close"].rolling(window=20).mean()
    result["Price_vs_MA20"] = result["Close"] - result["MA20"]

    # --- Dấu hiệu 1: Khối lượng thấp ---
    result["Volume_MA20"] = result["Volume"].rolling(window=20).mean()
    result["Volume_Ratio"] = result["Volume"] / result["Volume_MA20"].replace(0, np.nan)

    # --- Dấu hiệu 2: Yếu lực mua (Upper Shadow Rejection) ---
    result["Upper_Shadow"] = result["High"] - result[["Open", "Close"]].max(axis=1)
    result["Body"] = (result["Close"] - result["Open"]).abs()
    result["Upper_Shadow_Ratio"] = result["Upper_Shadow"] / result["Body"].replace(0, 1e-6)

    # --- Dấu hiệu 3: Nến Doji / Bóng dài ---
    result["Range"] = result["High"] - result["Low"]
    result["Body_Range_Ratio"] = result["Body"] / result["Range"].replace(0, np.nan)
    result["Is_Doji"] = (result["Body_Range_Ratio"] < 0.1).astype(int)

    # --- Dấu hiệu 4: Phân kỳ OBV ---
    result["OBV"] = compute_obv(result)
    result["OBV_MA20"] = result["OBV"].rolling(window=20).mean()
    price_up = result["Close"] > result["MA20"]
    money_out = result["OBV"] < result["OBV_MA20"]
    result["OBV_Divergence"] = (price_up & money_out).astype(int)

    # Loại bỏ hàng có NaN trong features
    result = result.dropna(subset=FEATURE_COLUMNS)
    return result


def label_bull_trap(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Gán nhãn Bull Trap dựa trên hành vi phá đỉnh giả.

    Nhãn 1 (Bull Trap): Giá cắt lên MA20 NHƯNG giá đóng cửa T+5 thấp hơn giá mua.
    Nhãn 0 (Breakout thật): Breakout và giá T+5 >= giá mua.

    Args:
        df: DataFrame đã có features.
        forward_days: Số ngày nhìn về phía trước.

    Returns:
        pd.DataFrame chỉ chứa các ngày Breakout có nhãn.
    """
    result = df.copy()

    prev_close = result["Close"].shift(1)
    prev_ma20  = result["MA20"].shift(1)
    result["Is_Breakout"] = (result["Close"] > result["MA20"]) & (prev_close <= prev_ma20)
    result["Future_Close"] = result["Close"].shift(-forward_days)

    result["Label"] = np.where(
        result["Is_Breakout"] & (result["Future_Close"] < result["Close"]),
        1,   # Bull Trap
        np.where(
            result["Is_Breakout"] & (result["Future_Close"] >= result["Close"]),
            0,   # Breakout thật
            np.nan,
        ),
    )

    labeled = result.dropna(subset=["Label"]).copy()
    labeled["Label"] = labeled["Label"].astype(int)
    return labeled


# ============================================================
# HÀM HUẤN LUYỆN MODEL
# ============================================================
def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "xgb"):
    """
    Huấn luyện mô hình XGBoost hoặc RandomForest với xử lý mất cân bằng nhãn.

    Args:
        X_train: DataFrame features.
        y_train: Series nhãn (0/1).
        model_type: 'xgb' hoặc 'rf'.

    Returns:
        sklearn Pipeline đã huấn luyện.
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    if model_type == "xgb":
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,  # Xử lý mất cân bằng nhãn
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",            # Xử lý mất cân bằng nhãn
            random_state=42,
            n_jobs=-1,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Đánh giá mô hình trên tập test.

    Returns:
        dict chứa các chỉ số AUC, Precision, Recall, F1.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True)
    auc     = roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0.5

    return {
        "auc_roc":        round(auc, 4),
        "precision_trap": round(report.get("1", {}).get("precision", 0), 4),
        "recall_trap":    round(report.get("1", {}).get("recall", 0), 4),
        "f1_trap":        round(report.get("1", {}).get("f1-score", 0), 4),
        "accuracy":       round(report.get("accuracy", 0), 4),
    }


def save_model(model, symbol: str) -> str:
    """
    Lưu model ra file .pkl.

    Returns:
        str: Đường dẫn file .pkl.
    """
    filepath = os.path.join(MODEL_DIR, f"{symbol}_bull_trap.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return filepath


# ============================================================
# PIPELINE CHÍNH
# ============================================================
def run_training_pipeline(
    symbols: list = None,
    model_type: str = "xgb",
    lookback_days: int = 730,
) -> dict:
    """
    Chạy vòng lặp huấn luyện model cho toàn bộ danh sách mã VN30.

    Args:
        symbols: Danh sách mã. Mặc định VN30_SYMBOLS.
        model_type: 'xgb' hoặc 'rf'.
        lookback_days: Số ngày lịch sử dữ liệu.

    Returns:
        dict kết quả huấn luyện từng mã.
    """
    if symbols is None:
        symbols = VN30_SYMBOLS

    results = {}

    for symbol in symbols:
        logger.info(f"Đang xử lý mã: {symbol}")
        try:
            # 1. Lấy dữ liệu từ SQL Server
            df_raw = fetch_stock_data(symbol, lookback_days=lookback_days)
            if len(df_raw) < 60:
                logger.warning(f"{symbol}: Không đủ dữ liệu (< 60 phiên). Bỏ qua.")
                results[symbol] = {"status": "skipped", "reason": "insufficient_data"}
                continue

            # 2. Tính features
            df_features = compute_features(df_raw)

            # 3. Gán nhãn Bull Trap
            df_labeled = label_bull_trap(df_features, forward_days=5)
            if len(df_labeled) < 20:
                logger.warning(f"{symbol}: Không đủ mẫu Breakout (< 20). Bỏ qua.")
                results[symbol] = {"status": "skipped", "reason": "insufficient_samples"}
                continue

            # 4. Chuẩn bị X, y
            X = df_labeled[FEATURE_COLUMNS]
            y = df_labeled["Label"]
            logger.info(f"{symbol}: Tổng={len(y)} | BullTrap={y.sum()} | Thật={len(y)-y.sum()}")

            # 5. Chia train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # 6. Huấn luyện
            model = train_model(X_train, y_train, model_type=model_type)

            # 7. Đánh giá
            metrics = evaluate_model(model, X_test, y_test)
            logger.info(f"{symbol}: AUC={metrics['auc_roc']} | F1={metrics['f1_trap']}")

            # 8. Lưu model
            model_path = save_model(model, symbol)
            logger.info(f"{symbol}: Đã lưu -> {model_path}")

            results[symbol] = {
                "status":     "success",
                "metrics":    metrics,
                "model_path": model_path,
                "n_samples":  len(y),
                "n_bull_trap": int(y.sum()),
            }

        except Exception as e:
            logger.error(f"{symbol}: Lỗi - {e}", exc_info=True)
            results[symbol] = {"status": "error", "reason": str(e)}

    success_count = sum(1 for v in results.values() if v["status"] == "success")
    logger.info(f"\nHoàn thành: {success_count}/{len(symbols)} mã thành công.")
    return results


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    logger.info("=== BẮT ĐẦU PIPELINE HUẤN LUYỆN BULL TRAP DETECTOR ===")
    final_results = run_training_pipeline(
        symbols=VN30_SYMBOLS,
        model_type="xgb",
        lookback_days=730,
    )

    logger.info("\n===== KẾT QUẢ HUẤN LUYỆN =====")
    for sym, res in final_results.items():
        if res["status"] == "success":
            m = res["metrics"]
            logger.info(
                f"{sym:5s} | AUC={m['auc_roc']:.3f} | "
                f"P={m['precision_trap']:.3f} | "
                f"R={m['recall_trap']:.3f} | "
                f"F1={m['f1_trap']:.3f}"
            )
        else:
            logger.info(f"{sym:5s} | {res['status'].upper()}: {res.get('reason', '')}")
