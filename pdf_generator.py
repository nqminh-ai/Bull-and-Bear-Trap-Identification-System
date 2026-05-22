"""
pdf_generator.py
================
Tự động tạo báo cáo PDF Bull Trap cho danh sách mã VN30.
Kết nối: SQL Server NGUYEN_MINH | Database: ProjectADY_StockDB | Bảng: dbo.daily_prices
Nguyên tắc: Tuyệt đối không dùng print() trong hàm xử lý, mọi hàm phải return.
"""

import os
import pickle
import logging
import warnings
import base64
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyodbc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jinja2 import Environment, BaseLoader

try:
    import pdfkit
    PDF_BACKEND = "pdfkit"
except ImportError:
    try:
        from weasyprint import HTML as WeasyprintHTML
        PDF_BACKEND = "weasyprint"
    except ImportError:
        PDF_BACKEND = None

warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pdf_report.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# CẤU HÌNH HỆ THỐNG
# ============================================================
DB_SERVER   = "NGUYEN_MINH"
DB_DATABASE = "ProjectADY_StockDB"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"

MODEL_DIR = "models"
REPORT_DIR = "reports"
CHART_DIR  = os.path.join(REPORT_DIR, "charts")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR,  exist_ok=True)

FEATURE_COLUMNS = [
    "RSI_14", "MA20", "Volume_Ratio", "Upper_Shadow_Ratio",
    "Body_Range_Ratio", "Is_Doji", "OBV", "OBV_MA20",
    "OBV_Divergence", "Price_vs_MA20",
]

VN30_SYMBOLS = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "NVL", "PDR",
    "PLX", "POW", "SAB", "SHB", "SSI",
    "STB", "TCB", "TPB", "VCB", "VHM",
    "VIB", "VIC", "VJC", "VNM", "VPB",
]

# ============================================================
# TEMPLATE HTML (JINJA2)
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: Arial, sans-serif; background:#fff; color:#1a1a2e; font-size:13px; line-height:1.6; }
  .page { max-width:900px; margin:0 auto; padding:30px 40px; }
  .report-header {
    background: linear-gradient(135deg, #0d1117 0%, #1c2128 100%);
    color:#e6edf3; padding:28px 36px; border-radius:12px;
    margin-bottom:28px; display:flex; justify-content:space-between; align-items:center;
  }
  .report-header h1 { font-size:22px; font-weight:700; }
  .report-header .subtitle { font-size:13px; color:#8b949e; margin-top:4px; }
  .report-header .timestamp { text-align:right; font-size:12px; color:#8b949e; }
  .report-header .timestamp strong { font-size:16px; color:#d29922; display:block; }
  .section-title {
    font-size:15px; font-weight:600; color:#0d1117;
    margin:24px 0 12px; padding-bottom:6px; border-bottom:2px solid #e1e4e8;
  }
  .summary-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
  .metric-box { border:1px solid #e1e4e8; border-radius:8px; padding:14px; text-align:center; }
  .metric-box.danger  { border-left:4px solid #cf222e; background:#fff0f0; }
  .metric-box.warning { border-left:4px solid #d29922; background:#fffbea; }
  .metric-box.safe    { border-left:4px solid #1a7f37; background:#f0fff4; }
  .metric-box.neutral { border-left:4px solid #0969da; background:#f0f8ff; }
  .metric-label { font-size:11px; color:#6e7781; text-transform:uppercase; margin-bottom:6px; }
  .metric-value { font-size:22px; font-weight:700; color:#1a1a2e; }
  .metric-sub   { font-size:11px; color:#6e7781; margin-top:4px; }
  table { width:100%; border-collapse:collapse; margin-bottom:20px; font-size:12px; }
  thead { background:#0d1117; color:#e6edf3; }
  thead th { padding:10px 12px; text-align:left; font-weight:600; }
  tbody tr:nth-child(even) { background:#f6f8fa; }
  tbody td { padding:8px 12px; border-bottom:1px solid #e1e4e8; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; }
  .badge-danger  { background:#ff818266; color:#cf222e; }
  .badge-safe    { background:#abf2bc66; color:#1a7f37; }
  .badge-warning { background:#ffe08566; color:#9a6700; }
  .chart-container { border:1px solid #e1e4e8; border-radius:8px; overflow:hidden; margin-bottom:20px; }
  .chart-container img { width:100%; display:block; }
  .chart-caption { font-size:11px; color:#6e7781; text-align:center; padding:8px; background:#f6f8fa; border-top:1px solid #e1e4e8; }
  .alert { padding:12px 16px; border-radius:6px; margin-bottom:16px; font-size:12px; }
  .alert-danger { background:#fff0f0; border:1px solid #cf222e; color:#cf222e; }
  .report-footer { margin-top:32px; padding-top:16px; border-top:1px solid #e1e4e8; font-size:11px; color:#8b949e; text-align:center; }
</style>
</head>
<body>
<div class="page">
  <div class="report-header">
    <div>
      <h1>⚠️ BÁO CÁO PHÂN TÍCH BULL TRAP</h1>
      <div class="subtitle">Hệ thống phát hiện bẫy giá VN30 — ML Powered</div>
    </div>
    <div class="timestamp">
      <strong>{{ report_date }}</strong>
      Tạo lúc: {{ generated_at }}
    </div>
  </div>

  <div class="section-title">📊 TỔNG QUAN PHIÊN PHÂN TÍCH</div>
  <div class="summary-grid">
    <div class="metric-box neutral">
      <div class="metric-label">Số mã phân tích</div>
      <div class="metric-value">{{ total_symbols }}</div>
      <div class="metric-sub">mã VN30</div>
    </div>
    <div class="metric-box danger">
      <div class="metric-label">Cảnh báo Bull Trap</div>
      <div class="metric-value">{{ high_risk_count }}</div>
      <div class="metric-sub">xác suất > 50%</div>
    </div>
    <div class="metric-box warning">
      <div class="metric-label">Cần theo dõi</div>
      <div class="metric-value">{{ medium_risk_count }}</div>
      <div class="metric-sub">xác suất 30–50%</div>
    </div>
    <div class="metric-box safe">
      <div class="metric-label">Tín hiệu an toàn</div>
      <div class="metric-value">{{ low_risk_count }}</div>
      <div class="metric-sub">xác suất < 30%</div>
    </div>
  </div>

  {% if high_risk_symbols %}
  <div class="alert alert-danger">
    🚨 <strong>CẢNH BÁO:</strong> Nguy cơ Bull Trap cao:
    <strong>{{ high_risk_symbols | join(', ') }}</strong>
  </div>
  {% endif %}

  {% for chart_info in charts %}
  <div class="section-title">📈 {{ chart_info.symbol }} — Chi tiết</div>
  <div class="chart-container">
    <img src="data:image/png;base64,{{ chart_info.chart_b64 }}">
    <div class="chart-caption">
      Phiên: {{ chart_info.latest_date }} | Xác suất Bull Trap: {{ chart_info.prob_str }}
    </div>
  </div>
  {% endfor %}

  <div class="section-title">📋 BẢNG KẾT QUẢ</div>
  <table>
    <thead>
      <tr>
        <th>Mã CP</th><th>Giá đóng cửa</th><th>RSI(14)</th>
        <th>Volume Ratio</th><th>Phân kỳ OBV</th>
        <th>Xác suất Bẫy</th><th>Rủi ro</th>
      </tr>
    </thead>
    <tbody>
    {% for row in table_rows %}
      <tr>
        <td><strong>{{ row.symbol }}</strong></td>
        <td>{{ row.close }}</td>
        <td>{{ row.rsi }}</td>
        <td>{{ row.vol_ratio }}</td>
        <td>{{ '⚡ CÓ' if row.divergence else '—' }}</td>
        <td><strong>{{ row.prob_str }}</strong></td>
        <td><span class="badge badge-{{ row.risk_class }}">{{ row.risk_label }}</span></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <div class="section-title">📝 GHI CHÚ</div>
  <p style="font-size:12px;color:#444;line-height:1.7;">
    Mô hình <strong>XGBoost/RandomForest</strong> huấn luyện trên dữ liệu lịch sử VN30.
    Features: RSI(14), MA20, Volume Ratio, Upper Shadow Rejection, Doji, OBV Divergence.
    Bull Trap = giá phá MA20 nhưng T+5 thấp hơn giá mua.<br><br>
    <em>⚠️ Báo cáo mang tính tham khảo, không phải khuyến nghị đầu tư.</em>
  </p>

  <div class="report-footer">
    Bull Trap Detector | Server: NGUYEN_MINH | DB: ProjectADY_StockDB | {{ generated_at }}
  </div>
</div>
</body>
</html>
"""


# ============================================================
# HÀM DATABASE
# ============================================================
def get_connection():
    """
    Tạo kết nối pyodbc đến SQL Server NGUYEN_MINH.

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


def fetch_latest_data(symbol: str, lookback_days: int = 90) -> pd.DataFrame:
    """
    Lấy dữ liệu OHLCV mới nhất từ dbo.daily_prices.

    Args:
        symbol: Mã cổ phiếu.
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

    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


# ============================================================
# HÀM FEATURE ENGINEERING
# ============================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Tính RSI. Returns pd.Series."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Tính OBV. Returns pd.Series."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toàn bộ features Bull Trap. Returns pd.DataFrame."""
    r = df.copy()
    r["RSI_14"]          = compute_rsi(r["Close"])
    r["MA20"]            = r["Close"].rolling(20).mean()
    r["Price_vs_MA20"]   = r["Close"] - r["MA20"]
    r["Volume_MA20"]     = r["Volume"].rolling(20).mean()
    r["Volume_Ratio"]    = r["Volume"] / r["Volume_MA20"].replace(0, np.nan)
    r["Upper_Shadow"]    = r["High"] - r[["Open", "Close"]].max(axis=1)
    r["Body"]            = (r["Close"] - r["Open"]).abs()
    r["Upper_Shadow_Ratio"] = r["Upper_Shadow"] / r["Body"].replace(0, 1e-6)
    r["Range"]           = r["High"] - r["Low"]
    r["Body_Range_Ratio"] = r["Body"] / r["Range"].replace(0, np.nan)
    r["Is_Doji"]         = (r["Body_Range_Ratio"] < 0.1).astype(int)
    r["OBV"]             = compute_obv(r)
    r["OBV_MA20"]        = r["OBV"].rolling(20).mean()
    price_up             = r["Close"] > r["MA20"]
    money_out            = r["OBV"] < r["OBV_MA20"]
    r["OBV_Divergence"]  = (price_up & money_out).astype(int)
    return r


# ============================================================
# HÀM DỰ ĐOÁN
# ============================================================
def load_model(symbol: str):
    """Load model .pkl. Returns model hoặc None."""
    path = os.path.join(MODEL_DIR, f"{symbol}_bull_trap.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_latest(df_features: pd.DataFrame, model) -> dict:
    """
    Dự đoán Bull Trap cho phiên gần nhất.

    Returns:
        dict thông tin dự đoán.
    """
    valid = df_features[FEATURE_COLUMNS].dropna()
    if valid.empty or model is None:
        return {"error": True, "probability": 0.0}

    last_row = valid.iloc[[-1]]
    pred     = int(model.predict(last_row)[0])

    try:
        probas = model.predict_proba(last_row)
        prob   = float(probas[0, 1]) if probas.shape[1] >= 2 else float(pred)
    except (IndexError, AttributeError):
        prob = float(pred)

    latest_date = last_row.index[0]
    return {
        "error":        False,
        "prediction":   pred,
        "probability":  prob,
        "latest_date":  latest_date.strftime("%d/%m/%Y"),
        "latest_close": float(df_features.loc[latest_date, "Close"]),
        "rsi":          float(df_features.loc[latest_date, "RSI_14"]),
        "vol_ratio":    float(df_features.loc[latest_date, "Volume_Ratio"]),
        "divergence":   int(df_features.loc[latest_date, "OBV_Divergence"]),
    }


# ============================================================
# HÀM VẼ BIỂU ĐỒ MATPLOTLIB (TĨNH)
# ============================================================
def draw_static_chart(df: pd.DataFrame, symbol: str, pred_info: dict) -> str:
    """
    Vẽ biểu đồ tĩnh Candlestick + MA20 + OBV bằng Matplotlib.

    Args:
        df: DataFrame features.
        symbol: Mã cổ phiếu.
        pred_info: Dict thông tin dự đoán.

    Returns:
        str: Đường dẫn file PNG.
    """
    plot_df = df.dropna(subset=["MA20", "OBV_MA20"]).tail(60).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios": [2, 1]},
                                   sharex=True)
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")

    dates     = np.arange(len(plot_df))
    up_mask   = plot_df["Close"] >= plot_df["Open"]
    down_mask = ~up_mask

    # Thân nến tăng
    ax1.bar(dates[up_mask],
            plot_df.loc[up_mask.values,   "Close"] - plot_df.loc[up_mask.values,   "Open"],
            width=0.6, bottom=plot_df.loc[up_mask.values, "Open"],
            color="#3fb950", zorder=2)
    # Thân nến giảm
    ax1.bar(dates[down_mask],
            plot_df.loc[down_mask.values, "Open"]  - plot_df.loc[down_mask.values, "Close"],
            width=0.6, bottom=plot_df.loc[down_mask.values, "Close"],
            color="#f85149", zorder=2)
    # Bóng nến
    ax1.vlines(dates[up_mask],   plot_df.loc[up_mask.values,   "Low"], plot_df.loc[up_mask.values,   "High"], color="#3fb950", linewidth=1)
    ax1.vlines(dates[down_mask], plot_df.loc[down_mask.values, "Low"], plot_df.loc[down_mask.values, "High"], color="#f85149", linewidth=1)

    # MA20
    ax1.plot(dates, plot_df["MA20"], color="#d29922", linewidth=1.5, linestyle="--", label="MA20")

    # Marker Bull Trap / An toàn tại Low * 0.98
    prob = pred_info.get("probability", 0)
    if prob > 0.5:
        ax1.scatter(len(plot_df) - 1, plot_df["Low"].iloc[-1] * 0.98,
                    marker="^", color="#f85149", s=120, zorder=5,
                    label=f"⚠ Bull Trap ({prob:.0%})")
    elif prob < 0.3:
        ax1.scatter(len(plot_df) - 1, plot_df["Low"].iloc[-1] * 0.98,
                    marker="^", color="#3fb950", s=100, zorder=5,
                    label=f"✓ An toàn ({1-prob:.0%})")

    ax1.set_title(f"{symbol} | {pred_info.get('latest_date', '')}",
                  color="#e6edf3", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", facecolor="#1c2128", edgecolor="#30363d",
               labelcolor="#e6edf3", fontsize=9)
    for ax in [ax1, ax2]:
        ax.tick_params(colors="#8b949e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.grid(axis="y", color="#21262d", linewidth=0.5)
    ax1.set_ylabel("Giá", color="#8b949e")

    # OBV subplot
    ax2.plot(dates, plot_df["OBV"],     color="#58a6ff", linewidth=1.5, label="OBV")
    ax2.plot(dates, plot_df["OBV_MA20"], color="#d29922", linewidth=1.5, linestyle="--", label="OBV MA20")
    ax2.fill_between(dates, plot_df["OBV"], alpha=0.1, color="#58a6ff")

    div_mask = plot_df["OBV_Divergence"] == 1
    if div_mask.any():
        ax2.scatter(dates[div_mask.values], plot_df.loc[div_mask.values, "OBV"],
                    color="#f85149", s=30, zorder=5, label="Phân kỳ")

    ax2.set_ylabel("OBV", color="#8b949e")
    ax2.legend(loc="upper left", facecolor="#1c2128", edgecolor="#30363d",
               labelcolor="#e6edf3", fontsize=9)

    # Nhãn trục X
    step        = max(1, len(plot_df) // 8)
    tick_pos    = dates[::step]
    tick_labels = [plot_df.index[i].strftime("%d/%m") for i in range(0, len(plot_df), step)]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, rotation=30, ha="right", color="#8b949e", fontsize=8)

    plt.tight_layout(pad=2.0)
    filepath = os.path.join(CHART_DIR, f"{symbol}_{datetime.now().strftime('%Y%m%d')}_chart.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return filepath


def image_to_base64(filepath: str) -> str:
    """Chuyển ảnh thành base64 để nhúng HTML. Returns str."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
# HÀM TẠO HTML VÀ PDF
# ============================================================
def render_html(template_data: dict) -> str:
    """Render Jinja2 template. Returns str HTML."""
    env  = Environment(loader=BaseLoader())
    tmpl = env.from_string(HTML_TEMPLATE)
    return tmpl.render(**template_data)


def save_html(html_content: str, filename: str) -> str:
    """Lưu HTML ra file. Returns str đường dẫn."""
    filepath = os.path.join(REPORT_DIR, f"{filename}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    return filepath


def convert_to_pdf(html_path: str, pdf_path: str) -> bool:
    """
    Chuyển HTML sang PDF dùng pdfkit hoặc weasyprint.

    Returns:
        bool: True nếu thành công.
    """
    if PDF_BACKEND == "pdfkit":
        options = {
            "page-size": "A4", "margin-top": "10mm", "margin-right": "10mm",
            "margin-bottom": "10mm", "margin-left": "10mm",
            "encoding": "UTF-8", "enable-local-file-access": "",
        }
        try:
            pdfkit.from_file(html_path, pdf_path, options=options)
            return True
        except Exception as e:
            logger.error(f"pdfkit lỗi: {e}")
            return False
    elif PDF_BACKEND == "weasyprint":
        try:
            WeasyprintHTML(filename=html_path).write_pdf(pdf_path)
            return True
        except Exception as e:
            logger.error(f"weasyprint lỗi: {e}")
            return False
    else:
        logger.error("Không có thư viện PDF. Cài: pip install pdfkit hoặc weasyprint")
        return False


# ============================================================
# PIPELINE BÁO CÁO CHÍNH
# ============================================================
def generate_report(symbols: list, chart_symbols: list = None, report_name: str = None) -> str:
    """
    Tạo báo cáo PDF đầy đủ cho danh sách mã VN30.

    Args:
        symbols: Danh sách mã phân tích.
        chart_symbols: Mã cần vẽ biểu đồ (tối đa 3).
        report_name: Tên file output.

    Returns:
        str: Đường dẫn file PDF (hoặc HTML nếu PDF thất bại).
    """
    if chart_symbols is None:
        chart_symbols = symbols[:3]
    if report_name is None:
        report_name = f"bull_trap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    table_rows        = []
    charts_data       = []
    high_risk_symbols = []
    medium_risk_count = 0
    low_risk_count    = 0

    logger.info(f"Bắt đầu tạo báo cáo cho {len(symbols)} mã...")

    for symbol in symbols:
        logger.info(f"Phân tích {symbol}...")
        try:
            df_raw      = fetch_latest_data(symbol, lookback_days=90)
            if len(df_raw) < 30:
                continue

            df_features = compute_features(df_raw)
            model       = load_model(symbol)
            pred_info   = predict_latest(df_features, model)

            if pred_info["error"]:
                continue

            prob = pred_info["probability"]
            if prob > 0.5:
                high_risk_symbols.append(symbol)
                risk_class, risk_label = "danger",  "🔴 Cao"
            elif prob > 0.3:
                medium_risk_count += 1
                risk_class, risk_label = "warning", "🟡 Trung bình"
            else:
                low_risk_count += 1
                risk_class, risk_label = "safe",    "🟢 Thấp"

            table_rows.append({
                "symbol":     symbol,
                "close":      f"{pred_info['latest_close']:,.0f}",
                "rsi":        f"{pred_info['rsi']:.1f}",
                "vol_ratio":  f"{pred_info['vol_ratio']:.2f}x",
                "divergence": bool(pred_info["divergence"]),
                "prob_str":   f"{prob:.1%}",
                "risk_class": risk_class,
                "risk_label": risk_label,
            })

            if symbol in chart_symbols:
                chart_path = draw_static_chart(df_features, symbol, pred_info)
                charts_data.append({
                    "symbol":      symbol,
                    "chart_b64":   image_to_base64(chart_path),
                    "latest_date": pred_info["latest_date"],
                    "prob_str":    f"{prob:.1%}",
                })

        except Exception as e:
            logger.error(f"{symbol}: {e}", exc_info=True)

    table_rows.sort(key=lambda r: float(r["prob_str"].replace("%", "")), reverse=True)

    template_data = {
        "report_date":       datetime.now().strftime("%d/%m/%Y"),
        "generated_at":      datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        "total_symbols":     len(table_rows),
        "high_risk_count":   len(high_risk_symbols),
        "medium_risk_count": medium_risk_count,
        "low_risk_count":    low_risk_count,
        "high_risk_symbols": high_risk_symbols,
        "charts":            charts_data,
        "table_rows":        table_rows,
    }

    html_content = render_html(template_data)
    html_path    = save_html(html_content, report_name)
    logger.info(f"HTML: {html_path}")

    pdf_path = os.path.join(REPORT_DIR, f"{report_name}.pdf")
    success  = convert_to_pdf(html_path, pdf_path)

    if success:
        logger.info(f"✅ PDF: {pdf_path}")
        return pdf_path
    else:
        logger.warning(f"⚠️ Dùng HTML: {html_path}")
        return html_path


# ============================================================
# LẬP LỊCH TỰ ĐỘNG
# ============================================================
def schedule_daily_report(symbols: list, run_hour: int = 16, run_minute: int = 15):
    """
    Lập lịch tạo báo cáo tự động sau khi thị trường đóng cửa (16:15 hàng ngày).

    Args:
        symbols: Danh sách mã.
        run_hour: Giờ chạy.
        run_minute: Phút chạy.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("Cài: pip install apscheduler")
        return

    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")

    def job():
        logger.info("=== BẮT ĐẦU TẠO BÁO CÁO TỰ ĐỘNG ===")
        generate_report(symbols=symbols, chart_symbols=symbols[:5])

    scheduler.add_job(
        func=job,
        trigger=CronTrigger(hour=run_hour, minute=run_minute,
                            day_of_week="mon-fri", timezone="Asia/Ho_Chi_Minh"),
        id="daily_bull_trap_report",
        misfire_grace_time=300,
        replace_existing=True,
    )
    logger.info(f"Lập lịch: {run_hour:02d}:{run_minute:02d} Thứ 2 - Thứ 6")
    scheduler.start()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys

    if "--once" in sys.argv:
        logger.info("=== CHẾ ĐỘ CHẠY MỘT LẦN ===")
        output = generate_report(
            symbols=VN30_SYMBOLS,
            chart_symbols=["VNM", "VPB", "FPT"],
        )
        logger.info(f"Output: {output}")
    else:
        logger.info("=== CHẾ ĐỘ LẬP LỊCH TỰ ĐỘNG ===")
        schedule_daily_report(symbols=VN30_SYMBOLS, run_hour=16, run_minute=15)
