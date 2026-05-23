"""
import_vnindex.py
================
Script lấy dữ liệu lịch sử VNIndex và import vào MSSQL.

Nguồn (ưu tiên theo thứ tự):
  1. vnstock  — thư viện Python chính thức, dùng cafef/TCBS làm backend
  2. s.cafef.vn Ajax API — fallback nếu không có vnstock
  3. vietstock.vn — fallback thứ 3

Cách dùng:
    pip install vnstock
    python import_vnindex.py            # Import 730 ngày
    python import_vnindex.py 365        # Import 365 ngày
    python import_vnindex.py --schedule # Lập lịch 16:30 hàng ngày
"""

import sys
import time
import logging
import warnings
from datetime import datetime, timedelta

import requests
import pandas as pd
import pyodbc

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("vnindex_import.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DB_SERVER   = "NGUYEN_MINH"
DB_DATABASE = "ProjectADY_StockDB"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"
TICKER_NAME = "VNINDEX"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://cafef.vn/",
    "Accept":  "application/json, text/javascript, */*",
}


# ── DB ───────────────────────────────────────────────────────
def get_connection() -> pyodbc.Connection:
    conn_str = (
        f"DRIVER={{{DB_DRIVER}}};SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


# ── Nguồn 1: vnstock ─────────────────────────────────────────
def fetch_via_vnstock(lookback_days: int = 730) -> pd.DataFrame:
    """
    Dùng thư viện vnstock để lấy lịch sử VNIndex.
    Hỗ trợ cả vnstock 3.x (mới) và vnstock 2.x (cũ).

    Returns:
        pd.DataFrame [Date, Open, High, Low, Close, Volume] hoặc rỗng.
    """
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Thử vnstock 3.x
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol="VNINDEX", source="VCI")
        df    = stock.quote.history(start=start_date, end=end_date, interval="1D")

        if df is not None and not df.empty:
            # Chuẩn hóa tên cột
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if "time" in cl or "date" in cl: col_map[c] = "Date"
                elif cl == "open":               col_map[c] = "Open"
                elif cl == "high":               col_map[c] = "High"
                elif cl == "low":                col_map[c] = "Low"
                elif cl in ("close","price"):    col_map[c] = "Close"
                elif "vol" in cl:                col_map[c] = "Volume"
            df = df.rename(columns=col_map)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
            df = df[df["Close"] > 0].sort_values("Date").reset_index(drop=True)
            logger.info(f"vnstock 3.x: {len(df)} phiên")
            return df
    except Exception as e:
        logger.warning(f"vnstock 3.x: {e}")

    # Thử vnstock 2.x
    try:
        from vnstock import stock_historical_data
        df = stock_historical_data(
            symbol="VNINDEX", start_date=start_date,
            end_date=end_date, resolution="1D", type="index",
        )
        if df is not None and not df.empty:
            col_map = {"tradingDate":"Date","open":"Open","high":"High",
                       "low":"Low","close":"Close","volume":"Volume"}
            df = df.rename(columns=col_map)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
            df = df.sort_values("Date").reset_index(drop=True)
            logger.info(f"vnstock 2.x: {len(df)} phiên")
            return df
    except Exception as e:
        logger.warning(f"vnstock 2.x: {e}")

    return pd.DataFrame()


# ── Nguồn 2: s.cafef.vn ──────────────────────────────────────
def fetch_via_scafef(page: int = 1, page_size: int = 20) -> pd.DataFrame:
    """
    Crawl từ s.cafef.vn/lich-su-giao-dich-vnindex-{page}.chn
    bằng cách parse JSON từ Ajax endpoint mới.

    Returns:
        pd.DataFrame hoặc rỗng.
    """
    # Endpoint Ajax mới của s.cafef.vn
    url = f"https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    params = {
        "Symbol":    "VNINDEX",
        "StartDate": "",
        "EndDate":   "",
        "PageIndex": page,
        "PageSize":  page_size,
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return pd.DataFrame()

        data  = resp.json()
        items = (data.get("Data") or {}).get("Data") or []
        if not items:
            return pd.DataFrame()

        rows = []
        for d in items:
            try:
                date_raw = d.get("Ngay") or d.get("NgayGiaoDich") or ""
                date_raw = date_raw.strip().replace("T00:00:00", "")
                if not date_raw:
                    continue
                if "/" in date_raw:
                    dt = datetime.strptime(date_raw, "%d/%m/%Y")
                else:
                    dt = datetime.strptime(date_raw[:10], "%Y-%m-%d")

                rows.append({
                    "Date":   dt,
                    "Open":   float(d.get("GiaMoCua")    or d.get("Open")   or 0),
                    "High":   float(d.get("GiaCaoNhat")  or d.get("High")   or 0),
                    "Low":    float(d.get("GiaThapNhat") or d.get("Low")    or 0),
                    "Close":  float(d.get("GiaDongCua")  or d.get("Close")  or 0),
                    "Volume": float(d.get("KhoiLuongKhopLenh") or d.get("Volume") or 0),
                })
            except Exception:
                continue

        df = pd.DataFrame(rows)
        df = df[df["Close"] > 0] if not df.empty else df
        return df.sort_values("Date").reset_index(drop=True)

    except Exception as e:
        logger.warning(f"s.cafef trang {page}: {e}")
        return pd.DataFrame()


def fetch_scafef_all(lookback_days: int = 730) -> pd.DataFrame:
    """Crawl toàn bộ lịch sử từ s.cafef.vn."""
    cutoff  = datetime.now() - timedelta(days=lookback_days)
    all_dfs = []

    for page in range(1, 50):
        logger.info(f"  s.cafef trang {page}...")
        df_p = fetch_via_scafef(page=page, page_size=20)

        if df_p.empty:
            logger.info(f"  Trang {page} trống — dừng.")
            break

        all_dfs.append(df_p)
        if df_p["Date"].min() <= cutoff:
            break

        time.sleep(0.4)

    if not all_dfs:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs).drop_duplicates("Date")
    df_all = df_all[df_all["Date"] >= cutoff]
    return df_all.sort_values("Date").reset_index(drop=True)


# ── Nguồn 3: VCI/TCBS trực tiếp ─────────────────────────────
def fetch_via_vci(lookback_days: int = 730) -> pd.DataFrame:
    """
    Lấy dữ liệu VNIndex từ API VCI (nguồn của vnstock mới).

    Returns:
        pd.DataFrame hoặc rỗng.
    """
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=lookback_days)).timestamp())

    url = "https://trading.vietcap.com.vn/api/chart/OHLCChart/history"
    params = {
        "symbol":     "VNINDEX",
        "resolution": "D",
        "from":       start_ts,
        "to":         end_ts,
    }
    vci_headers = {**HEADERS, "Referer": "https://trading.vietcap.com.vn/"}

    try:
        resp = requests.get(url, params=params, headers=vci_headers, timeout=12)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        ts   = data.get("t", [])
        if not ts:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Date":   pd.to_datetime(ts, unit="s"),
            "Open":   data.get("o", []),
            "High":   data.get("h", []),
            "Low":    data.get("l", []),
            "Close":  data.get("c", []),
            "Volume": data.get("v", []),
        })
        # Điều chỉnh timezone UTC → GMT+7
        df["Date"] = df["Date"] + timedelta(hours=7)
        df["Date"] = df["Date"].dt.normalize()
        df = df[df["Close"] > 0].sort_values("Date").reset_index(drop=True)
        logger.info(f"VCI API: {len(df)} phiên")
        return df

    except Exception as e:
        logger.warning(f"VCI API: {e}")
        return pd.DataFrame()


# ── Nguồn 4: TCBS ────────────────────────────────────────────
def fetch_via_tcbs(lookback_days: int = 730) -> pd.DataFrame:
    """
    Lấy dữ liệu VNIndex từ TCBS API.

    Returns:
        pd.DataFrame hoặc rỗng.
    """
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_days)
    url      = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params   = {
        "ticker":     "VNINDEX",
        "type":       "index",
        "resolution": "D",
        "from":       int(start_dt.timestamp()),
        "to":         int(end_dt.timestamp()),
    }
    tcbs_headers = {**HEADERS, "Referer": "https://tcinvest.tcbs.com.vn/"}

    try:
        resp = requests.get(url, params=params, headers=tcbs_headers, timeout=12)
        if resp.status_code != 200:
            return pd.DataFrame()

        items = resp.json().get("data", [])
        if not items:
            return pd.DataFrame()

        rows = []
        for d in items:
            try:
                rows.append({
                    "Date":   pd.to_datetime(d.get("tradingDate", "")[:10]),
                    "Open":   float(d.get("open",   0) or 0),
                    "High":   float(d.get("high",   0) or 0),
                    "Low":    float(d.get("low",    0) or 0),
                    "Close":  float(d.get("close",  0) or 0),
                    "Volume": float(d.get("volume", 0) or 0),
                })
            except Exception:
                continue

        df = pd.DataFrame(rows)
        df = df[df["Close"] > 0].sort_values("Date").reset_index(drop=True)
        logger.info(f"TCBS API: {len(df)} phiên")
        return df

    except Exception as e:
        logger.warning(f"TCBS API: {e}")
        return pd.DataFrame()


# ── Unified Fetcher ───────────────────────────────────────────
def fetch_vnindex_all(lookback_days: int = 730) -> pd.DataFrame:
    """
    Thử tuần tự các nguồn cho đến khi lấy được dữ liệu.

    Returns:
        pd.DataFrame đầy đủ hoặc rỗng nếu tất cả thất bại.
    """
    sources = [
        ("vnstock",  lambda: fetch_via_vnstock(lookback_days)),
        ("VCI API",  lambda: fetch_via_vci(lookback_days)),
        ("TCBS API", lambda: fetch_via_tcbs(lookback_days)),
        ("s.cafef",  lambda: fetch_scafef_all(lookback_days)),
    ]

    for name, fn in sources:
        logger.info(f"Thử nguồn: {name}...")
        try:
            df = fn()
            if df is not None and not df.empty and len(df) >= 10:
                logger.info(f"Thành công từ {name}: {len(df)} phiên "
                            f"({df['Date'].min().strftime('%d/%m/%Y')} → "
                            f"{df['Date'].max().strftime('%d/%m/%Y')})")
                return df
            else:
                logger.warning(f"{name}: Trả về ít hơn 10 phiên.")
        except Exception as e:
            logger.warning(f"{name}: {e}")

    logger.error("Tất cả nguồn đều thất bại!")
    return pd.DataFrame()


# ── Import vào MSSQL ─────────────────────────────────────────
def get_existing_dates(conn) -> set:
    """Lấy ngày đã có trong DB. Returns set."""
    try:
        df = pd.read_sql(
            f"SELECT CONVERT(date,[time]) AS dt FROM dbo.daily_prices "
            f"WHERE [ticker]='{TICKER_NAME}'", conn
        )
        return set(pd.to_datetime(df["dt"]).dt.date)
    except Exception:
        return set()


def import_to_db(df: pd.DataFrame) -> dict:
    """
    Insert dữ liệu VNIndex vào DB, bỏ qua ngày đã có.

    Returns:
        dict: {inserted, skipped, errors}
    """
    if df.empty:
        return {"inserted": 0, "skipped": 0, "errors": 0}

    conn     = get_connection()
    cursor   = conn.cursor()
    existing = get_existing_dates(conn)
    logger.info(f"DB đã có {len(existing)} phiên VNINDEX")

    inserted = skipped = errors = 0

    for _, row in df.iterrows():
        date_obj = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
        if date_obj in existing:
            skipped += 1
            continue
        try:
            cursor.execute(
                "INSERT INTO dbo.daily_prices "
                "([ticker],[time],[open],[high],[low],[close],[volume]) "
                "VALUES (?,?,?,?,?,?,?)",
                TICKER_NAME,
                date_obj,
                float(row.get("Open",  0) or 0),
                float(row.get("High",  0) or 0),
                float(row.get("Low",   0) or 0),
                float(row.get("Close", 0) or 0),
                int(row.get("Volume",  0) or 0),
            )
            inserted += 1
            if inserted % 50 == 0:
                conn.commit()
                logger.info(f"  Đã insert {inserted} phiên...")
        except Exception as e:
            if "duplicate" in str(e).lower() or "UQ_" in str(e):
                skipped += 1
            else:
                logger.error(f"  Insert {date_obj}: {e}")
                errors += 1

    conn.commit()
    cursor.close()
    conn.close()
    return {"inserted": inserted, "skipped": skipped, "errors": errors}


def verify_import() -> pd.DataFrame:
    """Kiểm tra 5 dòng gần nhất trong DB."""
    conn = get_connection()
    df   = pd.read_sql(
        f"SELECT TOP 5 [time] AS Date, [close] AS Close "
        f"FROM dbo.daily_prices WHERE [ticker]='{TICKER_NAME}' "
        f"ORDER BY [time] DESC",
        conn,
    )
    conn.close()
    return df


# ── Scheduler ────────────────────────────────────────────────
def schedule_daily_update():
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("Cần: pip install apscheduler")
        return

    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")

    def job():
        df = fetch_vnindex_all(lookback_days=5)
        if not df.empty:
            result = import_to_db(df)
            logger.info(f"Daily update: {result}")

    scheduler.add_job(job, CronTrigger(hour=16, minute=30,
                      day_of_week="mon-fri", timezone="Asia/Ho_Chi_Minh"))
    logger.info("Lập lịch 16:30 T2-T6")
    scheduler.start()


# ── Main ─────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    if "--schedule" in args:
        schedule_daily_update()
        return

    lookback = 730
    for a in args:
        if a.isdigit():
            lookback = int(a)
            break

    logger.info(f"=== IMPORT VNINDEX — {lookback} NGÀY ===")

    df = fetch_vnindex_all(lookback_days=lookback)
    if df.empty:
        logger.error("Không lấy được dữ liệu từ bất kỳ nguồn nào.")
        logger.info("Gợi ý: pip install vnstock")
        return

    logger.info(f"\nMẫu dữ liệu:\n{df.tail(3).to_string()}")

    result = import_to_db(df)
    logger.info(f"\nKết quả: inserted={result['inserted']} "
                f"skipped={result['skipped']} errors={result['errors']}")

    df_verify = verify_import()
    if not df_verify.empty:
        logger.info(f"\nVerify DB:\n{df_verify.to_string()}")
        logger.info("\nHoàn thành! Chạy lại app.py để xem Market Context.")
    else:
        logger.error("Không tìm thấy dữ liệu trong DB sau import!")


if __name__ == "__main__":
    main()