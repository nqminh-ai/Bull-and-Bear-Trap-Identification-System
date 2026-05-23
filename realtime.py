"""
realtime.py
===========
Module cập nhật giá realtime cho Bull & Bear Trap Detector.

Nguồn dữ liệu (ưu tiên theo thứ tự):
  1. SSI FastConnect API  — chính thức, cần token
  2. Crawl cafef.vn       — fallback không cần token
  3. DB MSSQL             — fallback cuối (giá cuối ngày)

Tần suất: mỗi 60 giây trong giờ giao dịch (9:00–15:00 T2-T6).

NGUYÊN TẮC: Không dùng print(). Mọi hàm phải return.
"""

import os
import json
import time
import logging
import warnings
import threading
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import pyodbc
import streamlit as st

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ── Cấu hình ─────────────────────────────────────────────────
DB_SERVER   = "NGUYEN_MINH"
DB_DATABASE = "ProjectADY_StockDB"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"

CACHE_FILE       = "realtime_cache.json"   # Cache giá realtime ra file
UPDATE_INTERVAL  = 60                       # Giây giữa 2 lần update
MARKET_OPEN      = (9,  0)                  # 9:00
MARKET_CLOSE     = (15, 0)                  # 15:00

# SSI FastConnect endpoints
SSI_AUTH_URL  = "https://fc-data.ssi.com.vn/api/v2/Market/AccessToken"
SSI_QUOTE_URL = "https://fc-data.ssi.com.vn/api/v2/Market/Securities"
SSI_PRICE_URL = "https://fc-data.ssi.com.vn/api/v2/Market/MarketPrice"

# Cafef endpoints (không cần token)
CAFEF_URL = "https://cafef.vn/Ajax/AjaxGetDataFeed.ashx"


# ============================================================
# 1. KIỂM TRA GIỜ GIAO DỊCH
# ============================================================
def is_market_open() -> bool:
    """
    Kiểm tra có đang trong giờ giao dịch không (9:00–15:00 T2–T6).

    Returns:
        bool: True nếu thị trường đang mở.
    """
    now = datetime.now()
    if now.weekday() >= 5:   # Cuối tuần
        return False
    hour, minute = now.hour, now.minute
    open_mins  = MARKET_OPEN[0]  * 60 + MARKET_OPEN[1]
    close_mins = MARKET_CLOSE[0] * 60 + MARKET_CLOSE[1]
    cur_mins   = hour * 60 + minute
    return open_mins <= cur_mins <= close_mins


def get_session_phase() -> str:
    """
    Xác định phiên giao dịch hiện tại.

    Returns:
        str: 'ATO' | 'Continuous' | 'ATC' | 'Closed' | 'Break'
    """
    now  = datetime.now()
    if now.weekday() >= 5:
        return "Closed"
    h, m = now.hour, now.minute
    mins = h * 60 + m
    if   mins < 9*60:            return "Closed"
    elif mins < 9*60+15:         return "ATO"
    elif mins < 11*60+30:        return "Continuous"
    elif mins < 13*60:           return "Break"
    elif mins < 14*60+45:        return "Continuous"
    elif mins < 15*60:           return "ATC"
    else:                        return "Closed"


# ============================================================
# 2. SSI FASTCONNECT API
# ============================================================
class SSIDataFeed:
    """
    SSI FastConnect REST API client.
    Tài liệu: https://github.com/SSIFCOpenAPI/ssi-fc-data-python-client
    """

    def __init__(self, consumer_id: str, consumer_secret: str):
        """
        Args:
            consumer_id: SSI Consumer ID.
            consumer_secret: SSI Consumer Secret.
        """
        self.consumer_id     = consumer_id
        self.consumer_secret = consumer_secret
        self._token          = None
        self._token_expires  = None

    def _get_token(self) -> Optional[str]:
        """
        Lấy hoặc refresh access token SSI.

        Returns:
            str token hoặc None nếu lỗi.
        """
        if (self._token and self._token_expires
                and datetime.now() < self._token_expires):
            return self._token

        try:
            resp = requests.post(
                SSI_AUTH_URL,
                json={"consumerID": self.consumer_id,
                      "consumerSecret": self.consumer_secret},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._token         = data.get("data", {}).get("accessToken")
                self._token_expires = datetime.now() + timedelta(hours=23)
                return self._token
        except Exception as e:
            logger.error(f"SSI auth error: {e}")
        return None

    def get_realtime_price(self, symbol: str) -> Optional[dict]:
        """
        Lấy giá realtime của 1 mã từ SSI.

        Args:
            symbol: Mã cổ phiếu (VD: 'HPG').

        Returns:
            dict: {symbol, price, open, high, low, volume,
                   change, change_pct, time} hoặc None.
        """
        token = self._get_token()
        if not token:
            return None

        try:
            headers = {"Authorization": f"Bearer {token}"}
            resp    = requests.get(
                SSI_PRICE_URL,
                params={"symbol": symbol, "market": "HOSE"},
                headers=headers,
                timeout=8,
            )
            if resp.status_code != 200:
                return None

            raw = resp.json().get("data", [])
            if not raw:
                return None

            d = raw[0] if isinstance(raw, list) else raw
            price  = float(d.get("lastPrice",   0) or 0) * 1000
            ref    = float(d.get("refPrice",    0) or 0) * 1000
            change = price - ref

            return {
                "symbol":     symbol,
                "price":      price,
                "open":       float(d.get("openPrice",  0) or 0) * 1000,
                "high":       float(d.get("highPrice",  0) or 0) * 1000,
                "low":        float(d.get("lowPrice",   0) or 0) * 1000,
                "volume":     int(d.get("totalVol",     0) or 0),
                "ref_price":  ref,
                "change":     round(change, 2),
                "change_pct": round(change / ref * 100, 2) if ref > 0 else 0,
                "time":       d.get("time", datetime.now().strftime("%H:%M:%S")),
                "source":     "SSI",
            }
        except Exception as e:
            logger.warning(f"SSI price error {symbol}: {e}")
            return None

    def get_batch_prices(self, symbols: list) -> dict:
        """
        Lấy giá realtime cho nhiều mã cùng lúc.

        Args:
            symbols: Danh sách mã.

        Returns:
            dict: {symbol: price_dict}
        """
        token = self._get_token()
        if not token:
            return {}

        results = {}
        # SSI hỗ trợ batch tối đa 20 mã/request
        batch_size = 20
        headers    = {"Authorization": f"Bearer {token}"}

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                resp = requests.get(
                    SSI_PRICE_URL,
                    params={"symbol": ",".join(batch), "market": "HOSE"},
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue

                raw_list = resp.json().get("data", [])
                for d in (raw_list if isinstance(raw_list, list) else [raw_list]):
                    sym   = d.get("symbol", "")
                    price = float(d.get("lastPrice", 0) or 0) * 1000
                    ref   = float(d.get("refPrice",  0) or 0) * 1000
                    if sym and price > 0:
                        results[sym] = {
                            "symbol":     sym,
                            "price":      price,
                            "open":       float(d.get("openPrice", 0) or 0) * 1000,
                            "high":       float(d.get("highPrice", 0) or 0) * 1000,
                            "low":        float(d.get("lowPrice",  0) or 0) * 1000,
                            "volume":     int(d.get("totalVol",    0) or 0),
                            "ref_price":  ref,
                            "change":     round(price - ref, 2),
                            "change_pct": round((price-ref)/ref*100, 2) if ref > 0 else 0,
                            "time":       d.get("time", ""),
                            "source":     "SSI",
                        }
            except Exception as e:
                logger.warning(f"SSI batch error: {e}")

        return results


# ============================================================
# 3. CAFEF FALLBACK CRAWLER
# ============================================================
def fetch_price_cafef(symbol: str) -> Optional[dict]:
    """
    Lấy giá từ cafef.vn (fallback, không cần token).
    Parse JSON từ endpoint Ajax của cafef.

    Args:
        symbol: Mã cổ phiếu.

    Returns:
        dict giá hoặc None nếu lỗi.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://cafef.vn/",
        }
        resp = requests.get(
            CAFEF_URL,
            params={"symbol": symbol, "type": "priceboard"},
            headers=headers,
            timeout=8,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        # Cấu trúc cafef: data.Data.Data[0]
        items = (data.get("Data", {}) or {}).get("Data", []) or []
        if not items:
            return None

        d     = items[0]
        price = float(d.get("Close",    0) or 0) * 1000
        ref   = float(d.get("RefPrice", 0) or 0) * 1000
        if price == 0:
            return None

        return {
            "symbol":     symbol,
            "price":      price,
            "open":       float(d.get("Open",      0) or 0) * 1000,
            "high":       float(d.get("High",      0) or 0) * 1000,
            "low":        float(d.get("Low",       0) or 0) * 1000,
            "volume":     int(d.get("Volume",      0) or 0),
            "ref_price":  ref,
            "change":     round(price - ref, 2),
            "change_pct": round((price - ref) / ref * 100, 2) if ref > 0 else 0,
            "time":       datetime.now().strftime("%H:%M:%S"),
            "source":     "Cafef",
        }
    except Exception as e:
        logger.warning(f"Cafef error {symbol}: {e}")
        return None


def fetch_batch_cafef(symbols: list) -> dict:
    """
    Lấy giá nhiều mã từ cafef (fallback).

    Returns:
        dict: {symbol: price_dict}
    """
    results = {}
    for sym in symbols:
        data = fetch_price_cafef(sym)
        if data:
            results[sym] = data
        time.sleep(0.1)   # Tránh rate limit
    return results


# ============================================================
# 4. DB FALLBACK (MSSQL)
# ============================================================
def fetch_price_from_db(symbol: str) -> Optional[dict]:
    """
    Lấy giá cuối ngày từ MSSQL làm fallback cuối cùng.

    Returns:
        dict giá hoặc None.
    """
    conn_str = (
        f"DRIVER={{{DB_DRIVER}}};SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};Trusted_Connection=yes;"
    )
    try:
        conn  = pyodbc.connect(conn_str)
        query = f"""
            SELECT TOP 2
                [time] AS Date, [open] AS Open, [high] AS High,
                [low]  AS Low,  [close] AS Close, [volume] AS Volume
            FROM dbo.daily_prices
            WHERE [ticker] = '{symbol}'
            ORDER BY [time] DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            return None

        latest = df.iloc[0]
        prev   = df.iloc[1] if len(df) >= 2 else latest
        price  = float(latest["Close"])
        ref    = float(prev["Close"])

        return {
            "symbol":     symbol,
            "price":      price,
            "open":       float(latest["Open"]),
            "high":       float(latest["High"]),
            "low":        float(latest["Low"]),
            "volume":     int(latest["Volume"]),
            "ref_price":  ref,
            "change":     round(price - ref, 2),
            "change_pct": round((price - ref) / ref * 100, 2) if ref > 0 else 0,
            "time":       str(latest["Date"])[:10],
            "source":     "DB (EOD)",
        }
    except Exception as e:
        logger.warning(f"DB fallback error {symbol}: {e}")
        return None


# ============================================================
# 5. REALTIME CACHE MANAGER
# ============================================================
class RealtimeCache:
    """
    Quản lý cache giá realtime trong memory + file JSON.
    Thread-safe với Lock.
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self._cache = {}           # {symbol: price_dict}
        self._last_update = {}     # {symbol: timestamp}
        self._load_from_file()

    def _load_from_file(self):
        """Load cache từ file JSON khi khởi động."""
        if not os.path.exists(CACHE_FILE):
            return
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self._cache = saved.get("prices", {})
        except Exception:
            pass

    def _save_to_file(self):
        """Lưu cache ra file JSON."""
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "prices":     self._cache,
                    "updated_at": datetime.now().isoformat(),
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def update(self, symbol: str, data: dict):
        """Cập nhật giá cho 1 mã. Thread-safe."""
        with self._lock:
            data["cached_at"] = datetime.now().isoformat()
            self._cache[symbol]       = data
            self._last_update[symbol] = datetime.now()
        self._save_to_file()

    def update_batch(self, prices: dict):
        """Cập nhật giá cho nhiều mã cùng lúc."""
        with self._lock:
            for sym, data in prices.items():
                data["cached_at"]         = datetime.now().isoformat()
                self._cache[sym]          = data
                self._last_update[sym]    = datetime.now()
        self._save_to_file()

    def get(self, symbol: str) -> Optional[dict]:
        """
        Lấy giá từ cache.

        Returns:
            dict hoặc None nếu không có.
        """
        with self._lock:
            return self._cache.get(symbol)

    def get_all(self) -> dict:
        """Lấy toàn bộ cache. Returns dict."""
        with self._lock:
            return dict(self._cache)

    def is_stale(self, symbol: str, max_age_sec: int = 120) -> bool:
        """
        Kiểm tra cache có hết hạn không.

        Returns:
            bool: True nếu stale hoặc chưa có.
        """
        with self._lock:
            last = self._last_update.get(symbol)
            if not last:
                return True
            return (datetime.now() - last).total_seconds() > max_age_sec

    def get_last_update_str(self, symbol: str) -> str:
        """Thời gian cập nhật cuối dạng string."""
        with self._lock:
            last = self._last_update.get(symbol)
            return last.strftime("%H:%M:%S") if last else "Chua cap nhat"


# Singleton cache instance
_cache = RealtimeCache()


# ============================================================
# 6. UNIFIED PRICE FETCHER
# ============================================================
def get_realtime_price(symbol: str,
                        ssi_client: Optional[SSIDataFeed] = None,
                        force_refresh: bool = False) -> dict:
    """
    Lấy giá realtime với fallback tự động:
    SSI → Cafef → DB.

    Args:
        symbol: Mã cổ phiếu.
        ssi_client: SSIDataFeed instance (None nếu không có token).
        force_refresh: Bỏ qua cache, lấy mới.

    Returns:
        dict giá với thêm trường 'source' và 'cached_at'.
    """
    # Trả cache nếu còn fresh
    if not force_refresh and not _cache.is_stale(symbol, max_age_sec=UPDATE_INTERVAL):
        cached = _cache.get(symbol)
        if cached:
            return cached

    data = None

    # Nguồn 1: SSI FastConnect
    if ssi_client and is_market_open():
        data = ssi_client.get_realtime_price(symbol)

    # Nguồn 2: Cafef fallback
    if data is None and is_market_open():
        data = fetch_price_cafef(symbol)

    # Nguồn 3: DB fallback
    if data is None:
        data = fetch_price_from_db(symbol)

    if data:
        _cache.update(symbol, data)
        return data

    # Trả cache cũ nếu có (dù stale)
    cached = _cache.get(symbol)
    if cached:
        cached["source"] += " (stale)"
        return cached

    return {"symbol": symbol, "price": 0, "change": 0,
            "change_pct": 0, "source": "N/A", "volume": 0}


def get_batch_realtime(symbols: list,
                        ssi_client: Optional[SSIDataFeed] = None) -> dict:
    """
    Lấy giá realtime cho nhiều mã, ưu tiên SSI batch.

    Returns:
        dict: {symbol: price_dict}
    """
    results = {}

    # Chỉ fetch các mã cần refresh
    stale = [s for s in symbols if _cache.is_stale(s, UPDATE_INTERVAL)]

    if stale:
        # SSI batch
        if ssi_client and is_market_open():
            ssi_data = ssi_client.get_batch_prices(stale)
            if ssi_data:
                _cache.update_batch(ssi_data)
                stale = [s for s in stale if s not in ssi_data]

        # Cafef fallback cho các mã còn lại
        if stale and is_market_open():
            cafef_data = fetch_batch_cafef(stale)
            if cafef_data:
                _cache.update_batch(cafef_data)
                stale = [s for s in stale if s not in cafef_data]

        # DB fallback
        for sym in stale:
            db_data = fetch_price_from_db(sym)
            if db_data:
                _cache.update(sym, db_data)

    # Lấy từ cache
    for sym in symbols:
        cached = _cache.get(sym)
        if cached:
            results[sym] = cached

    return results


# ============================================================
# 7. BACKGROUND AUTO-UPDATE THREAD
# ============================================================
class RealtimeUpdater:
    """
    Background thread tự động cập nhật giá mỗi UPDATE_INTERVAL giây.
    Chỉ chạy trong giờ giao dịch.
    """

    def __init__(self, symbols: list, ssi_client=None):
        self.symbols    = symbols
        self.ssi_client = ssi_client
        self._thread    = None
        self._stop_evt  = threading.Event()
        self._running   = False

    def start(self):
        """Khởi động background thread."""
        if self._running:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="RealtimeUpdater"
        )
        self._thread.start()
        self._running = True
        logger.info("RealtimeUpdater started.")

    def stop(self):
        """Dừng background thread."""
        self._stop_evt.set()
        self._running = False
        logger.info("RealtimeUpdater stopped.")

    def _run(self):
        """Vòng lặp chính của thread."""
        while not self._stop_evt.is_set():
            if is_market_open():
                try:
                    get_batch_realtime(self.symbols, self.ssi_client)
                except Exception as e:
                    logger.error(f"Auto-update error: {e}")
            # Nghỉ UPDATE_INTERVAL giây, kiểm tra stop mỗi giây
            for _ in range(UPDATE_INTERVAL):
                if self._stop_evt.is_set():
                    break
                time.sleep(1)

    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================
# 8. STREAMLIT UI COMPONENTS
# ============================================================
def render_realtime_ticker(symbol: str, price_data: dict):
    """
    Hiển thị ticker giá realtime dạng badge nhỏ gọn.

    Args:
        symbol: Mã cổ phiếu.
        price_data: Dict từ get_realtime_price().
    """
    price      = price_data.get("price",      0)
    change     = price_data.get("change",     0)
    change_pct = price_data.get("change_pct", 0)
    volume     = price_data.get("volume",     0)
    source     = price_data.get("source",     "N/A")
    updated_at = price_data.get("cached_at",  "")

    color  = "#16a34a" if change >= 0 else "#dc2626"
    bg     = "#f0fdf4" if change >= 0 else "#fef2f2"
    border = "#bbf7d0" if change >= 0 else "#fecaca"
    arrow  = "▲" if change >= 0 else "▼"

    updated_str = ""
    if updated_at:
        try:
            updated_str = datetime.fromisoformat(updated_at).strftime("%H:%M:%S")
        except Exception:
            updated_str = updated_at[:8]

    source_color = {"SSI": "#2563eb", "Cafef": "#7c3aed",
                    "DB (EOD)": "#94a3b8"}.get(source.split()[0], "#94a3b8")

    st.markdown(f"""
    <div style="background:{bg}; border:1px solid {border}; border-radius:10px;
                padding:14px 20px; display:flex; align-items:center;
                justify-content:space-between; margin-bottom:12px;">
        <div style="display:flex; align-items:center; gap:20px;">
            <div>
                <div style="font-size:11px; color:#64748b; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.8px;">
                    Gia hien tai — {symbol}
                </div>
                <div style="font-size:30px; font-weight:700; color:#0f172a;
                            line-height:1.2; margin-top:2px;">
                    {price:,.0f}
                </div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:20px; font-weight:700; color:{color};">
                    {arrow} {abs(change_pct):.2f}%
                </div>
                <div style="font-size:13px; color:{color};">
                    {'+' if change >= 0 else ''}{change:,.0f} VND
                </div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:11px; color:#94a3b8;">Volume</div>
                <div style="font-size:15px; font-weight:600; color:#0f172a;">
                    {volume/1e6:.1f}M
                </div>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="background:{source_color}15; color:{source_color};
                        font-size:11px; font-weight:600; padding:3px 10px;
                        border-radius:20px; border:1px solid {source_color}30;
                        margin-bottom:4px; display:inline-block;">
                {source}
            </div>
            <div style="font-size:11px; color:#94a3b8; margin-top:2px;">
                Cap nhat: {updated_str}
            </div>
            <div style="font-size:10px; color:#cbd5e1;">
                {get_session_phase()}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_realtime_scanner_row(prices: dict) -> pd.DataFrame:
    """
    Tạo DataFrame bảng giá realtime cho Scanner.

    Args:
        prices: Dict {symbol: price_data}.

    Returns:
        pd.DataFrame bảng giá đã format.
    """
    rows = []
    for sym, d in prices.items():
        pct    = d.get("change_pct", 0)
        change = d.get("change",     0)
        rows.append({
            "Ma":       sym,
            "Gia":      f"{d.get('price', 0):,.0f}",
            "+/-":      f"{'+' if change >= 0 else ''}{change:,.0f}",
            "%":        f"{'+' if pct >= 0 else ''}{pct:.2f}%",
            "Volume":   f"{d.get('volume', 0)/1e6:.1f}M",
            "Cao":      f"{d.get('high', 0):,.0f}",
            "Thap":     f"{d.get('low',  0):,.0f}",
            "Nguon":    d.get("source", "N/A"),
            "_pct":     pct,   # Cột ẩn để sort
        })

    df = pd.DataFrame(rows).sort_values("_pct", ascending=False).drop(columns=["_pct"])
    return df


def render_realtime_chart_overlay(df_hist: pd.DataFrame,
                                   price_data: dict) -> pd.DataFrame:
    """
    Thêm nến realtime vào DataFrame lịch sử để vẽ trên biểu đồ.
    Cập nhật nến cuối (nếu cùng ngày) hoặc thêm nến mới.

    Args:
        df_hist: DataFrame OHLCV lịch sử (index = Date).
        price_data: Giá realtime hiện tại.

    Returns:
        pd.DataFrame cập nhật với nến realtime.
    """
    if not price_data or price_data.get("price", 0) == 0:
        return df_hist

    today = pd.Timestamp(datetime.now().date())
    df    = df_hist.copy()

    new_row = pd.Series({
        "Open":   price_data.get("open",   df["Close"].iloc[-1]),
        "High":   price_data.get("high",   price_data["price"]),
        "Low":    price_data.get("low",    price_data["price"]),
        "Close":  price_data["price"],
        "Volume": price_data.get("volume", 0),
    }, name=today)

    if today in df.index:
        # Cập nhật nến hôm nay
        df.loc[today, "Close"]  = new_row["Close"]
        df.loc[today, "High"]   = max(df.loc[today, "High"],  new_row["High"])
        df.loc[today, "Low"]    = min(df.loc[today, "Low"],   new_row["Low"])
        df.loc[today, "Volume"] = new_row["Volume"]
    else:
        # Thêm nến mới
        df = pd.concat([df, new_row.to_frame().T])

    return df


def render_price_alert_config(symbol: str,
                               current_price: float,
                               prob: float):
    """
    Hiển thị cấu hình Price Alert — cảnh báo khi giá chạm ngưỡng.

    Args:
        symbol: Mã cổ phiếu.
        current_price: Giá hiện tại.
        prob: Xác suất Bull Trap.
    """
    with st.expander("Price Alert — Canh bao gia"):
        col1, col2, col3 = st.columns(3)

        with col1:
            alert_above = st.number_input(
                "Canh bao khi gia VUOT",
                value=float(round(current_price * 1.03)),
                step=100.0, format="%.0f",
                key=f"alert_above_{symbol}",
            )
        with col2:
            alert_below = st.number_input(
                "Canh bao khi gia DUOI",
                value=float(round(current_price * 0.97)),
                step=100.0, format="%.0f",
                key=f"alert_below_{symbol}",
            )
        with col3:
            alert_prob = st.slider(
                "Canh bao khi Bull Trap > (%)",
                30, 90, 60, 5,
                key=f"alert_prob_{symbol}",
            )

        # Lưu config vào session state
        alert_key = f"price_alerts_{symbol}"
        if st.button("Luu cai dat Alert", key=f"save_alert_{symbol}"):
            st.session_state[alert_key] = {
                "above":    alert_above,
                "below":    alert_below,
                "prob_thr": alert_prob,
            }
            st.success("Da luu! He thong se canh bao khi dieu kien duoc kich hoat.")

        # Kiểm tra trigger
        alerts   = st.session_state.get(alert_key, {})
        triggers = []
        if alerts.get("above") and current_price >= alerts["above"]:
            triggers.append(f"Gia {current_price:,.0f} da VUOT {alerts['above']:,.0f}")
        if alerts.get("below") and current_price <= alerts["below"]:
            triggers.append(f"Gia {current_price:,.0f} da XUONG DUOI {alerts['below']:,.0f}")
        if alerts.get("prob_thr") and prob * 100 >= alerts["prob_thr"]:
            triggers.append(f"Bull Trap {prob:.0%} vuot nguong {alerts['prob_thr']}%")

        for t in triggers:
            st.markdown(f"""<div class="banner danger">
                ALERT: {t}
            </div>""", unsafe_allow_html=True)
