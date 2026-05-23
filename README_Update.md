# Bull & Bear Trap Identification System

Hệ thống phát hiện bẫy giá **Bull Trap** và **Bear Trap** cho thị trường chứng khoán Việt Nam (VN30) sử dụng Machine Learning Ensemble, phân tích kỹ thuật nâng cao và dữ liệu realtime.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu hình Database](#cấu-hình-database)
- [Cách sử dụng](#cách-sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Chi tiết kỹ thuật](#chi-tiết-kỹ-thuật)
- [Giao diện Dashboard](#giao-diện-dashboard)

---

## Tổng quan

**Bull Trap** là hiện tượng giá cổ phiếu phá vỡ lên trên đường MA20 (tín hiệu tăng giả) nhưng sau đó quay đầu giảm trong vòng T+5 phiên — khiến nhà đầu tư mua theo bị kẹt hàng.

**Bear Trap** là hiện tượng ngược lại — giá phá xuống dưới MA20 (tín hiệu giảm giả) nhưng sau đó hồi phục mạnh — khiến nhà đầu tư bán khống bị kẹt.

Hệ thống này dùng **Ensemble ML** (XGBoost + RandomForest + LightGBM) kết hợp với **Feature Engineering** chuyên sâu để phát hiện hai loại bẫy giá này trên toàn bộ rổ VN30.

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────┐
│                    app.py (Controller)               │
│         Điều phối luồng dữ liệu — 6 Tabs            │
└──────┬──────────┬──────────┬────────────┬────────────┘
       │          │          │            │
       ▼          ▼          ▼            ▼
  ml_pipeline  analytics  custom_ui   realtime
  (Model)      (Logic)    (View)      (RT Data)
       │
       ▼
  Microsoft SQL Server
  (ProjectADY_StockDB)
  └── dbo.daily_prices
```

| File | Vai trò | Không chỉnh sửa |
|------|---------|-----------------|
| `ml_pipeline.py` | Train model, Feature Engineering, Labeling | Logic cốt lõi |
| `analytics.py` | Scanner, VPVR, R/R, MTF, Performance Tracker | — |
| `custom_ui.py` | Toàn bộ CSS, Components, Charts | — |
| `app.py` | Controller — gọi các module trên | — |
| `realtime.py` | Giá realtime SSI/Cafef/TCBS | — |
| `pdf_generator.py` | Xuất báo cáo PDF tự động | — |
| `import_vnindex.py` | Crawl & import dữ liệu VNIndex | — |

---

## Tính năng

### Model & Dự đoán
- **Bull Trap Detection** — phát hiện phá đỉnh giả lên MA20
- **Bear Trap Detection** — phát hiện phá đáy giả xuống MA20
- **Ensemble Prediction** — XGBoost (40%) + RandomForest (30%) + LightGBM (30%)
- **SMOTE Oversampling** — xử lý mất cân bằng nhãn
- **Auto-Retraining** — retrain mỗi tuần, chỉ deploy nếu AUC tốt hơn
- **Model Versioning** — lưu lịch sử tất cả phiên bản model

### Feature Engineering
| Feature | Mô tả |
|---------|-------|
| RSI(14) | Relative Strength Index |
| RSI Weekly | RSI khung tuần (Multi-Timeframe) |
| MA20 | Moving Average 20 ngày |
| Volume Ratio | Khối lượng / Volume MA20 |
| Upper Shadow Ratio | Râu nến trên / Thân nến |
| Lower Shadow Ratio | Râu nến dưới / Thân nến |
| Body Range Ratio | Thân nến / Dải giá |
| Is Doji | Nến Doji (thân < 10% dải) |
| OBV | On-Balance Volume |
| OBV Divergence | Phân kỳ OBV vs Giá |
| Is Shooting Star | Pattern nến Shooting Star |
| Is Bearish Engulfing | Pattern nến Bearish Engulfing |
| Is Evening Star | Pattern nến Evening Star |

### Dashboard (6 Tabs)

**Tab 1 — Biểu đồ kỹ thuật**
- Candlestick + MA20
- S/R động (Dynamic Support/Resistance)
- Fibonacci Retracement (tự động theo khung hiển thị)
- Volume Profile VPVR (POC / VAH / VAL / HVN / LVN)
- OBV + OBV MA20 + điểm phân kỳ
- Pattern Recognition markers
- Risk/Reward Calculator (SL/TP tự động)
- Price Alert (cảnh báo khi chạm ngưỡng)

**Tab 2 — Phân tích & Khuyến nghị**
- Market Context (phân tích VNIndex: Uptrend/Downtrend/Sideway)
- Multi-Timeframe Confirmation (Weekly RSI điều chỉnh confidence)
- Khuyến nghị tự động (Mua / Bán / Nắm giữ / Theo dõi)
- Storytelling — phân tích dạng văn bản
- Foreign Flow Analysis (dòng tiền khối ngoại nếu có dữ liệu)
- Telegram Alert (gửi cảnh báo tự động)

**Tab 3 — Scanner VN30**
- Bảng giá realtime toàn bộ VN30
- Quét AI đồng loạt 30+ mã, xếp hạng theo Bull Trap Prob
- Highlight TOP mã nguy hiểm nhất

**Tab 4 — Backtest**
- So sánh AI Strategy vs Buy & Hold
- Equity curve trực quan
- Chỉ số: Alpha Return, Win Rate, số giao dịch, lãi/lỗ TB

**Tab 5 — Trade Demo**
- Vốn ảo 100 tỷ VNĐ
- Đặt lệnh Mua/Bán (bội số 100 cổ phiếu)
- Giá vốn bình quân tự động
- Biểu đồ tỷ trọng danh mục (donut chart, NAV realtime)
- Lịch sử giao dịch (tô màu xanh/đỏ)
- Dữ liệu tồn tại qua refresh (lưu file JSON)

**Tab 6 — Model Performance**
- Ghi log mỗi lần predict
- Tự động verify sau 5 phiên với giá thực tế
- Rolling Accuracy / Precision / Recall / F1 (20 dự đoán gần nhất)
- Confusion Matrix
- Lịch sử dự đoán đã verify

### Realtime Data
- **SSI FastConnect API** — nguồn chính thức (cần Consumer ID/Secret)
- **Cafef.vn** — fallback miễn phí
- **MSSQL DB** — fallback cuối (giá EOD)
- Background thread tự cập nhật mỗi 60 giây trong giờ GD
- Cache thông minh (memory + file JSON)
- Nhận biết phiên: ATO / Continuous / ATC / Break / Closed

---

## Yêu cầu hệ thống

- Python 3.9+
- Microsoft SQL Server (có Windows Authentication)
- ODBC Driver 17 for SQL Server
- Kết nối internet (để crawl VNIndex và realtime data)

---

## Cài đặt

```bash
# 1. Clone hoặc copy project vào thư mục
cd "Bull and Bear Trap Identification System"

# 2. Cài thư viện
pip install pyodbc sqlalchemy pandas numpy scikit-learn
pip install xgboost lightgbm imbalanced-learn
pip install streamlit plotly matplotlib
pip install jinja2 pdfkit requests vnstock
pip install apscheduler

# 3. (Tuỳ chọn) Cài weasyprint thay pdfkit
pip install weasyprint
```

---

## Cấu hình Database

Sửa các biến sau trong **tất cả file** nếu thông tin server khác:

```python
DB_SERVER   = "NGUYEN_MINH"        # Tên SQL Server instance
DB_DATABASE = "ProjectADY_StockDB" # Tên database
DB_DRIVER   = "ODBC Driver 17 for SQL Server"
```

**Schema bảng dữ liệu:**

```sql
-- Bảng giá cổ phiếu (đã có sẵn)
dbo.daily_prices (
    ticker    NVARCHAR(20),   -- Mã CK: 'VNM', 'VNINDEX'...
    time      DATE,           -- Ngày giao dịch
    open      FLOAT,
    high      FLOAT,
    low       FLOAT,
    close     FLOAT,
    volume    BIGINT
)
```

---

## Cách sử dụng

### Bước 1 — Import dữ liệu VNIndex (chạy 1 lần)

```bash
python import_vnindex.py          # Import 2 năm lịch sử
python import_vnindex.py 365      # Hoặc chỉ 1 năm
python import_vnindex.py --schedule  # Lập lịch cập nhật 16:30 hàng ngày
```

### Bước 2 — Train Model

```bash
# Train toàn bộ VN30 (Bull + Bear Trap, Ensemble)
python ml_pipeline.py

# Train 1 mã cụ thể
python ml_pipeline.py VCB

# Lập lịch retrain hàng tuần (Chủ nhật 23h)
python -c "from ml_pipeline import schedule_weekly_retrain; schedule_weekly_retrain()"
```

### Bước 3 — Chạy Dashboard

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

### Bước 4 — Xuất báo cáo PDF (tuỳ chọn)

```bash
python pdf_generator.py --once     # Xuất PDF ngay lập tức
python pdf_generator.py            # Lập lịch 16:15 hàng ngày
```

---

## Cấu trúc thư mục

```
Bull and Bear Trap Identification System/
│
├── app.py                  # Streamlit Dashboard (Controller)
├── ml_pipeline.py          # Training Pipeline (Model)
├── analytics.py            # Phân tích nâng cao (Logic)
├── custom_ui.py            # UI Components (View)
├── realtime.py             # Realtime Data Module
├── pdf_generator.py        # Báo cáo PDF tự động
├── import_vnindex.py       # Crawl & import VNIndex
│
├── models/                 # Model .pkl đã train
│   ├── VNM_bull_trap.pkl
│   ├── VNM_bear_trap.pkl
│   ├── VNM_bull_meta.json  # Metadata AUC, trained_at
│   └── versions/           # Archive các phiên bản model cũ
│
├── reports/                # Báo cáo PDF xuất ra
│   └── charts/             # Biểu đồ PNG đính kèm PDF
│
├── performance/            # Model Performance Tracker
│   └── prediction_log.json # Log dự đoán để verify sau 5 phiên
│
├── realtime_cache.json     # Cache giá realtime
├── trade_demo_state.json   # Trạng thái Trade Demo (persist qua refresh)
├── training.log            # Log quá trình training
├── vnindex_import.log      # Log import VNIndex
└── README.md
```

---

## Chi tiết kỹ thuật

### Labeling Strategy

**Bull Trap (nhãn 1):**
```
Điều kiện Breakout:
  Close[t] > MA20[t]
  AND Close[t-1] < MA20[t-1]
  AND Close[t-2] < MA20[t-2]  ← cửa sổ 3 ngày

Bull Trap = Breakout AND Close[t+5] < Close[t] * 0.99
Breakout thật = Breakout AND Close[t+5] >= Close[t] * 0.99
```

**Bear Trap (nhãn 1) — đối xứng hoàn toàn:**
```
Breakdown: Close cắt xuống dưới MA20 sau ≥2 ngày trên MA20
Bear Trap = Breakdown AND Close[t+5] > Close[t] * 1.01
```

### Ensemble & Imbalance Handling

```
1. SMOTE Oversampling  → cân bằng số mẫu trap vs thật
2. XGBoost             → scale_pos_weight + min_child_weight
3. RandomForest        → class_weight='balanced_subsample'
4. LightGBM            → is_unbalance=True
5. Weighted Average    → XGB×0.4 + RF×0.3 + LGB×0.3
6. Ngưỡng quyết định  → 0.35 (thấp hơn 0.5 để tăng Recall)
```

### Multi-Timeframe Confidence

```
Weekly RSI > 70 + Bull Trap daily  → Tăng confidence ×1.15
Weekly RSI < 50 + Uptrend weekly   → Giảm confidence ×0.85
Weekly RSI > 65                    → Tăng nhẹ ×1.08
```

### Risk/Reward Calculator

```
Stop Loss  = min(MA20 × 0.995, Support gần nhất dưới Entry)
Take Profit = Resistance gần nhất phía trên Entry
R:R Ratio  = (TP - Entry) / (Entry - SL)
Position   = (1% vốn) / (Entry - SL), làm tròn bội số 100
```

---

## Giao diện Dashboard

**Sidebar:** Nhập mã cổ phiếu (tự động upper-case), chọn số ngày lịch sử, bật/tắt Fibonacci & S/R, cấu hình Telegram Alert, nhập SSI API token.

**Realtime Ticker:** Badge giá lớn hiển thị giá hiện tại, % thay đổi, volume, nguồn dữ liệu và phiên giao dịch (ATO/Continuous/ATC).

**4 Metric Cards:**
- Bull Trap Prob (adjusted theo MTF)
- Bear Trap Prob
- RSI(14) / Weekly RSI
- OBV Divergence / Volume Ratio

---

## Lưu ý quan trọng

> **Hệ thống này mang tính tham khảo kỹ thuật, không phải khuyến nghị đầu tư.**
> Nhà đầu tư cần kết hợp nhiều phương pháp phân tích và tự chịu trách nhiệm về quyết định của mình.

**Nguyên tắc code:**
- Tuyệt đối không dùng `print()` trong hàm xử lý dữ liệu
- Mọi hàm xử lý phải `return` giá trị cụ thể
- Logic model (`compute_features`, `predict_bull_trap`...) không được chỉnh sửa
- UI/UX tách biệt hoàn toàn khỏi Logic trong `custom_ui.py`

---

*Bull & Bear Trap Identification System — VN30 | Powered by XGBoost + RandomForest + LightGBM*
