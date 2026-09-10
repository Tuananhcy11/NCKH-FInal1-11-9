# -*- coding: utf-8 -*-
"""Sửa 3 notebook Colab của nhóm theo luồng SONG SONG.

    python src/sua_notebook_colab.py

Luồng (đầu vào duy nhất: tệp OHLCV vàng, khung bất kỳ):

                     OHLCV vàng
                    /           \\
    NB1 chỉ báo kỹ thuật      NB2 3 model AI              ← chạy SONG SONG, độc lập
    luật → −1 / 0 / 1         XGBoost, RF, Bi-LSTM → −1 / 0 / 1
    gold_price_technical_     gold_model_signals.csv
      signal.csv
                    \\           /
          NB3 backtest kiểu MT5 trên CÙNG giai đoạn → Profit, Sharpe, Max DD…

Ý nghĩa tín hiệu ở cả hai notebook: 1 = trend tăng, 0 = sideway, −1 = trend giảm.

Script đọc BẢN GỐC trong "Google colab/ban_goc/" và ghi bản đã sửa ra
"Google colab/". Chạy lại bao nhiêu lần cũng cho cùng kết quả. Các ô chỉ báo,
mô hình và vẽ biểu đồ của nhóm được giữ NGUYÊN VĂN; chỉ thay các chỗ nối.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
THU_MUC = GOC / "Google colab"
BAN_GOC = THU_MUC / "ban_goc"

NB1 = "XAY_DUNG_CAC_CHI_SO_KY_THUAT.ipynb"
NB2 = "3_model.ipynb"
NB3 = "nckh_ptt(muc3).ipynb"


# ════════════════════════════════════════════════════════ tiện ích ô
def md(chu: str) -> dict:
    d = chu.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [x + "\n" for x in d[:-1]] + [d[-1]]}


def code(ma: str) -> dict:
    d = ma.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [x + "\n" for x in d[:-1]] + [d[-1]]}


def giu(o: dict) -> dict:
    """Giữ nguyên văn một ô của nhóm, chỉ xóa output cũ."""
    o = copy.deepcopy(o)
    if o["cell_type"] == "code":
        o["outputs"], o["execution_count"] = [], None
    return o


def nguon(o: dict) -> str:
    return "".join(o["source"])


def doi_chu(cells: list, cap: list) -> list:
    """Thay chữ trong các ô (đã giu), mỗi cặp (cũ, mới) phải khớp ít nhất một lần.
    Chỉ đổi chú thích và tên nhãn, không đổi logic của nhóm."""
    dem = {a: 0 for a, _ in cap}
    ra = []
    for o in cells:
        s = nguon(o)
        for a, b in cap:
            dem[a] += s.count(a)
            s = s.replace(a, b)
        o = copy.deepcopy(o)
        d = s.split("\n")
        o["source"] = [x + "\n" for x in d[:-1]] + [d[-1]]
        ra.append(o)
    thieu = [a for a, n in dem.items() if n == 0]
    assert not thieu, "Bản gốc đã đổi, không tìm thấy: %r" % thieu
    return ra


def doc_goc(ten: str) -> dict:
    return json.loads((BAN_GOC / ten).read_text(encoding="utf-8"))


SO_DO = r'''
```
                     OHLCV vàng (khung bất kỳ)
                    /                          \
   NB1 chỉ báo kỹ thuật                    NB2 3 model AI
   luật → dự báo xu hướng                  XGBoost, RF, Bi-LSTM → dự báo xu hướng
                    \                          /
        NB3 chiến lược: xu hướng → lệnh BUY / SELL / FLAT
            → backtest trên CÙNG giai đoạn → Profit, Sharpe, Max DD…
```

NB1 và NB2 chỉ **dự báo xu hướng**: **1 = uptrend (trend tăng) · 0 = sideway ·
−1 = downtrend (trend giảm)** — chưa phải lệnh giao dịch. **NB3** mới áp chiến lược để
đổi xu hướng thành lệnh **BUY / SELL / FLAT** rồi backtest.

NB1 và NB2 **độc lập với nhau**: mỗi notebook tự đọc tệp OHLCV, chạy trước hay sau
đều được. NB3 chạy sau cùng, khi đã có tệp kết quả của cả hai.
'''


# ═══════════════════════════════════ các ô dùng chung cho NB1 và NB2
# Mỗi notebook Colab chạy trên một máy ảo riêng, nên tệp NB1/NB2 tạo ra KHÔNG tự
# xuất hiện ở NB3. Ba notebook trao đổi qua cùng một thư mục Google Drive.
LUU_TRU = r'''
# ── Nơi trao đổi tệp giữa 3 notebook ─────────────────────────────
# Mỗi notebook Colab chạy trên một máy ảo riêng: tệp NB1/NB2 tạo ra KHÔNG tự có
# mặt ở NB3. Vì vậy cả 3 notebook cùng đọc/ghi vào MỘT thư mục trên Google Drive.
THU_MUC_DRIVE = '/content/drive/MyDrive/Data_NghienCuu'

try:
    from google.colab import files, drive
    TREN_COLAB = True
except ImportError:
    files = drive = None
    TREN_COLAB = False

THU_MUC = '.'
if TREN_COLAB:
    try:
        drive.mount('/content/drive')
        THU_MUC = THU_MUC_DRIVE
    except Exception as loi:
        print('Không gắn được Google Drive (%s).' % loi)
        print('→ Dùng /content: nhớ tải tệp kết quả về và tải lên ở NB3.')
        THU_MUC = '/content'
os.makedirs(THU_MUC, exist_ok=True)
print('Thư mục trao đổi dữ liệu:', os.path.abspath(THU_MUC))


def tim_tep(ten):
    """Tìm tệp đầu vào: thư mục trao đổi → thư mục hiện tại → tải lên (Colab)."""
    for p in (os.path.join(THU_MUC, ten), ten):
        if os.path.exists(p):
            return p
    if TREN_COLAB:
        print('Chưa thấy %s trong %s — hãy tải tệp này lên:' % (ten, THU_MUC))
        up = files.upload()
        if up:
            return list(up.keys())[0]
    raise FileNotFoundError('Không tìm thấy %s. Hãy chạy notebook tạo ra tệp này trước.' % ten)


def luu_tep(bang, ten):
    p = os.path.join(THU_MUC, ten)
    bang.to_csv(p, index=False, encoding='utf-8-sig')
    print('Đã lưu: %s  (%d dòng × %d cột)' % (os.path.abspath(p), bang.shape[0], bang.shape[1]))
    return p
'''

NHAP_MD = r'''
Nhập bộ dữ liệu OHLCV

Để trống `DUONG_DAN_DU_LIEU` thì Colab hiện nút **Choose Files** để tải tệp lên.
Để khỏi tải cùng một tệp hai lần cho NB1 và NB2, có thể đặt tệp vào Drive rồi
điền đường dẫn, ví dụ `/content/drive/MyDrive/Data_NghienCuu/xau_h1.csv`.
Nhận `.csv`, `.txt` (tách bằng dấu phẩy, `;` hoặc tab), `.xlsx`, `.parquet`.
'''

NHAP = r'''
DUONG_DAN_DU_LIEU = ''

DUONG_DAN_DU_LIEU = DUONG_DAN_DU_LIEU or os.environ.get('NCKH_DU_LIEU', '')
if DUONG_DAN_DU_LIEU:
    file_name = DUONG_DAN_DU_LIEU
elif TREN_COLAB:
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
else:
    raise ValueError('Đang chạy ngoài Colab: hãy điền DUONG_DAN_DU_LIEU.')

print("Tệp dữ liệu:", file_name)
'''

DOC = r'''
ten_thuong = file_name.lower()
if ten_thuong.endswith(('.xlsx', '.xls')):
    df = pd.read_excel(file_name)
elif ten_thuong.endswith(('.parquet', '.pq')):
    df = pd.read_parquet(file_name)
else:
    df = pd.read_csv(file_name)
    # Tệp xuất từ MetaTrader thường tách cột bằng tab hoặc ';' → tự dò lại
    if df.shape[1] == 1:
        df = pd.read_csv(file_name, sep=None, engine='python')

print("Kích thước dữ liệu:", df.shape)
display(df.head())
'''

CHUAN_HOA_MD = r'''
Chuẩn hóa tên cột và cột thời gian (dùng được cho mọi khung)

Dữ liệu vàng từ các nguồn khác nhau đặt tên cột rất khác nhau. Ô dưới tự nhận
biết mà không cần sửa tay:

- **Tên cột thời gian:** `time`, `datetime`, `date`, `timestamp`, `Gmt time`,
  `<DATE>` + `<TIME>` tách rời (kiểu MetaTrader)…
- **Định dạng thời gian:** `2025-01-02 13:00`, `2025.01.02 13:00`, `02/01/2025`,
  số giây hoặc mili-giây Unix, `20250102`, có hoặc không có múi giờ.
- **Tên cột giá:** `Open/open/<OPEN>/o`, `Close/Adj Close/price`,
  `Volume/Tick Volume/tickvol`…
- **Định dạng số:** `1183.949`, `1183,949`, `1,183.949`, `1.183,949`.

Mọi thời điểm được đưa về **UTC, không kèm múi giờ**. Khung thời gian được suy
ra từ khoảng cách phổ biến nhất giữa hai nến liền nhau.

Ô này **giống hệt nhau ở NB1 và NB2**, nên hai notebook luôn đọc cùng một tệp ra
cùng một bảng dữ liệu.
'''

CHUAN_HOA = r'''
def _chuan_ten(c):
    return ' '.join(str(c).strip().lower().replace('<', ' ').replace('>', ' ').replace('_', ' ').split())

# Tên cột theo thứ tự ưu tiên
BI_DANH = {
    'time':   ['datetime', 'date time', 'timestamp', 'time', 'date', 'gmt time', 'local time',
               'time (utc)', 'datetime utc', 'open time', 'opentime', 'thoi gian', 'ngay'],
    'open':   ['open', 'o', 'open price', 'gia mo'],
    'high':   ['high', 'h', 'high price', 'gia cao'],
    'low':    ['low', 'l', 'low price', 'gia thap'],
    'close':  ['close', 'c', 'close price', 'adj close', 'price', 'last', 'gia dong'],
    'volume': ['volume', 'vol', 'tick volume', 'tickvol', 'real volume', 'khoi luong'],
}

def _tim_cot(cac_cot, loai):
    ten = {_chuan_ten(x): x for x in cac_cot}
    for ung_vien in BI_DANH[loai]:
        if ung_vien in ten:
            return ten[ung_vien]
    return None

def _giong_gio(s):
    """Cột chỉ chứa giờ dạng 13:00 hoặc 13:00:00 (cột <TIME> tách rời)."""
    v = s.dropna().astype(str).str.strip().head(200)
    return len(v) > 0 and v.str.fullmatch(r'\d{1,2}:\d{2}(:\d{2})?').mean() > 0.9

def _so(s):
    """Số thực. Chấp nhận 1183.949 · 1183,949 · 1,183.949 · 1.183,949."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str).str.strip().str.replace(' ', '', regex=False)
    mau = s.head(500)
    # Dấu nào đứng SAU CÙNG là dấu thập phân; dấu còn lại là phân cách hàng nghìn
    phay_la_thap_phan = (mau.str.rfind(',') > mau.str.rfind('.')).mean() > 0.5
    if phay_la_thap_phan:
        s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    else:
        s = s.str.replace(',', '', regex=False)
    return pd.to_numeric(s, errors='coerce')

def _doc_thoi_gian(s):
    """Đọc cột thời gian ở mọi định dạng thường gặp, trả về UTC không múi giờ."""
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors='coerce')
        m = v.dropna().abs().median()
        if 1e7 <= m < 1e8:                                   # dạng 20250102
            return pd.to_datetime(v.astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
        don_vi = 'ms' if m > 1e11 else 's'                   # Unix mili-giây hay giây
        return pd.to_datetime(v, unit=don_vi, errors='coerce', utc=True).dt.tz_localize(None)

    s = s.astype(str).str.strip()
    t = pd.to_datetime(s, errors='coerce', utc=True)
    if t.isna().mean() > 0.01:                               # định dạng lẫn lộn → đọc từng dòng
        t = pd.to_datetime(s, errors='coerce', utc=True, format='mixed')
    ung_vien = [t]
    if s.str.contains('/').mean() > 0.5:                     # 02/01/2025: ngày-trước hay tháng-trước?
        ung_vien.append(pd.to_datetime(s, errors='coerce', utc=True, format='mixed', dayfirst=True))
    # Dữ liệu giá luôn xếp theo thời gian: chọn cách đọc ít lỗi nhất và tăng dần nhiều nhất
    diem = lambda x: (x.notna().mean(), (x.diff().dt.total_seconds() > 0).mean())
    return max(ung_vien, key=diem).dt.tz_localize(None)

KHUNG_CHUAN = [(1, 'M1'), (5, 'M5'), (15, 'M15'), (30, 'M30'), (60, 'H1'),
               (240, 'H4'), (1440, 'D1'), (10080, 'W1'), (43200, 'MN')]

def nhan_dien_khung(t):
    phut = t.sort_values().diff().dt.total_seconds().div(60)
    buoc = phut[phut > 0].mode().iloc[0]
    return min(KHUNG_CHUAN, key=lambda k: abs(np.log(k[0] / buoc)))[1], buoc


# ── 1. Cột thời gian
cac_cot = list(df.columns)
ten_chuan = {_chuan_ten(x): x for x in cac_cot}
if 'date' in ten_chuan and 'time' in ten_chuan and _giong_gio(df[ten_chuan['time']]):
    cot_tg = '%s + %s' % (ten_chuan['date'], ten_chuan['time'])
    tho_tg = df[ten_chuan['date']].astype(str).str.strip() + ' ' + df[ten_chuan['time']].astype(str).str.strip()
else:
    cot_tg = _tim_cot(cac_cot, 'time')
    if cot_tg is None:
        raise ValueError('Không tìm thấy cột thời gian trong: %s' % cac_cot)
    tho_tg = df[cot_tg]

ra = pd.DataFrame({'Date': _doc_thoi_gian(tho_tg)})

# ── 2. Cột giá và khối lượng
anh_xa = {}
for loai, ten_moi in [('open', 'Open'), ('high', 'High'), ('low', 'Low'),
                      ('close', 'Close'), ('volume', 'Volume')]:
    cot = _tim_cot(cac_cot, loai)
    anh_xa[ten_moi] = cot
    ra[ten_moi] = _so(df[cot]).values if cot is not None else np.nan

if anh_xa['Close'] is None:
    raise ValueError('Không tìm thấy cột giá đóng cửa trong: %s' % cac_cot)
for ten_moi in ('Open', 'High', 'Low'):
    if anh_xa[ten_moi] is None:
        print('⚠ Thiếu cột %s → tạm dùng giá Close.' % ten_moi)
        ra[ten_moi] = ra['Close']
if anh_xa['Volume'] is None:
    ra['Volume'] = 1.0            # nhiều nguồn Forex không có khối lượng thật

df = ra
KHUNG, BUOC_PHUT = nhan_dien_khung(df['Date'].dropna())

print('Ánh xạ cột:')
print('  %-7s ← %s' % ('Date', cot_tg))
for k, v in anh_xa.items():
    print('  %-7s ← %s' % (k, v if v is not None else '(không có)'))
print('\nKhung thời gian nhận diện: %s  (bước phổ biến %.0f phút)' % (KHUNG, BUOC_PHUT))
print('Giai đoạn: %s → %s' % (df['Date'].min(), df['Date'].max()))
print('Dòng không đọc được thời gian: %d' % df['Date'].isna().sum())
'''

LAM_SACH = r'''
print("Trước xử lý:", df.shape)

# Bỏ dòng thiếu thời gian hoặc giá đóng cửa
df = df.dropna(subset=['Date', 'Close'])

# Giá phải lớn hơn 0
df = df[df['Close'] > 0]

# Bỏ nến vi phạm hình học: High < Low, hoặc Close nằm ngoài [Low, High]
hop_le = (df['High'] >= df['Low']) & (df['Close'] <= df['High']) & (df['Close'] >= df['Low'])
print("Nến vi phạm hình học OHLC:", int((~hop_le).sum()))
df = df[hop_le]

# Xóa dòng trùng thời gian, sắp xếp theo thời gian
df = (
    df
    .drop_duplicates(subset=['Date'], keep='last')
    .sort_values('Date')
    .reset_index(drop=True)
)

print("Sau xử lý:", df.shape)
if len(df) < 100:
    raise ValueError('Sau khi làm sạch chỉ còn %d dòng. Kiểm tra lại ánh xạ cột ở ô trên '
                     'và định dạng số của tệp (dấu thập phân, dấu phân cách hàng nghìn).' % len(df))
'''


# ══════════════════════════════════════════════════════════════ NB1
def nb1() -> dict:
    g = doc_goc(NB1)
    c = g["cells"]
    assert nguon(c[18]).strip().startswith("PHẦN B"), "Bản gốc NB1 đã đổi cấu trúc"
    assert "RSI(14)" in nguon(c[62]), "Bản gốc NB1 đã đổi cấu trúc"

    moi = [md(r'''
# NB1 — Chỉ số kỹ thuật → dự báo xu hướng (uptrend / sideway / downtrend)

Nhánh **kỹ thuật** của luồng nghiên cứu. Đầu vào duy nhất là tệp **OHLCV vàng**
ở khung bất kỳ (M1 … W1). Đầu ra `gold_price_technical_signal.csv` gồm các chỉ
báo và 4 dự báo xu hướng theo luật: MA, RSI, MACD và dự báo tổng hợp (≥ 2/3 luật
đồng ý). Cột `Trend` ghi dạng chữ UPTREND / SIDEWAY / DOWNTREND.
''' + SO_DO)]
    moi.append(md("Import thư viện và nơi lưu tệp"))
    moi.append(code(r'''
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
''' + LUU_TRU))
    moi += [md(NHAP_MD), code(NHAP), md("Đọc dữ liệu"), code(DOC)]
    moi += [giu(c[6]), giu(c[7])]                   # PHẦN A + in danh sách cột
    moi += [md(CHUAN_HOA_MD), code(CHUAN_HOA)]
    moi += [md("Loại bỏ dữ liệu lỗi và trùng"), code(LAM_SACH)]
    moi += [giu(c[16]), giu(c[17])]

    # PHẦN B → G của nhóm: chỉ báo, 3 tín hiệu, tổng hợp, thống kê, biểu đồ.
    # Giữ nguyên logic; chỉ đổi cách GỌI tên: NB1 dự báo XU HƯỚNG, chưa phải lệnh
    # BUY/SELL/FLAT — việc đổi xu hướng thành lệnh là của chiến lược ở NB3.
    moi += doi_chu([giu(x) for x in c[18:63]], [
        ("# BUY\n", "# Dự báo TREND TĂNG (1)\n"),
        ("# SELL\n", "# Dự báo TREND GIẢM (-1)\n"),
        ("Đếm số phiếu BUY và SELL", "Đếm số luật dự báo TREND TĂNG và TREND GIẢM"),
        ("# Có ít nhất 2 tín hiệu BUY", "# Có ít nhất 2 luật dự báo trend tăng → UPTREND"),
        ("# Có ít nhất 2 tín hiệu SELL", "# Có ít nhất 2 luật dự báo trend giảm → DOWNTREND"),
        ("Buy_Count", "Up_Count"),
        ("Sell_Count", "Down_Count"),
        ("PHẦN E — ĐỔI RA BUY / FLAT / SELL ĐỂ DỄ ĐỌC",
         "PHẦN E — ĐỔI RA UPTREND / SIDEWAY / DOWNTREND ĐỂ DỄ ĐỌC\n\n"
         "Đây là **dự báo xu hướng**, chưa phải lệnh. NB3 mới đổi xu hướng thành "
         "lệnh BUY / SELL / FLAT theo chiến lược."),
        ("-1: 'SELL',", "-1: 'DOWNTREND',"),
        ("0: 'FLAT',", "0: 'SIDEWAY',"),
        ("1: 'BUY'\n", "1: 'UPTREND'\n"),
        ("'Action'", "'Trend'"),
        ("Tỷ lệ BUY / FLAT / SELL", "Tỷ lệ UPTREND / SIDEWAY / DOWNTREND"),
        ("Số lượng tín hiệu:", "Số nến theo từng xu hướng dự báo:"),
        ("Tỷ lệ tín hiệu (%):", "Tỷ lệ xu hướng dự báo (%):"),
    ])

    moi.append(md(r'''
PHẦN H — XUẤT FILE KẾT QUẢ

Đổi tên cột thời gian và giá về chữ thường (`time, open, high, low, close,
volume`) để NB3 đọc thống nhất với tệp của NB2.
'''))
    moi.append(code(r'''
ket_qua = df.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high',
                             'Low': 'low', 'Close': 'close', 'Volume': 'volume'})

output_file = 'gold_price_technical_signal.csv'
duong_dan_ra = luu_tep(ket_qua, output_file)
print('Khung %s | %s → %s' % (KHUNG, ket_qua['time'].min(), ket_qua['time'].max()))
'''))
    moi.append(code(r'''
# Tải tệp về máy (không bắt buộc nếu đã lưu trên Google Drive)
if TREN_COLAB and not duong_dan_ra.startswith('/content/drive'):
    files.download(duong_dan_ra)
'''))
    g["cells"] = moi
    return g


# ══════════════════════════════════════════════════════════════ NB2
def nb2() -> dict:
    g = doc_goc(NB2)
    c = g["cells"]
    assert "def apply_dual_threshold" in nguon(c[3]), "Bản gốc NB2 đã đổi cấu trúc"
    assert "def run_xgboost" in nguon(c[5]) and "def run_random_forest" in nguon(c[7])
    assert "class BiLSTMModel" in nguon(c[9])

    # ── Sửa lỗi Bi-LSTM: CrossEntropyLoss đã tự áp softmax bên trong. Mô hình gốc
    # áp softmax thêm một lần trong forward → softmax hai lần, gradient bị nén và
    # mô hình học rất kém. Sửa: forward trả logit, chỉ softmax khi dự báo.
    s = nguon(c[9])
    sua = [
        ("        return self.softmax(self.fc(self.dropout(lstm_out[:, -1, :])))",
         "        # Trả về LOGIT: CrossEntropyLoss đã tự áp softmax bên trong,\n"
         "        # áp thêm ở đây sẽ thành softmax hai lần và mô hình học rất kém.\n"
         "        return self.fc(self.dropout(lstm_out[:, -1, :]))"),
        ("        probs = model(torch.tensor(X_test_seq, dtype=torch.float32).to(device)).cpu().numpy()",
         "        logits = model(torch.tensor(X_test_seq, dtype=torch.float32).to(device))\n"
         "        probs = torch.softmax(logits, dim=1).cpu().numpy()"),
        ("def run_bilstm(X_train, y_train, X_test, seq_len=8, tau=0.50, delta=0.15):\n",
         "def run_bilstm(X_train, y_train, X_test, seq_len=8, tau=0.50, delta=0.15):\n"
         "    torch.manual_seed(42)                     # kết quả lặp lại được\n"),
    ]
    for a, b in sua:
        assert a in s, "Không tìm thấy đoạn cần sửa trong Bi-LSTM: %r" % a[:60]
        s = s.replace(a, b)
    bilstm = code(s)

    moi = [md(r'''
# NB2 — XGBoost walk-forward 2020–2025 → dự báo xu hướng (uptrend / sideway / downtrend)

Nhánh **AI** của luồng nghiên cứu, chạy **độc lập với NB1**. Tự đọc tệp OHLCV,
tự tính chỉ báo, tự gán nhãn, huấn luyện **XGBoost** theo **walk-forward**: mỗi năm
2020 → 2025 được dự báo bởi mô hình chỉ học dữ liệu **trước** năm đó. Ghép 6 năm dự
báo ngoài mẫu ra `gold_model_signals.csv`. Random Forest và Bi-LSTM của nhóm vẫn
giữ, bật thêm bằng `MO_HINH` nếu muốn so sánh.
''' + SO_DO)]
    moi.append(md("Import thư viện và nơi lưu tệp"))
    moi.append(code(r'''
import os, json
import pandas as pd
import numpy as np
''' + LUU_TRU))
    moi += [md(NHAP_MD), code(NHAP), md("Đọc dữ liệu"), code(DOC)]
    moi += [md(CHUAN_HOA_MD), code(CHUAN_HOA)]
    moi += [md("Loại bỏ dữ liệu lỗi và trùng"), code(LAM_SACH)]

    moi.append(md(r'''
Tính chỉ báo kỹ thuật (cùng công thức với NB1)

Model dùng **đúng bộ chỉ báo của NB1** (MA, EMA, RSI, MACD, Bollinger, biến động),
tính lại tại đây để NB2 chạy độc lập. Nhờ vậy phép so sánh ở NB3 là công bằng:
**cùng một lượng thông tin**, chỉ khác cách ra quyết định — luật cố định (NB1)
hay model học từ dữ liệu (NB2).
'''))
    moi.append(code(r'''
C = df['Close']
df['Return'] = np.log(C / C.shift(1))
df['MA10'] = C.rolling(window=10).mean()
df['MA30'] = C.rolling(window=30).mean()
df['MA50'] = C.rolling(window=50).mean()
df['EMA12'] = C.ewm(span=12, adjust=False).mean()
df['EMA26'] = C.ewm(span=26, adjust=False).mean()

delta = C.diff()
avg_gain = delta.clip(lower=0).rolling(window=14).mean()
avg_loss = (-delta.clip(upper=0)).rolling(window=14).mean()
df['RSI14'] = 100 - 100 / (1 + avg_gain / avg_loss)

df['MACD'] = df['EMA12'] - df['EMA26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

df['MA20'] = C.rolling(window=20).mean()
df['STD20'] = C.rolling(window=20).std()
df['BB_Upper'] = df['MA20'] + 2 * df['STD20']
df['BB_Lower'] = df['MA20'] - 2 * df['STD20']
df['Volatility20'] = df['Return'].rolling(window=20).std()
print('Đã tính %d chỉ báo.' % (df.shape[1] - 6))
'''))

    moi.append(md(r'''
Chuẩn bị đặc trưng cho model

**Mọi chỉ báo có đơn vị USD được chia cho giá đóng cửa.** Vàng đi từ khoảng
1.050 USD (2015) lên hơn 4.500 USD (2025). Model dạng cây không ngoại suy được:
nếu học trên `MA10 = 1.800` rồi gặp `MA10 = 4.000` ở giai đoạn kiểm tra, nó chỉ
trả về vùng giá cao nhất từng thấy. Đổi sang khoảng cách tương đối (ví dụ
`MA10 / Close − 1`) giúp thước đo so sánh được qua mọi mức giá.
'''))
    moi.append(code(r'''
dac_trung = pd.DataFrame(index=df.index)

# Chỉ báo có đơn vị giá → khoảng cách tương đối so với giá đóng cửa
for cot in ['MA10', 'MA30', 'MA50', 'EMA12', 'EMA26', 'MA20', 'BB_Upper', 'BB_Lower']:
    dac_trung['kc_' + cot.lower()] = df[cot] / C - 1
for cot in ['MACD', 'MACD_Signal', 'MACD_Hist', 'STD20']:
    dac_trung[cot.lower() + '_tuong_doi'] = df[cot] / C

# Chỉ báo vốn đã không phụ thuộc mức giá
dac_trung['rsi14'] = df['RSI14'] / 100
dac_trung['return'] = df['Return']
dac_trung['volatility20'] = df['Volatility20']
dac_trung['bb_vi_tri'] = (C - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

dac_trung = dac_trung.replace([np.inf, -np.inf], np.nan)
feature_cols = list(dac_trung.columns)
print('Số đặc trưng đưa vào model: %d' % len(feature_cols))
print(', '.join(feature_cols))
'''))

    moi.append(md(r'''
Walk-forward 2020 → 2025 (cửa sổ huấn luyện mở rộng dần)

Mỗi **năm kiểm tra Y**: huấn luyện trên **toàn bộ dữ liệu trước 01/01/Y**, rồi dự báo cả
năm Y. Sang năm sau, mô hình được huấn luyện lại, có thêm năm vừa qua. Ghép các năm dự báo
lại → **6 năm kiểm tra ngoài mẫu liên tiếp** cho NB3.

| Năm kiểm tra | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Huấn luyện | 2015 → 2019 | 2015 → 2020 | 2015 → 2021 | 2015 → 2022 | 2015 → 2023 | 2015 → 2024 |

Trong **mỗi lần**: dò lại tham số nhãn `p` chỉ trên tập huấn luyện của lần đó, và bỏ `H`
nến cuối của tập huấn luyện (purging) vì nhãn của chúng đã nhìn sang năm kiểm tra.
Dữ liệu không phủ 2020–2025 → tự chuyển về một lần chia 80 % / 20 %.
'''))
    moi.append(code(r'''
NAM_KIEM_TRA = [2020, 2021, 2022, 2023, 2024, 2025]
MO_HINH = ['xgb']          # thêm 'rf', 'lstm' để chạy cả Random Forest, Bi-LSTM của nhóm (chậm hơn)

nam = df['Date'].dt.year.to_numpy()
FOLD = [(str(y), nam < y, nam == y) for y in NAM_KIEM_TRA
        if (nam < y).sum() > 1000 and (nam == y).sum() > 100]
if not FOLD:
    la_train = np.arange(len(df)) < int(len(df) * 0.8)
    FOLD = [('20% cuối', la_train, ~la_train)]
    print('Dữ liệu không phủ 2020–2025 → một lần chia: 80 % đầu huấn luyện, 20 % cuối kiểm tra.')
print('Walk-forward: %d lần huấn luyện, mô hình %s' % (len(FOLD), MO_HINH))
for ten, tr, te in FOLD:
    print('  Kiểm tra %-9s | huấn luyện %6d nến (%s → %s) | kiểm tra %5d nến'
          % (ten, tr.sum(), df.loc[tr, 'Date'].min().date(), df.loc[tr, 'Date'].max().date(), te.sum()))
'''))

    moi.append(md(r'''
Gán nhãn −1 / 0 / 1 — đáp án cho model học

Nhãn dùng phương pháp **Triple Barrier** với rào cản theo **phần trăm giá**:

- Tại nến `t`, đặt rào trên `Close × (1 + p)` và rào dưới `Close × (1 − p)`.
- Quét tối đa `H` nến tiếp theo:
  chạm rào trên trước → **1 (trend tăng)**, chạm rào dưới trước → **−1 (trend giảm)**,
  hết `H` nến mà không chạm, hoặc chạm cả hai trong cùng một nến → **0 (sideway)**.
- `H` nến cuối cùng chưa đủ dữ liệu tương lai nên để trống.

**Tự thích nghi theo khung.** Mỗi khung có biên độ rất khác nhau (M15 đi vài chục
cent, D1 đi vài chục USD), nên `p` không đặt cứng mà được **dò tự động** sao cho
lớp sideway chiếm khoảng 25 %. Việc dò chỉ dùng **tập huấn luyện**.
'''))
    moi.append(code(r'''
# Rào thời gian H (số nến tối đa) theo khung
RAO_THOI_GIAN = {'M1': 60, 'M5': 48, 'M15': 32, 'M30': 24, 'H1': 24,
                 'H4': 12, 'D1': 10, 'W1': 8, 'MN': 6}
SO_NEN_TOI_DA = RAO_THOI_GIAN.get(KHUNG, 24)
TY_LE_SIDEWAY_MUC_TIEU = 0.25
LUOI_PHAN_TRAM = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5,
                  2.0, 3.0, 4.0, 5.0, 7.5, 10.0]


def gan_nhan(cao, thap, dong, p, H):
    """Triple Barrier theo phần trăm giá. p tính theo %, H là số nến tối đa."""
    n = len(dong)
    tren, duoi = dong * (1 + p / 100), dong * (1 - p / 100)
    nhan = np.zeros(n)
    da_xong = np.zeros(n, dtype=bool)
    for k in range(1, H + 1):
        c_k = np.full(n, np.nan); c_k[:n - k] = cao[k:]
        t_k = np.full(n, np.nan); t_k[:n - k] = thap[k:]
        cham_tren, cham_duoi = c_k >= tren, t_k <= duoi
        moi_cham = ~da_xong & (cham_tren | cham_duoi)
        nhan[moi_cham & cham_tren & ~cham_duoi] = 1
        nhan[moi_cham & cham_duoi & ~cham_tren] = -1
        da_xong |= moi_cham               # chạm cả hai trong cùng một nến → giữ 0
    nhan[max(n - H, 0):] = np.nan         # chưa đủ H nến tương lai
    return nhan


cao, thap, dong = df['High'].to_numpy(float), df['Low'].to_numpy(float), df['Close'].to_numpy(float)


def phan_bo(p, n_do):
    """Tỷ lệ tăng / sideway / giảm (%) khi gán nhãn n_do nến đầu với tham số p."""
    nh = gan_nhan(cao[:n_do], thap[:n_do], dong[:n_do], p, SO_NEN_TOI_DA)
    nh = nh[~np.isnan(nh)]
    return 100 * (nh == 1).mean(), 100 * (nh == 0).mean(), 100 * (nh == -1).mean()


def do_p(n_do):
    """Dò p sao cho sideway ≈ mục tiêu, CHỈ trên n_do nến đầu (tập huấn luyện của lần đó).

    Tỷ lệ sideway theo p có dạng CHỮ U, không đơn điệu:
     - p rất nhỏ: nến kế tiếp chạm CẢ HAI rào cùng lúc → gán 0. Đó là nhiễu, không phải sideway.
     - p lớn: hết H nến mà không chạm rào nào → 0 đúng nghĩa sideway.
    Vì vậy chỉ dò trên NHÁNH PHẢI (p lớn hơn điểm đáy), rồi chia đôi khoảng để đạt đúng mục tiêu.
    """
    bang = pd.DataFrame([dict(zip(['p_%', 'tang_%', 'sideway_%', 'giam_%'], (p,) + phan_bo(p, n_do)))
                         for p in LUOI_PHAN_TRAM]).round(2)
    muc_tieu = 100 * TY_LE_SIDEWAY_MUC_TIEU
    i_day = bang['sideway_%'].idxmin()
    nhanh_phai = bang.loc[i_day:]
    vuot = nhanh_phai[nhanh_phai['sideway_%'] >= muc_tieu]
    if vuot.empty:
        return float(nhanh_phai['p_%'].iloc[-1]), bang
    j = vuot.index[0]
    thap_p, cao_p = float(bang.loc[max(j - 1, i_day), 'p_%']), float(bang.loc[j, 'p_%'])
    for _ in range(20):
        giua = (thap_p + cao_p) / 2
        if phan_bo(giua, n_do)[1] < muc_tieu:
            thap_p = giua
        else:
            cao_p = giua
    return round(cao_p, 4), bang


# Minh họa trên tập huấn luyện của lần walk-forward ĐẦU TIÊN
n_dau = int(FOLD[0][1].sum())
P_CHON, bang_do = do_p(n_dau)
print('Khung %s | rào thời gian %d nến | minh họa lần kiểm tra %s: dò trên %d nến huấn luyện'
      % (KHUNG, SO_NEN_TOI_DA, FOLD[0][0], n_dau))
display(bang_do)
tg, sw, gm = phan_bo(P_CHON, n_dau)
print('→ Chọn p = %.4f %%  →  Tăng %.1f %% | Sideway %.1f %% | Giảm %.1f %%  (mục tiêu sideway %.0f %%)'
      % (P_CHON, tg, sw, gm, 100 * TY_LE_SIDEWAY_MUC_TIEU))
print('Mỗi lần walk-forward sẽ dò lại p trên tập huấn luyện của chính lần đó.')
'''))

    moi.append(md(r'''
Vùng đệm (purging) — áp dụng trong từng lần walk-forward

Nhãn tại nến `t` nhìn tới `H` nến sau. Vì vậy trong mỗi lần, chỉ những nến huấn luyện
có `t + H` còn nằm **trước** năm kiểm tra mới được dùng; `H` nến sát ranh giới bị loại để
nhãn không "nhìn thấy" năm kiểm tra.
'''))

    # Giữ nguyên logic của nhóm; chỉ đổi cách gọi: model dự báo XU HƯỚNG, không ra lệnh
    moi += doi_chu([giu(c[2]), giu(c[3]), giu(c[4]), giu(c[5]), giu(c[6]), giu(c[7]), giu(c[8]), bilstm], [
        ("sang tín hiệu giao dịch", "sang dự báo xu hướng"),
        ("Dùng chung cho cả 3 chiến lược dưới", "Dùng chung cho cả 3 mô hình dưới"),
        ("thành tín hiệu -1 (Sell), 0 (Flat), 1 (Buy)",
         "thành dự báo xu hướng -1 (downtrend), 0 (sideway), 1 (uptrend)"),
        ("# Mua khi xác suất tăng", "# Dự báo UPTREND khi xác suất tăng"),
        ("# Bán khi xác suất giảm", "# Dự báo DOWNTREND khi xác suất giảm"),
        ("-> Đứng ngoài", "-> SIDEWAY"),
        ("# Đệm 0 (Flat)", "# Đệm 0 (sideway)"),
    ])

    moi.append(md(r'''
Outputs — huấn luyện walk-forward và ghi tệp dự báo xu hướng cho NB3

Mỗi lần: dò `p` → gán nhãn → lấy nến huấn luyện (bỏ vùng đệm) → huấn luyện → dự báo năm
kiểm tra. Bảng đánh giá ghi **tỉ lệ dự báo trend đúng với nhãn** từng năm — độ chính xác
thuần của mô hình, trước khi đưa vào chiến lược ở NB3.
'''))
    moi.append(code(r'''
# Ngưỡng kép để xác định xu hướng (mặc định cấu hình Swing).
# Scalping: tau=0.45, delta=0.10 | Position: tau=0.60, delta=0.20
tau_config = 0.50
delta_config = 0.15

HAM = {'xgb': ('XGBoost', lambda Xtr, ytr, Xte: run_xgboost(Xtr, ytr, Xte, tau=tau_config, delta=delta_config)),
       'rf': ('Random Forest', lambda Xtr, ytr, Xte: run_random_forest(Xtr, ytr, Xte, tau=tau_config, delta=delta_config)),
       'lstm': ('Bi-LSTM', lambda Xtr, ytr, Xte: run_bilstm(Xtr, ytr, Xte, seq_len=8, tau=tau_config, delta=delta_config))}

du_dac_trung = dac_trung.notna().all(axis=1).to_numpy()
X_all = dac_trung[feature_cols].to_numpy()
cac_phan, danh_gia = [], []
for ten, la_train, la_test in FOLD:
    n_do = int(la_train.sum())                                   # tập huấn luyện là phần đầu chuỗi
    p = do_p(n_do)[0]
    nhan = gan_nhan(cao, thap, dong, p, SO_NEN_TOI_DA)
    i_train = np.flatnonzero(la_train & du_dac_trung & ~np.isnan(nhan))
    i_train = i_train[i_train < n_do - SO_NEN_TOI_DA]            # vùng đệm purging
    i_test = np.flatnonzero(la_test & du_dac_trung)
    X_train, y_train, X_test = X_all[i_train], nhan[i_train].astype(int), X_all[i_test]

    phan = df.iloc[i_test][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    phan['label'], phan['fold'], phan['p_nhan_%'] = nhan[i_test], ten, p
    for mh in MO_HINH:
        ten_mh, ham = HAM[mh]
        phan['sig_' + mh] = ham(X_train, y_train, X_test)
        tin, nh = phan['sig_' + mh], phan['label']
        co = tin.ne(0) & nh.notna()
        danh_gia.append({'Năm kiểm tra': ten, 'Mô hình': ten_mh, 'Nến huấn luyện': len(i_train),
                         'p nhãn (%)': p, 'Uptrend': int((tin == 1).sum()), 'Sideway': int((tin == 0).sum()),
                         'Downtrend': int((tin == -1).sum()),
                         'Dự báo trend đúng nhãn (%)': round(100 * (tin[co] == nh[co]).mean(), 2) if co.any() else np.nan})
    cac_phan.append(phan)
    print('  Năm %-9s: huấn luyện %6d nến, p = %.4f %% → %s' % (ten, len(i_train), p,
          ', '.join('%s %s' % (mh, phan['sig_' + mh].value_counts().sort_index().to_dict()) for mh in MO_HINH)))

df_test = pd.concat(cac_phan)
danh_gia = pd.DataFrame(danh_gia)
print('\n--- ĐÁNH GIÁ WALK-FORWARD (tín hiệu: -1 downtrend, 0 sideway, 1 uptrend) ---')
print(danh_gia.to_string(index=False))
luu_tep(danh_gia, 'danh_gia_walk_forward.csv')

# Ghi tệp cho NB3. Cột dự báo xu hướng có tiền tố "sig_" để NB3 nhận đúng.
ra = df_test.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high',
                             'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
duong_dan_ra = luu_tep(ra, 'gold_model_signals.csv')

# Xem các nến mô hình dự báo có xu hướng (khác sideway)
co_tin = (df_test[['sig_' + mh for mh in MO_HINH]] != 0).any(axis=1)
display(df_test[co_tin][['Date', 'fold', 'label'] + ['sig_' + mh for mh in MO_HINH]].head(15))
'''))
    moi.append(code(r'''
# Tải tệp về máy (không bắt buộc nếu đã lưu trên Google Drive)
if TREN_COLAB and not duong_dan_ra.startswith('/content/drive'):
    files.download(duong_dan_ra)
'''))
    g["cells"] = moi
    return g


# ══════════════════════════════════════════════════════════════ NB3
def nb3() -> dict:
    """NB3 mới: 15 chiến lược × 7 nguồn dự báo, engine đánh 2 chiều (src/nb3_chien_luoc.py)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nb3_chien_luoc as K
    g = doc_goc(NB3)
    c = g["cells"]
    assert "def plot_results" in nguon(c[4]), "Bản gốc NB3 đã đổi cấu trúc"
    ve = nguon(c[4])                       # hàm vẽ của nhóm; mũi tên đặt ở thời điểm VÀO lệnh
    for a, b in (("x=buys['exit_time'], y=buys['entry_price']", "x=buys['entry_time'], y=buys['entry_price']"),
                 ("x=sells['exit_time'], y=sells['entry_price']", "x=sells['entry_time'], y=sells['entry_price']")):
        assert a in ve, "Không tìm thấy đoạn cần sửa trong plot_results: %r" % a
        ve = ve.replace(a, b)

    g["cells"] = [
        md(K.GIOI_THIEU + "\n" + SO_DO), code(K.IMPORT + LUU_TRU),
        md(K.CAU_HINH_MD), code(K.CAU_HINH),
        md(K.NAP_MD), code(K.NAP),
        md(K.CHIEN_LUOC_MD), code(K.CHIEN_LUOC),
        md(K.ENGINE_MD), code(K.ENGINE),
        md(K.BAO_CAO_MD), code(K.BAO_CAO),
        md("Hàm vẽ khớp lệnh của nhóm"), code(ve),
        md(K.CHAY_MD), code(K.CHAY),
        md(K.NGAU_NHIEN_MD), code(K.NGAU_NHIEN),
        md(K.SO_SANH_MD), code(K.SO_SANH),
        md(K.HINH_MD), code(K.HINH),
        code(K.TAI_VE),
    ]
    return g


def nb3_cu() -> dict:
    """NB3 trước đây (một chiến lược, đóng lệnh theo tín hiệu). Giữ để tham khảo."""
    g = doc_goc(NB3)
    c = g["cells"]
    assert "def plot_results" in nguon(c[4]), "Bản gốc NB3 đã đổi cấu trúc"

    # Hàm vẽ của nhóm đặt mũi tên vào lệnh tại THỜI ĐIỂM ĐÓNG lệnh (exit_time) nhưng
    # ở GIÁ VÀO lệnh — mũi tên lệch khỏi nến thật. Sửa: dùng thời điểm vào lệnh.
    ve = nguon(c[4])
    for a, b in (("x=buys['exit_time'], y=buys['entry_price']", "x=buys['entry_time'], y=buys['entry_price']"),
                 ("x=sells['exit_time'], y=sells['entry_price']", "x=sells['entry_time'], y=sells['entry_price']")):
        assert a in ve, "Không tìm thấy đoạn cần sửa trong plot_results: %r" % a
        ve = ve.replace(a, b)
    ve = code(ve)

    moi = [md(r'''
# NB3 — Backtest kiểu MT5: so sánh NB1 (kỹ thuật) và NB2 (AI)

Đọc tệp kết quả của **cả hai nhánh**, ghép theo thời gian, rồi backtest mọi tín
hiệu **trên cùng một giai đoạn** bằng một engine mô phỏng **Strategy Tester của
MetaTrader 5**: balance, đòn bẩy, lot, spread, commission, swap, SL/TP, margin,
stop-out. Báo cáo theo đúng mẫu MT5.
''' + SO_DO + r'''

| Nhóm | Trường hợp |
|---|---|
| NB1 — Kỹ thuật | `sig_ma`, `sig_rsi`, `sig_macd`, `sig_tong_hop_ky_thuat` (≥ 2/3 luật đồng ý) |
| NB2 — AI | `sig_xgb`, `sig_rf`, `sig_lstm` |
| Mốc chuẩn | `sig_mua_giu` — mua ở nến đầu, giữ tới nến cuối |
''')]
    moi.append(code(r'''
import os, io, json
import warnings
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
''' + LUU_TRU))

    # ── Thông số MT5
    moi.append(md(r'''
BƯỚC 0: THÔNG SỐ BACKTEST (giống Strategy Tester của MT5)

| Nhóm | Thông số | Tương ứng trong MT5 |
|---|---|---|
| Tài khoản | `so_du_ban_dau`, `don_bay`, `stop_out_phan_tram` | Settings → Deposit, Leverage; mức Stop Out của sàn |
| Symbol | `kich_thuoc_hop_dong`, `point`, `spread_points`, `lot_min/max/step` | Chuột phải XAUUSD → **Specification** |
| Chi phí | `hoa_hong_lot_1_chieu`, `swap_long_points`, `swap_short_points`, `ngay_swap_x3` | Commission và Swap của tài khoản |
| Lệnh (Inputs) | `che_do_lot`, `lot`, `rui_ro_phan_tram`, `sl_points`, `tp_points` | Tab **Inputs** của EA |

**Swap để mặc định 0** vì mỗi sàn một mức khác nhau. Hãy mở Specification của
XAUUSD trên MT5 rồi điền đúng `swap_long_points` và `swap_short_points` của sàn.
Để 0 thì các lệnh giữ qua đêm sẽ trông tốt hơn thực tế.

Với vàng: `1 lot = 100 oz`, `1 point = 0,01 USD`. Vậy giá đi **1 point** thì một lệnh
**1 lot** lãi/lỗ **1 USD**, và 30 points spread = 0,30 USD/oz = 30 USD mỗi lot.

**Cách khớp lệnh (như chế độ "Open prices only" của MT5).** Tín hiệu chốt khi nến
`t` đóng, lệnh khớp ở giá **mở** của nến `t + 1`. Giá trong dữ liệu là giá **Bid**;
**Ask = Bid + spread**. Lệnh BUY mua ở Ask và đóng ở Bid, lệnh SELL bán ở Bid và
đóng ở Ask. SL/TP được kiểm tra bằng High/Low trong nến; nếu một nến chạm cả SL
lẫn TP thì coi như **SL khớp trước** (giả định thận trọng).
'''))
    moi.append(code(r'''
@dataclass
class CauHinhMT5:
    # ── Tài khoản (tab Settings của Strategy Tester)
    so_du_ban_dau: float = 10_000.0      # Initial deposit (USD)
    don_bay: int = 100                   # Leverage 1:100
    stop_out_phan_tram: float = 50.0     # Stop Out: margin level ≤ 50 % → sàn tự đóng lệnh

    # ── Symbol XAUUSD (chuột phải symbol → Specification)
    ky_hieu: str = 'XAUUSD'
    kich_thuoc_hop_dong: float = 100.0   # Contract size: 1 lot = 100 oz
    point: float = 0.01                  # 1 point = 0,01 USD
    spread_points: int = 30              # 30 points = 0,30 USD
    lot_min: float = 0.01
    lot_max: float = 100.0
    lot_step: float = 0.01

    # ── Chi phí giao dịch
    hoa_hong_lot_1_chieu: float = 3.5    # Commission mỗi lot mỗi chiều (vào 3,5 + ra 3,5 = 7 USD/lot)
    swap_long_points: float = 0.0        # Swap BUY (points/lot/đêm) — ĐIỀN THEO SÀN
    swap_short_points: float = 0.0       # Swap SELL (points/lot/đêm) — ĐIỀN THEO SÀN
    gio_qua_dem_utc: int = 22            # thời điểm rollover tính swap (UTC)
    ngay_swap_x3: int = 2                # 0 = Thứ Hai … 2 = Thứ Tư: ngày tính swap ×3 (bù cuối tuần)

    # ── Lệnh (tab Inputs của EA)
    che_do_lot: str = 'co_dinh'          # 'co_dinh' = lot cố định | 'rui_ro' = % số dư theo SL
    lot: float = 0.10                    # dùng khi che_do_lot = 'co_dinh'
    rui_ro_phan_tram: float = 1.0        # dùng khi che_do_lot = 'rui_ro' (bắt buộc sl_points > 0)
    sl_points: int = 0                   # Stop Loss (points); 0 = không đặt
    tp_points: int = 0                   # Take Profit (points); 0 = không đặt

    @property
    def usd_moi_point_moi_lot(self):     # 100 oz × 0,01 = 1 USD
        return self.kich_thuoc_hop_dong * self.point


CH = CauHinhMT5()
if CH.che_do_lot == 'rui_ro' and CH.sl_points <= 0:
    raise ValueError("che_do_lot = 'rui_ro' cần sl_points > 0 để tính khối lượng.")
for k, v in asdict(CH).items():
    print('  %-22s %s' % (k, v))
print('  %-22s %.2f USD' % ('1 point × 1 lot =', CH.usd_moi_point_moi_lot))
'''))

    # ── Nạp và ghép
    moi.append(md(r'''
BƯỚC 1: NẠP VÀ GHÉP KẾT QUẢ CỦA NB1 VÀ NB2

NB1 cho tín hiệu trên **toàn bộ** dữ liệu, còn NB2 chỉ có tín hiệu trên **tập kiểm
tra** (dự báo ngoài mẫu). Hai tệp được ghép theo thời gian và **chỉ giữ phần
chung**, để mọi chiến lược được backtest trên đúng cùng một giai đoạn.

Ô dưới cũng đối chiếu giá đóng cửa của hai tệp để phát hiện trường hợp NB1 và NB2
vô tình được nạp **hai bộ dữ liệu khác nhau**.
'''))
    moi.append(code(r'''
def _doc(ten):
    d = pd.read_csv(tim_tep(ten), encoding='utf-8-sig')
    d.columns = [c.strip().lower() for c in d.columns]
    d['time'] = pd.to_datetime(d['time'])
    return d.sort_values('time').drop_duplicates('time')

ky_thuat = _doc('gold_price_technical_signal.csv')     # NB1
ai = _doc('gold_model_signals.csv')                     # NB2
print('NB1: %d nến, %s → %s' % (len(ky_thuat), ky_thuat['time'].min(), ky_thuat['time'].max()))
print('NB2: %d nến, %s → %s' % (len(ai), ai['time'].min(), ai['time'].max()))

# Tín hiệu luật của NB1 → đổi tên có tiền tố "sig_"
nb1 = ky_thuat[['time', 'open', 'high', 'low', 'close', 'volume', 'rsi14', 'macd_hist',
                'ma_signal', 'rsi_signal', 'macd_signal_final', 'signal']].rename(columns={
    'ma_signal': 'sig_ma', 'rsi_signal': 'sig_rsi',
    'macd_signal_final': 'sig_macd', 'signal': 'sig_tong_hop_ky_thuat'})
nb2 = ai[['time', 'close', 'sig_xgb', 'sig_rf', 'sig_lstm']].rename(columns={'close': 'close_nb2'})

df_master = nb1.merge(nb2, on='time', how='inner').sort_values('time').reset_index(drop=True)
if df_master.empty:
    raise ValueError('NB1 và NB2 không có mốc thời gian chung — kiểm tra lại hai tệp đầu vào.')

lech = (df_master['close'] - df_master['close_nb2']).abs() > 1e-6
if lech.mean() > 0.01:
    print('⚠ %.1f %% số nến có giá đóng cửa KHÁC NHAU giữa NB1 và NB2 — hai notebook '
          'có thể đã được nạp hai bộ dữ liệu khác nhau.' % (100 * lech.mean()))
else:
    print('✓ Giá đóng cửa của NB1 và NB2 khớp nhau: cùng một bộ dữ liệu.')
df_master = df_master.drop(columns='close_nb2').rename(columns={'time': 'timestamp'})
print('\nGiai đoạn backtest chung: %s → %s (%d nến)'
      % (df_master['timestamp'].min(), df_master['timestamp'].max(), len(df_master)))
'''))

    moi.append(md(r'''
BƯỚC 2: CHIẾN LƯỢC — ĐỔI DỰ BÁO XU HƯỚNG THÀNH LỆNH BUY / SELL / FLAT

NB1 và NB2 chỉ cho **dự báo xu hướng** (1 = uptrend, 0 = sideway, −1 = downtrend).
Bước này áp **chiến lược** để quyết định lệnh ở mỗi nến:

| `CHIEN_LUOC` | Uptrend | Sideway | Downtrend |
|---|---|---|---|
| `'theo_xu_huong'` (mặc định) | **BUY** | **FLAT** — đóng lệnh, đứng ngoài | **SELL** |
| `'giu_lenh_khi_sideway'` | **BUY** | giữ nguyên lệnh đang có | **SELL** |
| `'chi_mua'` | **BUY** | **FLAT** | **FLAT** (không bán khống) |

Mọi dự báo (của NB1 lẫn NB2) đi qua **cùng một chiến lược**, nên chênh lệch kết quả chỉ
đến từ chất lượng dự báo xu hướng. Mốc chuẩn `sig_mua_giu` là **lệnh** BUY giữ suốt
giai đoạn, không qua chiến lược. Từ đây, các cột `sig_*` mang **lệnh**:
1 = BUY, 0 = FLAT, −1 = SELL; dự báo gốc được giữ ở cột `xu_huong_*`.
'''))
    moi.append(code(r'''
CHIEN_LUOC = 'theo_xu_huong'      # 'theo_xu_huong' | 'giu_lenh_khi_sideway' | 'chi_mua'

def xu_huong_thanh_lenh(xu_huong, chien_luoc):
    """Dự báo xu hướng (1 / 0 / -1) → lệnh (1 = BUY, 0 = FLAT, -1 = SELL)."""
    x = pd.Series(xu_huong).fillna(0).astype(int)
    if chien_luoc == 'theo_xu_huong':
        return x
    if chien_luoc == 'giu_lenh_khi_sideway':
        return x.replace(0, np.nan).ffill().fillna(0).astype(int)
    if chien_luoc == 'chi_mua':
        return x.clip(lower=0)
    raise ValueError('CHIEN_LUOC không hợp lệ: %r' % chien_luoc)

cot_du_bao = [c for c in df_master.columns if c.startswith('sig_')]
for cot in cot_du_bao:
    df_master['xu_huong_' + cot[4:]] = df_master[cot]
    df_master[cot] = xu_huong_thanh_lenh(df_master[cot], CHIEN_LUOC).values

# Mốc chuẩn: lệnh BUY ở nến đầu và giữ tới cuối (không phải dự báo)
df_master['sig_mua_giu'] = 1

# Chỉ nhận cột BẮT ĐẦU bằng "sig_"
signal_columns = [c for c in df_master.columns if c.startswith('sig_')]
NHOM = {'sig_ma': 'NB1 Kỹ thuật', 'sig_rsi': 'NB1 Kỹ thuật', 'sig_macd': 'NB1 Kỹ thuật',
        'sig_tong_hop_ky_thuat': 'NB1 Kỹ thuật', 'sig_xgb': 'NB2 AI', 'sig_rf': 'NB2 AI',
        'sig_lstm': 'NB2 AI', 'sig_mua_giu': 'Mốc chuẩn'}

TEN_XU_HUONG = {1: 'UPTREND', 0: 'SIDEWAY', -1: 'DOWNTREND'}
TEN_LENH = {1: 'BUY', 0: 'FLAT', -1: 'SELL'}
print('Chiến lược: %s — %d trường hợp sẽ backtest\n' % (CHIEN_LUOC, len(signal_columns)))
print('  %-24s %-13s %-46s %s' % ('Trường hợp', 'Nhóm', 'Dự báo xu hướng (số nến)', 'Lệnh sau chiến lược'))
for cot in signal_columns:
    xh = 'xu_huong_' + cot[4:]
    du_bao = (df_master[xh].map(TEN_XU_HUONG).value_counts().to_dict() if xh in df_master else '—')
    lenh = df_master[cot].map(TEN_LENH).value_counts().to_dict()
    print('  %-24s %-13s %-46s %s' % (cot, NHOM.get(cot, ''), du_bao, lenh))
'''))

    # ── Engine MT5
    moi.append(md(r'''
BƯỚC 3: ENGINE BACKTEST KIỂU MT5

Mỗi nến, engine làm đúng bốn việc theo thứ tự thời gian:

1. **Swap qua đêm.** Nếu giữa nến trước và nến này có mốc rollover (22:00 UTC từ
   Thứ Hai đến Thứ Sáu) thì cộng swap cho lệnh đang mở. Ngày Thứ Tư tính ×3 để bù
   hai ngày cuối tuần, như quy ước của sàn.
2. **Khớp lệnh ở giá mở theo tín hiệu nến trước.** Tín hiệu đổi thì đóng lệnh cũ,
   rồi mở lệnh mới nếu tín hiệu khác 0. Trước khi mở, engine kiểm tra **ký quỹ**
   `lot × 100 oz × giá / đòn bẩy`: không đủ margin tự do thì bỏ lệnh, giống lỗi
   *"Not enough money"* của MT5.
3. **SL/TP trong nến.** Kiểm tra bằng High/Low (lệnh SELL dùng giá Ask = Bid +
   spread). Nếu giá mở nhảy qua SL thì khớp ở giá mở, vì thị trường không có giá ở
   mức SL. Sau khi dính SL/TP, engine chờ **tín hiệu mới** mới vào lại, không vào
   lại ngay ở nến kế tiếp.
4. **Cập nhật Equity và kiểm tra Stop Out.** Equity = Balance + lãi/lỗ thả nổi +
   swap. Nếu mức ký quỹ `Equity / Margin` ≤ 50 % thì đóng lệnh cưỡng bức.

Hết dữ liệu mà lệnh vẫn mở thì đóng ở giá cuối, giống MT5 đóng lệnh khi kết thúc
test.
'''))
    moi.append(code(r'''
def don_vi_swap(thoi_gian, ch):
    """Số 'đêm swap' phát sinh giữa nến i−1 và nến i (Thứ Tư ×3, bỏ Thứ Bảy và Chủ nhật)."""
    t = pd.to_datetime(pd.Series(thoi_gian)).reset_index(drop=True)
    ngay = ((t - pd.Timedelta(hours=ch.gio_qua_dem_utc)).dt.floor('D')
            - pd.Timestamp('1970-01-01')).dt.days.to_numpy()
    ra = np.zeros(len(t))
    for i in np.nonzero(np.diff(ngay) > 0)[0] + 1:
        for d in range(ngay[i - 1] + 1, ngay[i] + 1):
            thu = (d + 3) % 7                     # 01/01/1970 là Thứ Năm → Thứ Hai = 0
            if thu < 5:
                ra[i] += 3 if thu == ch.ngay_swap_x3 else 1
    return ra


def _lam_tron_lot(lot, ch):
    lot = np.floor(lot / ch.lot_step + 1e-9) * ch.lot_step
    return float(min(max(lot, ch.lot_min), ch.lot_max))


def backtest_mt5(df, signal_col, ch, dv_swap=None):
    """Trả về (đường vốn theo nến, danh sách lệnh, danh sách deal kiểu MT5)."""
    t = df['timestamp'].to_numpy()
    o, h, l, c = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    sig = df[signal_col].fillna(0).astype(int).to_numpy()
    n = len(df)
    sp = ch.spread_points * ch.point
    hd = ch.kich_thuoc_hop_dong
    dv_swap = don_vi_swap(t, ch) if dv_swap is None else dv_swap

    balance = ch.so_du_ban_dau
    vt = None                   # vị thế đang mở
    khoa = None                 # tín hiệu bị khóa sau SL/TP: chờ tín hiệu mới mới vào lại
    deals, lenh = [], []
    so_deal = [0]
    bal_arr, eq_arr, ml_arr = np.empty(n), np.empty(n), np.full(n, np.nan)

    def _deal(i, loai, huong, lot, gia, hh, swap, loi_nhuan, ghi_chu):
        so_deal[0] += 1
        deals.append({'Time': t[i], 'Deal': so_deal[0], 'Symbol': ch.ky_hieu, 'Type': loai,
                      'Direction': huong, 'Volume': lot, 'Price': round(gia, 3),
                      'Commission': round(-hh, 2), 'Swap': round(swap, 2),
                      'Profit': round(loi_nhuan, 2), 'Balance': round(balance, 2),
                      'Comment': ghi_chu})

    def _dong(i, gia, ghi_chu):
        nonlocal balance, vt
        loi_nhuan = (gia - vt['gia']) * vt['huong'] * vt['lot'] * hd
        hh = ch.hoa_hong_lot_1_chieu * vt['lot']
        balance += loi_nhuan + vt['swap'] - hh
        _deal(i, 'sell' if vt['huong'] > 0 else 'buy', 'out', vt['lot'], gia, hh,
              vt['swap'], loi_nhuan, ghi_chu)
        lenh.append({'entry_time': t[vt['i']], 'exit_time': t[i],
                     'type': 'BUY' if vt['huong'] > 0 else 'SELL', 'lot': vt['lot'],
                     'entry_price': vt['gia'], 'exit_price': gia,
                     'commission': -(vt['hh'] + hh), 'swap': vt['swap'], 'profit': loi_nhuan,
                     'pnl': loi_nhuan + vt['swap'] - vt['hh'] - hh, 'comment': ghi_chu})
        vt = None

    for i in range(n):
        # 1. Swap cho lệnh giữ qua mốc rollover giữa nến i−1 và nến i
        if vt is not None and dv_swap[i] > 0:
            diem = ch.swap_long_points if vt['huong'] > 0 else ch.swap_short_points
            vt['swap'] += diem * ch.usd_moi_point_moi_lot * vt['lot'] * dv_swap[i]

        # 2. Khớp lệnh ở giá MỞ của nến i theo tín hiệu đã chốt ở nến i−1
        if i > 0:
            muon = sig[i - 1]
            if khoa is not None and muon != khoa:
                khoa = None
            if vt is not None and muon != vt['huong']:
                _dong(i, o[i] if vt['huong'] > 0 else o[i] + sp, 'tín hiệu')
            if vt is None and muon != 0 and khoa is None:
                gia = o[i] + sp if muon > 0 else o[i]          # BUY ở Ask, SELL ở Bid
                if ch.che_do_lot == 'rui_ro':
                    lot = _lam_tron_lot(balance * ch.rui_ro_phan_tram / 100
                                        / (ch.sl_points * ch.usd_moi_point_moi_lot), ch)
                else:
                    lot = _lam_tron_lot(ch.lot, ch)
                ky_quy = lot * hd * gia / ch.don_bay
                if ky_quy > balance:
                    _deal(i, 'buy' if muon > 0 else 'sell', 'in', lot, gia, 0, 0, 0,
                          'bỏ lệnh: không đủ ký quỹ (Not enough money)')
                    so_deal[0] -= 1
                    deals.pop()
                else:
                    hh = ch.hoa_hong_lot_1_chieu * lot
                    balance -= hh
                    vt = {'i': i, 'huong': int(muon), 'lot': lot, 'gia': gia, 'hh': hh, 'swap': 0.0,
                          'sl': (gia - muon * ch.sl_points * ch.point) if ch.sl_points > 0 else None,
                          'tp': (gia + muon * ch.tp_points * ch.point) if ch.tp_points > 0 else None}
                    _deal(i, 'buy' if muon > 0 else 'sell', 'in', lot, gia, hh, 0, 0, 'tín hiệu')

        # 3. SL/TP trong nến — chạm cả hai trong cùng nến thì coi như SL khớp trước
        if vt is not None:
            if vt['huong'] > 0:       # BUY đóng ở Bid
                if vt['sl'] is not None and l[i] <= vt['sl']:
                    _dong(i, min(vt['sl'], o[i]), 'sl'); khoa = 1
                elif vt['tp'] is not None and h[i] >= vt['tp']:
                    _dong(i, max(vt['tp'], o[i]), 'tp'); khoa = 1
            else:                     # SELL đóng ở Ask = Bid + spread
                if vt['sl'] is not None and h[i] + sp >= vt['sl']:
                    _dong(i, max(vt['sl'], o[i] + sp), 'sl'); khoa = -1
                elif vt['tp'] is not None and l[i] + sp <= vt['tp']:
                    _dong(i, min(vt['tp'], o[i] + sp), 'tp'); khoa = -1

        # 4. Equity, mức ký quỹ và Stop Out
        if vt is not None:
            gia_dong = c[i] if vt['huong'] > 0 else c[i] + sp
            tha_noi = (gia_dong - vt['gia']) * vt['huong'] * vt['lot'] * hd + vt['swap']
            ky_quy = vt['lot'] * hd * c[i] / ch.don_bay
            ml = 100 * (balance + tha_noi) / ky_quy
            if ml <= ch.stop_out_phan_tram:
                _dong(i, gia_dong, 'so %.1f%%' % ml)
                bal_arr[i] = eq_arr[i] = balance
            else:
                bal_arr[i], eq_arr[i], ml_arr[i] = balance, balance + tha_noi, ml
        else:
            bal_arr[i] = eq_arr[i] = balance

    if vt is not None:              # hết dữ liệu: đóng lệnh như MT5 khi kết thúc test
        _dong(n - 1, c[n - 1] if vt['huong'] > 0 else c[n - 1] + sp, 'end of test')
        bal_arr[n - 1] = eq_arr[n - 1] = balance

    df_result = df[['timestamp', 'open', 'high', 'low', 'close']].copy()
    for k in ('rsi14', 'macd_hist'):
        if k in df.columns:
            df_result[k] = df[k].values
    df_result['balance'], df_result['equity'], df_result['margin_level'] = bal_arr, eq_arr, ml_arr
    return df_result, pd.DataFrame(lenh), pd.DataFrame(deals)
'''))

    # ── Báo cáo MT5
    moi.append(md(r'''
BƯỚC 4: BÁO CÁO THEO MẪU MT5

| Chỉ số | Cách tính |
|---|---|
| Total Net Profit | Gross Profit + Gross Loss (đã trừ commission, cộng swap) |
| Profit Factor | Gross Profit / \|Gross Loss\| |
| Expected Payoff | Lãi ròng trung bình mỗi lệnh |
| Recovery Factor | Net Profit / Equity Drawdown Maximal |
| Balance / Equity Drawdown Maximal | Mức sụt lớn nhất tính từ đỉnh, bằng USD và % |
| Drawdown Absolute | Số dư ban đầu − số dư thấp nhất (nếu thấp hơn ban đầu) |
| Sharpe, Sortino, Calmar | Tính trên đường Equity theo từng nến, quy năm theo số nến thực tế mỗi năm |

**Lưu ý:** MT5 tính Sharpe theo công thức riêng, nên con số ở đây có thể lệch so
với báo cáo MT5 thật. Các chỉ số còn lại tính đúng theo định nghĩa của MT5.
'''))
    moi.append(code(r'''
def _sut_giam(chuoi):
    """(sụt lớn nhất USD, % tại lần đó, % lớn nhất, USD tại lần đó)"""
    dinh = np.maximum.accumulate(chuoi)
    usd, pct = dinh - chuoi, (dinh - chuoi) / dinh
    i_usd, i_pct = int(np.argmax(usd)), int(np.argmax(pct))
    return usd[i_usd], 100 * pct[i_usd], 100 * pct[i_pct], usd[i_pct]


def _chuoi_lien_tiep(pnl, thang):
    tot = dem = 0; tien_tot = tien = 0.0
    for x in pnl:
        if (x > 0) == thang and x != 0:
            dem += 1; tien += x
            if dem > tot or (dem == tot and abs(tien) > abs(tien_tot)):
                tot, tien_tot = dem, tien
        else:
            dem, tien = 0, 0.0
    return tot, tien_tot


def bao_cao_mt5(df_result, trades, ch):
    von0 = ch.so_du_ban_dau
    bal, eq = df_result['balance'].to_numpy(), df_result['equity'].to_numpy()
    ts = df_result['timestamp']
    so_nam = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / (365.25 * 86400), 1e-9)
    nen_moi_nam = len(eq) / so_nam

    # Lợi suất mỗi nến:
    #  · Lot CỐ ĐỊNH: lãi/lỗ cộng dồn theo USD → chia cho VỐN BAN ĐẦU. Chia cho equity hiện
    #    tại sẽ phóng đại từng khoản lãi khi tài khoản đã sụt sâu, có thể cho Sharpe DƯƠNG
    #    dù tài khoản lỗ (đã gặp: mua-và-giữ lỗ −36 % mà Sharpe = +0,52).
    #  · Lot theo % RỦI RO: lãi kép → dùng % thay đổi của equity.
    if ch.che_do_lot == 'rui_ro':
        r = pd.Series(eq).pct_change().dropna()
    else:
        r = pd.Series(eq).diff().dropna() / von0
    # Độ lệch phía dưới tính trên MỌI nến (nến lãi tính là 0), không phải độ lệch chuẩn
    # của riêng các nến lỗ — cách sau làm Sortino của chiến lược giao dịch thưa bị thấp sai.
    sd, sd_giam = r.std(), np.sqrt((np.minimum(r, 0) ** 2).mean())
    b_usd, b_pct, b_rel_pct, b_rel_usd = _sut_giam(bal)
    e_usd, e_pct, e_rel_pct, e_rel_usd = _sut_giam(eq)

    pnl = trades['pnl'] if len(trades) else pd.Series(dtype=float)
    lai, lo = pnl[pnl > 0].sum(), pnl[pnl < 0].sum()
    net = bal[-1] - von0
    lai_nam = (max(eq[-1], 1e-9) / von0) ** (1 / so_nam) - 1
    mua = trades[trades['type'] == 'BUY'] if len(trades) else trades
    ban = trades[trades['type'] == 'SELL'] if len(trades) else trades
    tl = lambda x: 100 * (x['pnl'] > 0).mean() if len(x) else 0.0
    ct_thang = _chuoi_lien_tiep(pnl, True)
    ct_thua = _chuoi_lien_tiep(pnl, False)
    ml = df_result['margin_level'].dropna()

    return {
        'Initial Deposit': von0,
        'Total Net Profit': round(net, 2),
        'Total Return (%)': round(100 * net / von0, 2),
        'Gross Profit': round(lai, 2),
        'Gross Loss': round(lo, 2),
        'Profit Factor': round(lai / -lo, 3) if lo < 0 else np.nan,
        'Expected Payoff': round(net / len(trades), 2) if len(trades) else 0.0,
        'Recovery Factor': round(net / e_usd, 3) if e_usd > 0 else np.nan,
        'Sharpe Ratio': round(r.mean() / sd * np.sqrt(nen_moi_nam), 3) if sd > 0 else np.nan,
        'Sortino Ratio': round(r.mean() / sd_giam * np.sqrt(nen_moi_nam), 3) if sd_giam > 0 else np.nan,
        'Calmar Ratio': round(lai_nam / (e_rel_pct / 100), 3) if e_rel_pct > 0 else np.nan,
        'Balance Drawdown Absolute': round(max(von0 - bal.min(), 0), 2),
        'Balance Drawdown Maximal': '%.2f (%.2f%%)' % (b_usd, b_pct),
        'Balance Drawdown Relative': '%.2f%% (%.2f)' % (b_rel_pct, b_rel_usd),
        'Equity Drawdown Absolute': round(max(von0 - eq.min(), 0), 2),
        'Equity Drawdown Maximal': '%.2f (%.2f%%)' % (e_usd, e_pct),
        'Equity Drawdown Relative': '%.2f%% (%.2f)' % (e_rel_pct, e_rel_usd),
        'Max Drawdown (%)': round(e_rel_pct, 2),
        'Total Trades': len(trades),
        'Short Trades (won %)': '%d (%.2f%%)' % (len(ban), tl(ban)),
        'Long Trades (won %)': '%d (%.2f%%)' % (len(mua), tl(mua)),
        'Profit Trades (% of total)': '%d (%.2f%%)' % ((pnl > 0).sum(), tl(trades) if len(trades) else 0),
        'Loss Trades (% of total)': '%d (%.2f%%)' % ((pnl <= 0).sum(),
                                                     100 - tl(trades) if len(trades) else 0),
        'Win Rate (%)': round(tl(trades), 2) if len(trades) else 0.0,
        'Largest profit trade': round(pnl.max(), 2) if len(pnl) else 0.0,
        'Largest loss trade': round(pnl.min(), 2) if len(pnl) else 0.0,
        'Average profit trade': round(pnl[pnl > 0].mean(), 2) if (pnl > 0).any() else 0.0,
        'Average loss trade': round(pnl[pnl < 0].mean(), 2) if (pnl < 0).any() else 0.0,
        'Maximum consecutive wins ($)': '%d (%.2f)' % ct_thang,
        'Maximum consecutive losses ($)': '%d (%.2f)' % ct_thua,
        'Total Commission': round(trades['commission'].sum(), 2) if len(trades) else 0.0,
        'Total Swap': round(trades['swap'].sum(), 2) if len(trades) else 0.0,
        'Minimal margin level (%)': round(ml.min(), 2) if len(ml) else np.nan,
    }


def in_bao_cao_mt5(bc, ten):
    """In báo cáo hai cột như tab Backtest của MT5."""
    print('=' * 92)
    print('STRATEGY TESTER REPORT — %s | %s | %s → %s'
          % (ten, CH.ky_hieu, df_master['timestamp'].min(), df_master['timestamp'].max()))
    print('=' * 92)
    k = list(bc.items())
    nua = (len(k) + 1) // 2
    for (a, x), (b, y) in zip(k[:nua], k[nua:] + [('', '')]):
        print('  %-30s %-16s  %-30s %s' % (a, x, b, y))
'''))

    moi.append(md("BƯỚC 5: HÀM VẼ CHART"))
    moi.append(ve)
    moi.append(code(r'''
def plot_mt5(df_result, ten):
    """Đồ thị Balance / Equity như tab Graph của MT5, kèm mức ký quỹ."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.72, 0.28],
                        subplot_titles=['Balance / Equity — %s' % ten, 'Margin level (%)'])
    fig.add_trace(go.Scatter(x=df_result['timestamp'], y=df_result['balance'], name='Balance',
                             line=dict(color='#1f4e9c', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_result['timestamp'], y=df_result['equity'], name='Equity',
                             line=dict(color='#2e9e4f', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_result['timestamp'], y=df_result['margin_level'],
                             name='Margin level', line=dict(color='#b5651d', width=1)), row=2, col=1)
    fig.add_hline(y=CH.stop_out_phan_tram, line_dash='dash', line_color='red', row=2, col=1)
    fig.update_layout(height=650, template='plotly_white')
    fig.show()


def plot_equity_so_sanh(results_store, ds_cot):
    """Đường Equity của mọi trường hợp trên cùng một biểu đồ."""
    fig = go.Figure()
    for cot in ds_cot:
        d = results_store[cot][0]
        net = 'dash' if cot == 'sig_mua_giu' else ('dot' if NHOM[cot].startswith('NB1') else 'solid')
        fig.add_trace(go.Scatter(x=d['timestamp'], y=d['equity'], mode='lines',
                                 line=dict(dash=net), name='%s (%s)' % (cot, NHOM[cot])))
    fig.update_layout(title='Equity — NB1 kỹ thuật (chấm) · NB2 AI (liền) · mua-và-giữ (gạch)',
                      height=550, template='plotly_white', yaxis_title='USD')
    fig.show()
'''))

    moi.append(md(r'''
BƯỚC 6: CHẠY BACKTEST MỌI TRƯỜNG HỢP VÀ LẬP BẢNG SO SÁNH

Bảng xếp theo **Sharpe** giảm dần. Chọn chiến lược cần xem báo cáo chi tiết bằng
`MO_HINH_VE` (để trống: tự chọn chiến lược có Sharpe cao nhất, không tính mốc
mua-và-giữ) và chỉ báo vẽ kèm bằng `CHI_BAO_VE`.
'''))
    moi.append(code(r'''
MO_HINH_VE = ''          # ví dụ 'sig_xgb'; để trống → chiến lược có Sharpe cao nhất
CHI_BAO_VE = 'rsi14'     # chỉ báo vẽ kèm, ví dụ 'rsi14' hoặc 'macd_hist'

DV_SWAP = don_vi_swap(df_master['timestamp'], CH)       # tính một lần, dùng cho mọi trường hợp

benchmark_results, results_store = [], {}
for col in signal_columns:
    df_res, trades_res, deals_res = backtest_mt5(df_master, col, CH, DV_SWAP)
    bc = bao_cao_mt5(df_res, trades_res, CH)
    benchmark_results.append({'Tên Mô Hình': col, 'Nhóm': NHOM.get(col, ''), **bc})
    results_store[col] = (df_res, trades_res, deals_res, bc)

df_benchmark = pd.DataFrame(benchmark_results).sort_values('Sharpe Ratio', ascending=False)
cols_order = ['Tên Mô Hình', 'Nhóm', 'Total Net Profit', 'Total Return (%)', 'Profit Factor',
              'Expected Payoff', 'Recovery Factor', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
              'Max Drawdown (%)', 'Win Rate (%)', 'Total Trades', 'Total Commission', 'Total Swap']
print("=" * 92)
print("BẢNG SO SÁNH — %s, lot %s, đòn bẩy 1:%d, spread %d points, commission %.2f USD/lot/chiều"
      % (CH.ky_hieu, CH.lot if CH.che_do_lot == 'co_dinh' else '%.1f%% rủi ro' % CH.rui_ro_phan_tram,
         CH.don_bay, CH.spread_points, CH.hoa_hong_lot_1_chieu))
print("=" * 92)
print(df_benchmark[cols_order].to_string(index=False))
duong_dan_bao_cao = luu_tep(df_benchmark, 'backtest_comparison_report.csv')
'''))

    moi.append(md(r'''
So sánh trực tiếp NB1 và NB2

Mỗi nhánh lấy **trường hợp tốt nhất theo Sharpe** và **trung bình của cả nhóm**,
đặt cạnh mốc mua-và-giữ.
'''))
    moi.append(code(r'''
chi_tieu = ['Total Net Profit', 'Total Return (%)', 'Profit Factor', 'Recovery Factor',
            'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown (%)',
            'Win Rate (%)', 'Total Trades']
nhanh = df_benchmark.assign(Nhánh=df_benchmark['Nhóm'].str.split().str[0])
hang = []
for ten in ['NB1', 'NB2']:
    g = nhanh[nhanh['Nhánh'] == ten]
    tot = g.iloc[0]                                    # đã sắp theo Sharpe
    hang.append({'So sánh': '%s tốt nhất (%s)' % (ten, tot['Tên Mô Hình']),
                 **{k: tot[k] for k in chi_tieu}})
    hang.append({'So sánh': '%s trung bình (%d trường hợp)' % (ten, len(g)),
                 **g[chi_tieu].apply(pd.to_numeric, errors='coerce').mean().round(3).to_dict()})
mg = nhanh[nhanh['Nhánh'] == 'Mốc'].iloc[0]
hang.append({'So sánh': 'Mốc chuẩn mua-và-giữ', **{k: mg[k] for k in chi_tieu}})
print(pd.DataFrame(hang).to_string(index=False))
'''))

    moi.append(md("BƯỚC 7: BÁO CÁO CHI TIẾT, LỊCH SỬ DEAL VÀ BIỂU ĐỒ"))
    moi.append(code(r'''
plot_equity_so_sanh(results_store, signal_columns)

if not MO_HINH_VE:
    MO_HINH_VE = df_benchmark[df_benchmark['Nhóm'] != 'Mốc chuẩn'].iloc[0]['Tên Mô Hình']
df_chart, trades_chart, deals_chart, bc_chart = results_store[MO_HINH_VE]

in_bao_cao_mt5(bc_chart, MO_HINH_VE)
print('\nLịch sử deal (20 dòng đầu) — giống tab History của MT5:')
print(deals_chart.head(20).to_string(index=False))
luu_tep(deals_chart, 'lich_su_deal_%s.csv' % MO_HINH_VE)

plot_mt5(df_chart, MO_HINH_VE)
plot_results(df_chart, trades_chart, MO_HINH_VE, CHI_BAO_VE)
'''))
    moi.append(code(r'''
# Tải bảng so sánh về máy (không bắt buộc nếu đã lưu trên Google Drive)
if TREN_COLAB and not duong_dan_bao_cao.startswith('/content/drive'):
    files.download(duong_dan_bao_cao)
'''))
    g["cells"] = moi
    return g


# ═══════════════════════════════════════════════════════════ xuất
def kiem_tra(nb: dict, ten: str) -> tuple[int, int]:
    """Xác thực như Jupyter: nối source bằng "".join() rồi phân tích cú pháp."""
    ma = md_ = 0
    for i, o in enumerate(nb["cells"]):
        for j, dong in enumerate(o["source"][:-1]):
            assert dong.endswith("\n"), "%s ô %d dòng %d thiếu \\n" % (ten, i, j)
        if o["cell_type"] == "code":
            ast.parse("".join(o["source"]))
            ma += 1
        else:
            md_ += 1
    return ma, md_


def chay():
    for ten, ham in ((NB1, nb1), (NB2, nb2), (NB3, nb3)):
        nb = ham()
        nb.setdefault("metadata", {}).setdefault("colab", {})["provenance"] = []
        ma, md_ = kiem_tra(nb, ten)
        (THU_MUC / ten).write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print("  %-40s %2d ô mã, %2d ô markdown" % (ten, ma, md_))


if __name__ == "__main__":
    chay()
