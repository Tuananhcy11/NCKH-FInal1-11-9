# -*- coding: utf-8 -*-
"""Sinh Google colab/01_Lam_sach_du_lieu.ipynb — làm sạch dữ liệu XAUUSD Dukascopy.

    python src/tao_nb_lam_sach.py
"""
import io
import json
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
DICH = GOC / "Google colab" / "01_Lam_sach_du_lieu.ipynb"


def dong(s):
    s = s.strip("\n").split("\n")
    return [x + "\n" for x in s[:-1]] + [s[-1]]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": dong(s)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dong(s)}


GIOI_THIEU = r'''
# 01 — Làm sạch dữ liệu XAUUSD (Dukascopy 2015–2025)

Chạy **sau** notebook 00 và **trước** NB1, NB2. Bấm **Run all**.

```
00 Tải Dukascopy → 01 Làm sạch → dukascopy_XAUUSD_<khung>_sach.csv → NB1 ∥ NB2 → NB3
```

NB1 và NB2 cùng đọc một tệp đã làm sạch, nên so sánh backtest ở NB3 là công bằng.

| Bước | Nội dung | Cách xử lý |
|---|---|---|
| 1 | Chuẩn hóa ngày giờ | Đưa về UTC không múi giờ, sắp xếp, căn nến về đúng mốc của khung, bỏ nến thứ Bảy, thêm cột `ngay_giao_dich` |
| 2 | Bản ghi trùng lặp | Trùng hoàn toàn → bỏ; cùng thời điểm khác giá → giữ bản đầu, ghi báo cáo |
| 3a | Ô trống / giá không hợp lệ | **Bỏ dòng** (không lấy Close thay O/H/L), chỗ đó thành khoảng trống được gắn cờ |
| 3b | Thị trường đóng cửa | Cuối tuần, nghỉ hằng ngày ~21–22 giờ UTC, ngày lễ → **không điền, chỉ đánh dấu** |
| 3c | Mất nến giữa phiên | **Không điền**, gắn cờ `khoang_trong = 1`, liệt kê trong báo cáo |
| 4 | Nến sai cấu trúc OHLC | Sửa `High = max(O,H,L,C)`, `Low = min(O,H,L,C)`, gắn cờ `da_sua = 1` |
| 6 | Kiểm tra chéo và đầu ra | Gộp khung nhỏ lên khung lớn phải khớp; ghi tệp sạch, báo cáo, biểu đồ |

Không có bước nào **tự tạo ra giá**: dữ liệu chỉ bị bỏ dòng hỏng, sửa râu nến sai
cấu trúc, và thêm các cột đánh dấu để minh bạch.
'''

THIET_LAP = r'''
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.easter import easter
from pandas.tseries.holiday import USFederalHolidayCalendar

try:                                   # Trên Colab: đọc/ghi Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    THU_MUC = '/content/drive/MyDrive/Data_NghienCuu'
except ImportError:                    # Chạy trên máy tính
    THU_MUC = os.environ.get('NCKH_DU_LIEU', 'data/raw')

KHUNG_DS = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
BUOC = {'M1': pd.Timedelta('1min'), 'M5': pd.Timedelta('5min'), 'M15': pd.Timedelta('15min'),
        'M30': pd.Timedelta('30min'), 'H1': pd.Timedelta('1h'), 'H4': pd.Timedelta('4h'),
        'D1': pd.Timedelta('1D')}
COT_GIA = ['open', 'high', 'low', 'close']
COT = COT_GIA + ['volume']
print('Thư mục lưu kết quả:', THU_MUC)
'''

CHON_MD = r'''
## Chọn dữ liệu cần làm sạch

Chọn ở ô dưới (bảng bên phải ô trên Colab):

- **`NGUON = 'tai_len'`** — chạy ô sẽ hiện nút **Choose Files / Chọn tệp**: chọn một hoặc
  nhiều tệp `.csv` / `.xlsx` trên máy tính (giữ Ctrl để chọn nhiều tệp).
- **`NGUON = 'drive'`** — dùng tệp có sẵn trong `MyDrive/Data_NghienCuu`. Gõ tên tệp vào
  `TEN_TEP_DRIVE` (nhiều tệp cách nhau dấu phẩy); để trống thì lấy mọi tệp
  `dukascopy_XAUUSD_*.csv` do notebook 00 tải về. Ô sẽ in danh sách tệp đang có trên Drive.

Tên tệp đặt thế nào cũng được: **khung thời gian tự nhận ra từ dữ liệu**. Tên cột có thể là
`time/Date/Datetime/Gmt time`, `Open/High/Low/Close`, `Volume/Tick volume`… Kết quả lưu vào
Drive thành `<tên gốc>_sach.csv`; bật `TAI_VE_MAY` để tải luôn về máy.
'''

CHON = r'''
#@title Chọn dữ liệu cần làm sạch
NGUON = 'tai_len'        #@param ['tai_len', 'drive']
TEN_TEP_DRIVE = ''       #@param {type:'string'}
TAI_VE_MAY = False       #@param {type:'boolean'}

import glob
DUOI_HOP_LE = ('.csv', '.txt', '.xlsx', '.xls')

def tep_tren_drive():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(THU_MUC, '*'))
                  if p.lower().endswith(DUOI_HOP_LE) and '_sach' not in os.path.basename(p)
                  and not os.path.basename(p).startswith(('bao_cao', 'danh_sach', 'kiem_tra')))

try:
    from google.colab import files as _files
except ImportError:
    _files = None

if NGUON == 'tai_len' and _files is not None:
    thu_muc_tai = '/content/tai_len'
    os.makedirs(thu_muc_tai, exist_ok=True)
    print('Bấm "Choose Files" và chọn tệp dữ liệu vàng cần làm sạch:')
    da_tai = _files.upload()
    TEP_CHON = []
    for ten, noi_dung in da_tai.items():
        p = os.path.join(thu_muc_tai, ten)
        with open(p, 'wb') as f:
            f.write(noi_dung)
        TEP_CHON.append(p)
else:
    if _files is None and os.environ.get('NCKH_TEP'):        # chạy trên máy tính
        TEP_CHON = [p.strip() for p in os.environ['NCKH_TEP'].split(';') if p.strip()]
    else:
        co_san = tep_tren_drive()
        print('Tệp đang có trong %s:' % THU_MUC)
        for x in co_san:
            print('  ·', x)
        if TEN_TEP_DRIVE.strip():
            TEP_CHON = [os.path.join(THU_MUC, x.strip()) for x in TEN_TEP_DRIVE.split(',') if x.strip()]
        else:
            TEP_CHON = [os.path.join(THU_MUC, x) for x in co_san if x.startswith('dukascopy_XAUUSD_')]

thieu = [p for p in TEP_CHON if not os.path.exists(p)]
if thieu:
    raise FileNotFoundError('Không thấy tệp: %s' % thieu)
if not TEP_CHON:
    raise ValueError('Chưa chọn tệp nào để làm sạch.')
print('\nSẽ làm sạch %d tệp:' % len(TEP_CHON))
for p in TEP_CHON:
    print('  ·', os.path.basename(p))
'''

B1_MD = r'''
## Bước 1 — Chuẩn hóa ngày giờ

- Đọc cột `time` ở mọi dạng (có hoặc không kèm múi giờ) và đưa về **UTC, không kèm múi giờ**.
  Dukascopy vốn đã là UTC nên không phải xử lý giờ mùa hè.
- **Căn mốc:** nến M30 phải rơi vào phút :00/:30, H1 vào :00, H4 vào 0/4/8/12/16/20 giờ,
  D1 vào giờ mở phiên (22:00 UTC với Dukascopy — tự nhận ra từ dữ liệu). Nến lệch mốc được
  đưa về mốc đầu nến và đếm vào báo cáo.
- **Bỏ nến thứ Bảy** — thị trường vàng đóng cửa.
- Thêm cột **`ngay_giao_dich`**: phiên vàng mở lúc 22:00–23:00 UTC tối hôm trước, nên nến
  22:00 Chủ nhật thuộc phiên thứ Hai.
'''

B1 = r'''
BI_DANH = {'time': ['time', 'datetime', 'date time', 'timestamp', 'date', 'gmt time', 'local time',
                    'time (utc)', 'open time', 'thoi gian'],
           'open': ['open', 'o'], 'high': ['high', 'h'], 'low': ['low', 'l'],
           'close': ['close', 'c', 'adj close', 'price', 'last'],
           'volume': ['volume', 'vol', 'tick volume', 'tickvol', 'real volume', 'volume ']}


def doc_tep(p):
    if p.lower().endswith(('.xlsx', '.xls')):
        d = pd.read_excel(p)
    else:
        d = pd.read_csv(p)
        if d.shape[1] == 1:                                  # tách bằng tab hoặc ';'
            d = pd.read_csv(p, sep=None, engine='python')
    chuan = {' '.join(str(c).strip().lower().replace('<', ' ').replace('>', ' ').replace('_', ' ').split()): c
             for c in d.columns}
    ra = pd.DataFrame(index=d.index)
    for cot, ds in BI_DANH.items():
        goc = next((chuan[x] for x in ds if x in chuan), None)
        if goc is None and cot == 'volume':
            print('  ⚠ Tệp không có cột volume → volume = 0 (giá không bị ảnh hưởng).')
            ra['volume'] = 0.0
        elif goc is None:
            raise ValueError('Tệp %s không có cột %s. Các cột đang có: %s' % (p, cot, list(d.columns)))
        else:
            ra[cot] = d[goc]
    return ra


def nhan_dien_khung(d):
    t = doc_thoi_gian(d['time']).dropna().sort_values()
    buoc = t.diff().dropna()
    buoc = buoc[buoc > pd.Timedelta(0)].mode().iloc[0]
    return min(BUOC, key=lambda k: abs(np.log(BUOC[k] / buoc)))


def doc_thoi_gian(s):
    """Mọi dạng thời gian → UTC. 02/01/2025 có thể là ngày-trước hoặc tháng-trước: thử cả
    hai, chọn cách đọc được nhiều dòng nhất và cho thời gian tăng dần nhiều nhất."""
    if pd.api.types.is_numeric_dtype(s):                        # số giây / mili-giây Unix
        don_vi = 'ms' if s.abs().median() > 1e11 else 's'
        return pd.to_datetime(s, unit=don_vi, errors='coerce', utc=True)
    s = s.astype(str).str.strip()
    ung_vien = [pd.to_datetime(s, errors='coerce', utc=True, format='mixed')]
    if s.str.contains('/').mean() > 0.5:
        ung_vien.append(pd.to_datetime(s, errors='coerce', utc=True, format='mixed', dayfirst=True))
    diem = lambda t: (t.notna().mean(), (t.diff().dt.total_seconds() > 0).mean())
    return max(ung_vien, key=diem)


def chuan_hoa_thoi_gian(d, khung, nk):
    t = doc_thoi_gian(d['time'])
    d['time'] = t.dt.tz_localize(None).astype('datetime64[ns]')
    nk['thoi_gian_loi'] = int(d['time'].isna().sum())
    d = d.dropna(subset=['time'])

    # Mốc của khung. D1: mốc mở phiên phổ biến nhất (Dukascopy: 22:00 UTC)
    lech = (d['time'] - d['time'].dt.floor('D')).mode().iloc[0] if khung == 'D1' else pd.Timedelta(0)
    moc = (d['time'] - lech).dt.floor(BUOC[khung]) + lech
    nk['lech_moc'] = int((moc != d['time']).sum())
    nk['moc_mo_nen'] = str(lech)[-8:] if khung == 'D1' else ''
    d['time'] = moc
    return d.sort_values('time', kind='stable').reset_index(drop=True), lech


def bo_thu_bay(d, nk):
    thu_bay = d['time'].dt.dayofweek == 5
    nk['thu_bay'] = int(thu_bay.sum())
    return d[~thu_bay].reset_index(drop=True)


def them_ngay_giao_dich(d):
    d['ngay_giao_dich'] = (d['time'] + pd.Timedelta(hours=2)).dt.strftime('%Y-%m-%d')
    return d
'''

B2_MD = r'''
## Bước 2 — Bản ghi trùng lặp

- **Trùng hoàn toàn** (cùng thời điểm, cùng giá, cùng volume): bỏ bản sau.
- **Cùng thời điểm nhưng khác giá:** giữ bản đầu tiên, bỏ bản sau, liệt kê trong báo cáo.

Dòng hỏng được bỏ **trước** bước này (bước 3a), để một bản trùng hỏng không đẩy bản
đúng ra ngoài.
'''

B2 = r'''
def bo_trung(d, nk):
    hoan_toan = d.duplicated(subset=['time'] + COT, keep='first')
    nk['trung_hoan_toan'] = int(hoan_toan.sum())
    d = d[~hoan_toan]
    khac_gia = d.duplicated(subset=['time'], keep='first')
    nk['trung_thoi_gian_khac_gia'] = int(khac_gia.sum())
    ds = d[d['time'].isin(d.loc[khac_gia, 'time'])]
    if len(ds):
        print('  Cùng thời điểm khác giá (giữ dòng đầu):')
        print(ds.head(10).to_string(index=False))
    return d[~khac_gia].reset_index(drop=True)
'''

B3_MD = r'''
## Bước 3 — Dữ liệu khuyết thiếu

**3a. Ô trống hoặc giá không hợp lệ** (thiếu một trong O/H/L/C/V, giá ≤ 0, volume âm):
**bỏ cả dòng**. Không lấy Close thay cho O/H/L vì như vậy là tự tạo giá. Chỗ bị bỏ trở
thành một khoảng trống và được gắn cờ ở 3c.

**3b, 3c. Khoảng trống giữa hai nến liền nhau.** Vàng **không giao dịch 24/7**, nên
khoảng trống được phân loại chứ không điền:

| Loại (`loai_khoang_trong`) | Nhận biết | Xử lý |
|---|---|---|
| `cuoi_tuan` | Khoảng thiếu có chứa ngày thứ Bảy | Chỉ đánh dấu |
| `ngay_le` | Khoảng thiếu chạm ngày lễ: 24–26/12, 31/12–2/1, Thứ Sáu Tuần Thánh, ngày lễ liên bang Mỹ, Thứ Sáu sau Lễ Tạ ơn | Chỉ đánh dấu |
| `nghi_hang_ngay` | Bắt đầu từ 20:00 UTC trở đi và dài không quá 3 giờ | Chỉ đánh dấu |
| `bat_thuong` | Mọi khoảng còn lại (mất nến giữa phiên) | Gắn cờ `khoang_trong = 1`, liệt kê trong báo cáo |

Nhãn được ghi vào **nến ngay sau khoảng trống**, kèm `so_nen_thieu`. **Không chèn nến
giả**: nến phẳng hay nội suy ở giờ đóng cửa sẽ tạo lợi suất bằng 0 không có thật, làm
lệch độ biến động, các chỉ báo và nhãn Triple Barrier.
'''

B3 = r'''
def bo_o_trong(d, nk):
    for c in COT:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    hong = d[COT].isna().any(axis=1) | (d[COT_GIA] <= 0).any(axis=1) | (d['volume'] < 0)
    nk['o_trong_gia_khong_hop_le'] = int(hong.sum())
    return d[~hong].reset_index(drop=True)


def ngay_le(nam_tu, nam_den):
    le = set()
    for y in range(nam_tu, nam_den + 1):
        le |= {date(y, 12, 24), date(y, 12, 25), date(y, 12, 26), date(y, 12, 31),
               date(y, 1, 1), date(y, 1, 2)}
        le.add(easter(y) - timedelta(days=2))                   # Thứ Sáu Tuần Thánh
    lich = USFederalHolidayCalendar().holidays('%d-01-01' % nam_tu, '%d-12-31' % nam_den)
    for x in lich.date:
        le.add(x)
        if x.month == 11 and 22 <= x.day <= 28:                 # Lễ Tạ ơn → cả Thứ Sáu kế tiếp
            le.add(x + timedelta(days=1))
    return le


def phan_loai_khoang(d, khung, le):
    buoc = BUOC[khung]
    t = d['time']
    kc = t.diff()
    loai_cot = np.full(len(d), '', dtype=object)
    so_thieu_cot = np.zeros(len(d), dtype=int)
    ds = []
    for i in np.flatnonzero((kc > buoc).to_numpy()):
        a, b = t.iloc[i - 1] + buoc, t.iloc[i]                  # các mốc bị thiếu: [a, b)
        so_thieu = int((b - a) / buoc)
        cac_ngay = pd.date_range(a.normalize(), (b - pd.Timedelta(seconds=1)).normalize()).date
        if any(x.weekday() == 5 for x in cac_ngay):
            loai = 'cuoi_tuan'
        elif any(x in le for x in cac_ngay):
            loai = 'ngay_le'
        elif a.hour >= 20 and b - a <= pd.Timedelta(hours=3):
            loai = 'nghi_hang_ngay'
        else:
            loai = 'bat_thuong'
        loai_cot[i], so_thieu_cot[i] = loai, so_thieu
        ds.append({'khung': khung, 'nen_truoc': t.iloc[i - 1], 'nen_sau': b,
                   'so_nen_thieu': so_thieu, 'loai': loai})
    d['loai_khoang_trong'] = loai_cot
    d['so_nen_thieu'] = so_thieu_cot
    d['khoang_trong'] = (d['loai_khoang_trong'] == 'bat_thuong').astype(int)
    return d, ds
'''

B4_MD = r'''
## Bước 4 — Nến sai cấu trúc OHLC

Một nến đúng phải có `High ≥ max(Open, Close)` và `Low ≤ min(Open, Close)`. Nến sai được
**sửa chứ không xóa** (xóa sẽ tạo khoảng trống giả):

- `High = max(Open, High, Low, Close)`, `Low = min(Open, High, Low, Close)`
- Gắn cờ `da_sua = 1` và ghi mức sai lớn nhất vào báo cáo.
'''

B4 = r'''
def sua_ohlc(d, nk):
    cao, thap = d[COT_GIA].max(axis=1), d[COT_GIA].min(axis=1)
    sai = (d['high'] < cao) | (d['low'] > thap)
    nk['sai_ohlc_da_sua'] = int(sai.sum())
    nk['sai_ohlc_lon_nhat'] = round(float(max((cao - d['high']).max(), (d['low'] - thap).max(), 0.0)), 5)
    d['high'], d['low'] = cao, thap
    d['da_sua'] = sai.astype(int)
    return d
'''

CHAY_MD = r'''
## Chạy bước 1 → 4 cho từng khung và ghi tệp sạch
'''

CHAY = r'''
LE = ngay_le(2000, 2035)
sach, nhat_ky, ds_khoang, moc_d1, tep_ra = {}, [], [], pd.Timedelta(0), []

for p in TEP_CHON:
    d = doc_tep(p)
    k = nhan_dien_khung(d)
    ten = os.path.splitext(os.path.basename(p))[0]
    print('\n══ %s — khung %s ══' % (os.path.basename(p), k))
    nk = {'khung': k, 'tep': os.path.basename(p)}
    nk['so_dong_vao'] = len(d)
    d, lech = chuan_hoa_thoi_gian(d, k, nk)          # bước 1
    if k == 'D1':
        moc_d1 = lech
    d = bo_o_trong(d, nk)                            # bước 3a (trước bước 2)
    d = bo_trung(d, nk)                              # bước 2
    d = bo_thu_bay(d, nk)                            # bước 1
    d = sua_ohlc(d, nk)                              # bước 4
    d, kh = phan_loai_khoang(d, k, LE)               # bước 3b, 3c
    d = them_ngay_giao_dich(d)                       # bước 1

    for loai in ['cuoi_tuan', 'ngay_le', 'nghi_hang_ngay', 'bat_thuong']:
        nk['khoang_' + loai] = sum(x['loai'] == loai for x in kh)
    nk['so_nen_thieu_bat_thuong'] = sum(x['so_nen_thieu'] for x in kh if x['loai'] == 'bat_thuong')
    nk['so_dong_ra'] = len(d)
    nk['tu'], nk['den'] = str(d['time'].min()), str(d['time'].max())

    assert d['time'].is_monotonic_increasing and d['time'].is_unique
    assert (d['high'] >= d[['open', 'close']].max(axis=1)).all()
    assert (d['low'] <= d[['open', 'close']].min(axis=1)).all()
    assert (d['time'].dt.dayofweek != 5).all()

    ra = d[['time'] + COT + ['ngay_giao_dich', 'loai_khoang_trong', 'so_nen_thieu', 'khoang_trong', 'da_sua']]
    p_ra = os.path.join(THU_MUC, ten + '_sach.csv')
    ra.to_csv(p_ra, index=False, date_format='%Y-%m-%d %H:%M:%S')
    tep_ra.append(p_ra)
    if k in sach:
        print('  ⚠ Đã có một tệp khung %s trước đó — kiểm tra chéo dùng tệp mới nhất.' % k)
    sach[k], ds_khoang = ra, ds_khoang + kh
    nhat_ky.append(nk)
    print('  %d dòng vào → %d dòng ra | %s → %s' % (nk['so_dong_vao'], nk['so_dong_ra'], nk['tu'], nk['den']))
    print('  Bỏ: thời gian lỗi %d, ô trống/giá lỗi %d, trùng %d + %d, thứ Bảy %d | lệch mốc %d | sửa OHLC %d'
          % (nk['thoi_gian_loi'], nk['o_trong_gia_khong_hop_le'], nk['trung_hoan_toan'],
             nk['trung_thoi_gian_khac_gia'], nk['thu_bay'], nk['lech_moc'], nk['sai_ohlc_da_sua']))
    print('  Khoảng trống: cuối tuần %d, ngày lễ %d, nghỉ hằng ngày %d, BẤT THƯỜNG %d (%d nến)'
          % (nk['khoang_cuoi_tuan'], nk['khoang_ngay_le'], nk['khoang_nghi_hang_ngay'],
             nk['khoang_bat_thuong'], nk['so_nen_thieu_bat_thuong']))
    print('  → %s' % p_ra)

if TAI_VE_MAY and _files is not None:           # tải tệp sạch về máy tính
    for p_ra in tep_ra:
        _files.download(p_ra)
'''

B6_MD = r'''
## Bước 6 — Kiểm tra chéo giữa các khung

Cả 4 khung đều gộp từ cùng dữ liệu M1 của Dukascopy, nên gộp khung nhỏ lên khung lớn phải
ra **đúng** khung lớn: High của nến H4 bằng High lớn nhất của 4 nến H1 bên trong, tương
tự với Open, Low, Close, Volume. Nến lệch nghĩa là một trong hai tệp có vấn đề (hoặc vừa bị
bỏ/sửa ở bước trên) — xem bảng dưới.
'''

B6 = r'''
def gop(d, quy_tac, lech):
    g = d.set_index('time')[COT].resample(quy_tac, offset=lech)
    return g.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()


kiem_cheo = []
for nho, lon, quy_tac, lech in [('M30', 'H1', '1h', pd.Timedelta(0)), ('H1', 'H4', '4h', pd.Timedelta(0)),
                                ('H1', 'D1', '24h', moc_d1)]:
    if nho not in sach or lon not in sach:
        continue
    g, l = gop(sach[nho], quy_tac, lech), sach[lon].set_index('time')[COT]
    chung = g.index.intersection(l.index)
    lech_gia = (g.loc[chung, COT_GIA] - l.loc[chung, COT_GIA]).abs()
    lech_vol = (g.loc[chung, 'volume'] - l.loc[chung, 'volume']).abs()
    kiem_cheo.append({
        'gop_tu': nho, 'so_voi': lon, 'nen_chung': len(chung),
        'nen_lech_gia': int((lech_gia > 1e-6).any(axis=1).sum()),
        'lech_gia_lon_nhat': round(float(lech_gia.max().max()), 5) if len(chung) else 0.0,
        'nen_lech_volume': int((lech_vol > 0.01 + 1e-6 * l.loc[chung, 'volume'].abs()).sum()),
        'chi_co_o_' + 'khung_nho': len(g.index.difference(l.index)),
        'chi_co_o_' + 'khung_lon': len(l.index.difference(g.index)),
    })
kiem_cheo = pd.DataFrame(kiem_cheo)
print(kiem_cheo.to_string(index=False) if len(kiem_cheo) else 'Không đủ khung để kiểm tra chéo.')
'''

BAO_CAO_MD = r'''
## Báo cáo làm sạch

- `bao_cao_lam_sach.csv` — mỗi khung một dòng: số dòng vào/ra và số dòng bị ảnh hưởng ở từng bước.
- `danh_sach_khoang_trong.csv` — mọi khoảng trống kèm loại; lọc `loai == 'bat_thuong'` để xem nến mất giữa phiên.
- `kiem_tra_cheo_khung.csv` — kết quả bước 6.
'''

BAO_CAO = r'''
bao_cao = pd.DataFrame(nhat_ky).set_index('tep')
bao_cao.to_csv(os.path.join(THU_MUC, 'bao_cao_lam_sach.csv'))
khoang = pd.DataFrame(ds_khoang, columns=['khung', 'nen_truoc', 'nen_sau', 'so_nen_thieu', 'loai'])
khoang.to_csv(os.path.join(THU_MUC, 'danh_sach_khoang_trong.csv'), index=False)
kiem_cheo.to_csv(os.path.join(THU_MUC, 'kiem_tra_cheo_khung.csv'), index=False)

print(bao_cao.T.to_string())
bt = khoang[khoang['loai'] == 'bat_thuong']
print('\nKhoảng trống bất thường dài nhất:')
print(bt.sort_values('so_nen_thieu', ascending=False).head(15).to_string(index=False) if len(bt) else '  (không có)')
'''

HINH_MD = r'''
## Biểu đồ

1. Số khoảng trống theo loại ở từng khung, và giờ (UTC) xảy ra các khoảng trống bất thường.
2. Giá đóng cửa có đánh dấu nến đã sửa OHLC (đỏ) và nến ngay sau khoảng trống bất thường (cam).
'''

HINH = r'''
loai_ds = ['cuoi_tuan', 'ngay_le', 'nghi_hang_ngay', 'bat_thuong']
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
dem = khoang.groupby(['khung', 'loai']).size().unstack(fill_value=0).reindex(columns=loai_ds, fill_value=0)
dem = dem.reindex([k for k in KHUNG_DS if k in dem.index])
dem.plot.bar(ax=ax[0], rot=0)
ax[0].set_yscale('log'); ax[0].set_title('Số khoảng trống theo loại'); ax[0].set_ylabel('số khoảng (thang log)')
for k in ['M30', 'H1']:
    x = bt[bt['khung'] == k]
    if len(x):
        (x['nen_truoc'] + BUOC[k]).dt.hour.value_counts().sort_index().plot(ax=ax[1], marker='o', label=k)
ax[1].set_title('Khoảng trống bất thường theo giờ bắt đầu (UTC)'); ax[1].set_xlabel('giờ'); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(THU_MUC, 'hinh_lam_sach_khoang_trong.png'), dpi=120); plt.show()

k = 'H1' if 'H1' in sach else next(iter(sach))
d = sach[k]
fig, ax = plt.subplots(figsize=(15, 4.5))
ax.plot(d['time'], d['close'], lw=0.6, color='steelblue', label='Close %s' % k)
x = d[d['da_sua'] == 1]
ax.scatter(x['time'], x['close'], color='red', s=14, zorder=3, label='đã sửa OHLC (%d)' % len(x))
x = d[d['khoang_trong'] == 1]
ax.scatter(x['time'], x['close'], color='orange', s=14, zorder=3, label='sau khoảng trống bất thường (%d)' % len(x))
ax.set_title('XAUUSD %s sau làm sạch' % k); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(THU_MUC, 'hinh_lam_sach_gia.png'), dpi=120); plt.show()
'''

KET_MD = r'''
## Dùng tệp sạch ở NB1 và NB2

Điền đường dẫn tệp sạch, ví dụ

`DUONG_DAN_DU_LIEU = '/content/drive/MyDrive/Data_NghienCuu/dukascopy_XAUUSD_H1_sach.csv'`

NB1/NB2 chỉ đọc 6 cột `time, open, high, low, close, volume`; các cột đánh dấu
(`ngay_giao_dich`, `loai_khoang_trong`, `so_nen_thieu`, `khoang_trong`, `da_sua`) được giữ
trong tệp để tra cứu và trình bày trong báo cáo.
'''

cells = [md(GIOI_THIEU), code(THIET_LAP), md(CHON_MD), code(CHON),
         md(B1_MD), code(B1), md(B2_MD), code(B2), md(B3_MD), code(B3), md(B4_MD), code(B4),
         md(CHAY_MD), code(CHAY), md(B6_MD), code(B6), md(BAO_CAO_MD), code(BAO_CAO),
         md(HINH_MD), code(HINH), md(KET_MD)]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}}

if __name__ == "__main__":
    with io.open(DICH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    lai = json.load(io.open(DICH, encoding="utf-8"))
    for o in lai["cells"]:
        if o["cell_type"] == "code":
            compile("".join(o["source"]), "o", "exec")
    print("Đã ghi %s — %d ô (%d ô mã)" % (DICH.name, len(lai["cells"]),
                                          sum(o["cell_type"] == "code" for o in lai["cells"])))
