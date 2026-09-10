# -*- coding: utf-8 -*-
"""Nội dung các ô của NB3: 9 chiến lược giao dịch trên dự báo xu hướng,
so sánh phân tích kỹ thuật (NB1) và XGBoost walk-forward 2020–2025 (NB2).

sua_notebook_colab.nb3() lắp các chuỗi dưới đây thành notebook. Tách riêng để dễ đọc
và để bài kiểm tra có thể exec đúng mã sẽ chạy trên Colab.
"""

GIOI_THIEU = r'''
# NB3 — Phân tích kỹ thuật và XGBoost: 9 chiến lược, walk-forward 2020–2025

**Câu hỏi nghiên cứu:** cùng một cách giao dịch, dự báo xu hướng của **XGBoost** hay của
**phân tích kỹ thuật** cho kết quả đầu tư tốt hơn — và có ổn định qua **nhiều năm** không?

- NB1 (kỹ thuật) và NB2 (XGBoost) chỉ cho **dự báo xu hướng**: 1 = uptrend · 0 = sideway · −1 = downtrend.
- Dự báo XGBoost là **walk-forward 2020 → 2025**: mỗi năm được dự báo bởi mô hình chỉ học dữ
  liệu **trước** năm đó → 6 năm kiểm tra ngoài mẫu liên tiếp.
- NB3 áp **9 chiến lược** (luật cố định: uptrend / sideway / downtrend thì làm gì + SL, TP,
  R:R) **y hệt** cho 5 nguồn → **45 kịch bản**; chênh lệch chỉ đến từ **chất lượng dự báo**.
- **Đầu ra lệnh chỉ có BUY = 1 và SELL = 0.** Được đánh 2 chiều (tối đa 1 BUY + 1 SELL; trừ S13).

| Nhánh | Nguồn dự báo |
|---|---|
| Phân tích kỹ thuật (NB1) | MA (MA10/MA30) · RSI (30/70) · MACD (cắt tín hiệu) · TH (tổng hợp ≥ 2/3 luật) |
| Mô hình AI (NB2) | **XGB** (XGBoost walk-forward) — thêm RF, LSTM nếu bật ở NB2 |

| Mã | Chiến lược | Uptrend | Sideway | Downtrend | Quản lý lệnh |
|---|---|---|---|---|---|
| S03 | Position theo trend | BUY | không vào | SELL | SL 3 % · 1:3 |
| S06 | Trend xác nhận 3 nến | BUY khi 3 nến liền uptrend | không vào | SELL khi 3 nến liền downtrend | SL 1 % · 1:2 |
| S07 | Bắt đầu xu hướng | BUY khi vừa chuyển sang uptrend | không vào | SELL khi vừa chuyển sang downtrend | SL 1 % · 1:2 |
| S08 | Vào lệnh khi giá hồi | BUY nếu Close < MA20 | không vào | SELL nếu Close > MA20 | SL 1 % · 1:2 |
| S09 | Trend + đánh vùng | BUY (1 % · 1:2) | BUY ở BB dưới · SELL ở BB trên (0,3 % · 1:1,5) | SELL (1 % · 1:2) | theo trạng thái |
| S12 | Đánh ngược dự báo (kiểm chứng) | SELL | không vào | BUY | SL 1 % · 1:2 |
| S13 | Luôn theo trend, đảo chiều | BUY (đang SELL → đóng rồi BUY) | giữ lệnh | SELL (đang BUY → đóng rồi SELL) | SL 2 %, không TP, không hedge |
| S14 | Trend + dời SL về hòa vốn | BUY | không vào | SELL | SL 1 % · TP 3 %; lãi 1R → SL về giá vào |
| S15 | Trend + SL kéo theo giá | BUY | không vào | SELL | SL 1 % kéo theo giá tốt nhất, không TP |

**Mốc chuẩn B0:** mua ở nến đầu, giữ tới cuối (lot 0,10). **Đối chứng B1:** mỗi nguồn được
**xáo trộn ngẫu nhiên** (giữ nguyên tỉ lệ uptrend/sideway/downtrend) rồi chạy lại — nguồn có
năng lực dự báo thật phải **thắng ngẫu nhiên**.
'''

IMPORT = r'''
import os, io, json, time
import warnings
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
'''

CAU_HINH_MD = r'''
BƯỚC 0: THÔNG SỐ BACKTEST (giống Strategy Tester của MT5)

| Nhóm | Thông số | Mặc định |
|---|---|---|
| Tài khoản | số dư, đòn bẩy, stop-out | 10.000 USD · 1:100 · 50 % |
| Symbol XAUUSD | 1 lot, 1 point, spread | 100 oz · 0,01 USD · 30 points (0,30 USD) |
| Chi phí | commission, swap | 3,5 USD/lot/chiều · swap 0 (**điền theo sàn**) |
| Khối lượng | rủi ro mỗi lệnh | **1 % số dư** → lot = rủi ro ÷ (khoảng cách SL × 100 oz), làm tròn xuống 0,01 |
| Giữ 2 chiều | `ty_le_ky_quy_doi_ung` | 1,0 = tính đủ ký quỹ cho cả 2 lệnh (thận trọng) |

**Khớp lệnh:** dự báo chốt khi nến `t` đóng → lệnh khớp ở giá **mở** nến `t + 1`. Giá dữ
liệu là **Bid**, **Ask = Bid + spread**: BUY mua ở Ask, đóng ở Bid; SELL bán ở Bid, đóng ở
Ask. SL/TP tính theo **% giá vào lệnh**; một nến chạm cả SL lẫn TP → **SL khớp trước**;
giá mở nhảy qua SL/TP → khớp ở giá mở.
'''

CAU_HINH = r'''
@dataclass
class CauHinhMT5:
    # ── Tài khoản
    so_du_ban_dau: float = 10_000.0      # Initial deposit (USD)
    don_bay: int = 100                   # Leverage 1:100
    stop_out_phan_tram: float = 50.0     # margin level ≤ 50 % → đóng cưỡng bức lệnh lỗ nhất
    # ── Symbol XAUUSD
    ky_hieu: str = 'XAUUSD'
    kich_thuoc_hop_dong: float = 100.0   # 1 lot = 100 oz
    point: float = 0.01
    spread_points: int = 30              # 0,30 USD
    lot_min: float = 0.01
    lot_max: float = 10.0
    lot_step: float = 0.01
    # ── Chi phí
    hoa_hong_lot_1_chieu: float = 3.5
    swap_long_points: float = 0.0        # ĐIỀN THEO SÀN (points/lot/đêm)
    swap_short_points: float = 0.0       # ĐIỀN THEO SÀN
    gio_qua_dem_utc: int = 22
    ngay_swap_x3: int = 2                # Thứ Tư ×3
    # ── Khối lượng và giữ 2 chiều
    rui_ro_phan_tram: float = 1.0        # % số dư rủi ro mỗi lệnh
    ty_le_ky_quy_doi_ung: float = 1.0    # ký quỹ phần lệnh đối ứng khi giữ 2 chiều

    @property
    def usd_moi_point_moi_lot(self):
        return self.kich_thuoc_hop_dong * self.point


CH = CauHinhMT5()

# Kỳ hạn giao dịch: SL theo % giá vào lệnh, TP = SL × R:R
KY_HAN = {
    'scalping': dict(sl=0.3, rr=1.5),    # ngắn hạn  (S09 khi đánh vùng sideway)
    'swing':    dict(sl=1.0, rr=2.0),    # trung hạn (S06, S07, S08, S09, S12)
    'position': dict(sl=3.0, rr=3.0),    # dài hạn   (S03)
}
N_NGAU_NHIEN = 50           # số lần xáo trộn cho đối chứng B1 (0 = bỏ qua; 100 → lâu gấp đôi)
HAT_GIONG = 42

for k, v in asdict(CH).items():
    print('  %-22s %s' % (k, v))
for k, v in KY_HAN.items():
    print('  %-9s SL %.2f %% · R:R 1:%.1f · TP %.2f %% · thắng hòa vốn %.1f %%'
          % (k, v['sl'], v['rr'], v['sl'] * v['rr'], 100 / (1 + v['rr'])))
'''

NAP_MD = r'''
BƯỚC 1: NẠP VÀ GHÉP DỰ BÁO CỦA NB1 VÀ NB2

NB1 có dự báo trên **toàn bộ** dữ liệu; NB2 có dự báo **walk-forward** cho các năm kiểm tra
(2020 → 2025). Hai tệp được ghép theo thời gian và **chỉ giữ phần chung**, nên mọi kịch bản
chạy trên đúng cùng giai đoạn. Cột `xh_<nguồn>` là dự báo xu hướng của từng nguồn (1 / 0 / −1).
'''

NAP = r'''
def _doc(ten):
    d = pd.read_csv(tim_tep(ten), encoding='utf-8-sig')
    d.columns = [c.strip().lower() for c in d.columns]
    d['time'] = pd.to_datetime(d['time'])
    return d.sort_values('time').drop_duplicates('time')

ky_thuat = _doc('gold_price_technical_signal.csv')     # NB1
ai = _doc('gold_model_signals.csv')                     # NB2 (walk-forward)

# Nguồn dự báo: mã → (cột trong tệp, nhánh). Nguồn AI lấy theo cột có trong tệp NB2.
NGUON = {'MA': ('ma_signal', 'Kỹ thuật'), 'RSI': ('rsi_signal', 'Kỹ thuật'),
         'MACD': ('macd_signal_final', 'Kỹ thuật'), 'TH': ('signal', 'Kỹ thuật')}
for ma_ng, cot in [('XGB', 'sig_xgb'), ('RF', 'sig_rf'), ('LSTM', 'sig_lstm')]:
    if cot in ai.columns:
        NGUON[ma_ng] = (cot, 'AI')

cot_kt = [v[0] for v in NGUON.values() if v[1] == 'Kỹ thuật']
cot_ai = [v[0] for v in NGUON.values() if v[1] == 'AI']
nb1 = ky_thuat[['time', 'open', 'high', 'low', 'close', 'volume', 'rsi14', 'macd_hist',
                'ma20', 'bb_upper', 'bb_lower'] + cot_kt]
nb2 = ai[['time', 'close'] + [c for c in ['fold'] if c in ai.columns] + cot_ai].rename(columns={'close': 'close_nb2'})
df_master = nb1.merge(nb2, on='time', how='inner').sort_values('time').reset_index(drop=True)
if df_master.empty:
    raise ValueError('NB1 và NB2 không có mốc thời gian chung — kiểm tra lại hai tệp đầu vào.')

lech = (df_master['close'] - df_master['close_nb2']).abs() > 1e-6
print('⚠ %.1f %% số nến có giá đóng cửa KHÁC NHAU giữa NB1 và NB2.' % (100 * lech.mean())
      if lech.mean() > 0.01 else '✓ Giá đóng cửa của NB1 và NB2 khớp nhau: cùng một bộ dữ liệu.')
for ma_ng, (cot, _) in NGUON.items():
    df_master['xh_' + ma_ng] = df_master[cot].fillna(0).astype(int)
df_master = df_master.drop(columns=['close_nb2'] + cot_kt + cot_ai).rename(columns={'time': 'timestamp'})
NAM = df_master['timestamp'].dt.year

print('Giai đoạn backtest chung: %s → %s (%d nến, %d năm)\n'
      % (df_master['timestamp'].min(), df_master['timestamp'].max(), len(df_master), NAM.nunique()))
TEN_XU_HUONG = {1: 'UPTREND', 0: 'SIDEWAY', -1: 'DOWNTREND'}
print('Dự báo xu hướng của từng nguồn (số nến):')
print(pd.DataFrame({ng: df_master['xh_' + ng].map(TEN_XU_HUONG).value_counts() for ng in NGUON})
      .reindex(['UPTREND', 'SIDEWAY', 'DOWNTREND']).fillna(0).astype(int).to_string())
print('\nSố nến dự báo trend (≠ sideway) theo năm:')
print(pd.DataFrame({ng: (df_master['xh_' + ng] != 0).groupby(NAM).sum() for ng in NGUON}).to_string())
'''

CHIEN_LUOC_MD = r'''
BƯỚC 2: 9 CHIẾN LƯỢC — DỰ BÁO XU HƯỚNG → Ý ĐỊNH VÀO LỆNH

`tao_lenh(ma, x, d)` đổi chuỗi dự báo `x` (1 / 0 / −1) thành, ở **mỗi nến**: ý định vào lệnh
(**+1 = BUY**, **−1 = SELL**, 0 = không mở lệnh mới) và SL %, TP % cho lệnh đó. Mọi điều
kiện chỉ dùng thông tin **đã có khi nến đóng** (dự báo, Close, MA20, Bollinger của chính
nến đó), lệnh khớp ở nến sau — không nhìn trước tương lai.
'''

CHIEN_LUOC = r'''
SC, SW, PO = KY_HAN['scalping'], KY_HAN['swing'], KY_HAN['position']
CHIEN_LUOC = {
    'S03': dict(ten='Position theo trend',          nhom='Theo xu hướng',    quan_ly='co_dinh', **PO),
    'S06': dict(ten='Trend xác nhận 3 nến',         nhom='Lọc tín hiệu',     quan_ly='co_dinh', **SW),
    'S07': dict(ten='Bắt đầu xu hướng',             nhom='Lọc tín hiệu',     quan_ly='co_dinh', **SW),
    'S08': dict(ten='Vào lệnh khi giá hồi (MA20)',  nhom='Lọc tín hiệu',     quan_ly='co_dinh', **SW),
    'S09': dict(ten='Trend + đánh vùng sideway',    nhom='Có đánh sideway',  quan_ly='co_dinh', **SW),
    'S12': dict(ten='Đánh ngược dự báo',            nhom='Kiểm chứng',       quan_ly='co_dinh', **SW),
    'S13': dict(ten='Luôn theo trend, đảo chiều',   nhom='Quản lý lệnh',     quan_ly='dao_chieu', sl=2.0, rr=0.0),
    'S14': dict(ten='Trend + dời SL về hòa vốn',    nhom='Quản lý lệnh',     quan_ly='hoa_von', sl=1.0, rr=3.0),
    'S15': dict(ten='Trend + SL kéo theo giá',      nhom='Quản lý lệnh',     quan_ly='trailing', sl=1.0, rr=0.0, keo=1.0),
}


def _lui(a, k, dien):
    """Dịch mảng k nến về sau (giá trị của nến t−k đặt tại t)."""
    r = np.empty_like(a)
    r[:k] = dien
    r[k:] = a[:-k]
    return r


def tao_lenh(ma, x, d):
    """Dự báo xu hướng x (1/0/-1) → (ý định vào lệnh +1/-1/0, SL %, TP %) ở mỗi nến."""
    x = np.nan_to_num(np.asarray(x, dtype=float)).astype(int)
    n, cl = len(x), CHIEN_LUOC[ma]
    c = d['close'].to_numpy(float)
    ma20 = d['ma20'].to_numpy(float)
    bb_tren, bb_duoi = d['bb_upper'].to_numpy(float), d['bb_lower'].to_numpy(float)
    sl, rr = np.full(n, cl['sl']), np.full(n, cl['rr'])

    if ma in ('S03', 'S13', 'S14', 'S15'):
        y = x.copy()                                             # uptrend → BUY, downtrend → SELL
    elif ma == 'S06':                                            # 3 nến liền cùng dự báo
        y = np.where((x != 0) & (x == _lui(x, 1, 9)) & (x == _lui(x, 2, 9)), x, 0)
    elif ma == 'S07':                                            # vừa chuyển sang trend
        y = np.where((x != 0) & (x != _lui(x, 1, 0)), x, 0)
    elif ma == 'S08':                                            # giá hồi về phía MA20
        y = np.where((x == 1) & (c < ma20), 1, np.where((x == -1) & (c > ma20), -1, 0))
    elif ma == 'S09':                                            # trend + đánh vùng Bollinger khi sideway
        vung = np.where((x == 0) & (c <= bb_duoi), 1, np.where((x == 0) & (c >= bb_tren), -1, 0))
        y = np.where(x != 0, x, vung)
        sl = np.where(x != 0, SW['sl'], SC['sl'])
        rr = np.where(x != 0, SW['rr'], SC['rr'])
    elif ma == 'S12':
        y = -x                                                   # đánh ngược dự báo
    else:
        raise ValueError('Không có chiến lược %s' % ma)
    return y.astype(int), sl.astype(float), (sl * rr).astype(float)


print('%d chiến lược × %d nguồn = %d kịch bản' % (len(CHIEN_LUOC), len(NGUON), len(CHIEN_LUOC) * len(NGUON)))
for ma, cl in CHIEN_LUOC.items():
    print('  %s  %-30s %-17s SL %.1f %% · %s' % (ma, cl['ten'], cl['nhom'], cl['sl'],
          'R:R 1:%.1f' % cl['rr'] if cl['rr'] else 'không TP'))
'''

ENGINE_MD = r'''
BƯỚC 3: ENGINE BACKTEST KIỂU MT5 — ĐÁNH 2 CHIỀU, SL/TP THEO R:R

Mỗi nến, engine làm theo thứ tự thời gian:

1. **Swap qua đêm** cho các lệnh đang mở (22:00 UTC, Thứ Tư ×3).
2. **Mở lệnh ở giá mở** theo ý định của nến trước. Mỗi chiều tối đa 1 lệnh: đang có BUY thì
   bỏ qua ý định BUY mới, nhưng vẫn **mở được SELL** (và ngược lại). Lot theo **1 % rủi ro**;
   không đủ ký quỹ thì bỏ lệnh (*Not enough money*). S13: ý định ngược chiều **đóng lệnh
   cũ rồi đảo chiều**, không giữ 2 chiều.
3. **SL / TP trong nến**, xét riêng từng lệnh (chạm cả hai → SL trước; nhảy giá → khớp ở
   giá mở). Sau đó mới cập nhật SL cho nến sau: **S14** lãi đạt 1R → SL về giá vào;
   **S15** SL kéo theo giá tốt nhất.
4. **Equity, mức ký quỹ, Stop Out**: margin level ≤ 50 % → đóng lệnh đang lỗ nhiều nhất.

Hết dữ liệu mà còn lệnh → đóng ở giá cuối (`het_du_lieu`). Lịch sử lệnh ghi cột
**`lenh`: 1 = BUY, 0 = SELL**, lý do đóng (`tp`, `sl`, `hoa_von`, `trailing`, `dao_chieu`,
`stop_out`, `het_du_lieu`) và kết quả theo **R** (lãi/lỗ chia cho số tiền rủi ro lúc vào).
'''

ENGINE = r'''
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


def backtest(d, y, sl_pct, tp_pct, quan_ly='co_dinh', keo_pct=0.0, ch=None, dv_swap=None,
             lot_co_dinh=None, ghi_lenh=True):
    """y[i]: ý định vào lệnh chốt ở nến i (+1 BUY / -1 SELL / 0), khớp ở giá mở nến i+1.
    Trả về dict: equity, balance theo nến, bảng lệnh và thống kê giữ 2 chiều."""
    ch = ch or CH
    t = d['timestamp'].to_numpy()
    o, h, l, c = (d[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    n = len(d)
    y = np.asarray(y, dtype=int)
    dv_swap = don_vi_swap(t, ch) if dv_swap is None else dv_swap
    sp, hd, lev = ch.spread_points * ch.point, ch.kich_thuoc_hop_dong, ch.don_bay
    r_rui_ro, cm = ch.rui_ro_phan_tram / 100, ch.hoa_hong_lot_1_chieu
    swap_usd = {1: ch.swap_long_points * ch.point * hd, -1: ch.swap_short_points * ch.point * hd}
    doi_ung = ch.ty_le_ky_quy_doi_ung

    balance = ch.so_du_ban_dau
    vt = {1: None, -1: None}
    lenh = []
    eq, bal = np.empty(n), np.empty(n)
    tk = {'so_nen_2_chieu': 0, 'so_lan_2_chieu': 0, 'bo_lenh_ky_quy': 0}
    dem_lenh, dang_2_chieu = 0, False

    def ky_quy(gia):
        m = [p['lot'] * hd * gia / lev for p in vt.values() if p is not None]
        return max(m) + doi_ung * min(m) if len(m) == 2 else (m[0] if m else 0.0)

    def tha_noi(i, gia_bid):
        s = 0.0
        for p in vt.values():
            if p is not None:
                gia = gia_bid if p['huong'] > 0 else gia_bid + sp
                s += (gia - p['gia']) * p['huong'] * p['lot'] * hd + p['swap']
        return s

    def dong(p, i, gia, ly_do):
        nonlocal balance
        loi = (gia - p['gia']) * p['huong'] * p['lot'] * hd
        hh = cm * p['lot']
        balance += loi + p['swap'] - hh
        vt[p['huong']] = None
        if ghi_lenh:
            pnl = loi + p['swap'] - p['hh'] - hh
            lenh.append({'ma_lenh': p['ma'], 'type': 'BUY' if p['huong'] > 0 else 'SELL',
                         'lenh': 1 if p['huong'] > 0 else 0,
                         'entry_time': t[p['i']], 'exit_time': t[i], 'entry_price': p['gia'],
                         'sl_ban_dau': p['sl0'], 'tp': p['tp'], 'exit_price': gia, 'ly_do': ly_do,
                         'lot': p['lot'], 'profit': loi, 'swap': p['swap'],
                         'commission': -(p['hh'] + hh), 'pnl': pnl,
                         'R': pnl / p['rui_ro'] if p['rui_ro'] > 0 else np.nan,
                         'so_nen_giu': i - p['i']})

    def mo(i, huong, sl_p, tp_p):
        nonlocal balance, dem_lenh
        gia = o[i] + sp if huong > 0 else o[i]
        sl = gia * (1 - huong * sl_p / 100) if sl_p > 0 else None
        tp = gia * (1 + huong * tp_p / 100) if tp_p > 0 else None
        if lot_co_dinh is not None:
            lot = lot_co_dinh
        else:
            lot = np.floor(balance * r_rui_ro / (abs(gia - sl) * hd) / ch.lot_step + 1e-9) * ch.lot_step
            lot = float(min(max(lot, ch.lot_min), ch.lot_max))
        cu = ky_quy(o[i])
        m_moi = lot * hd * gia / lev
        m_sau = (max(cu, m_moi) + doi_ung * min(cu, m_moi)) if cu > 0 else m_moi
        if m_sau > balance + tha_noi(i, o[i]):
            tk['bo_lenh_ky_quy'] += 1                     # Not enough money
            return
        hh = cm * lot
        balance -= hh
        dem_lenh += 1
        vt[huong] = {'ma': dem_lenh, 'i': i, 'huong': huong, 'lot': lot, 'gia': gia, 'sl': sl,
                     'sl0': sl, 'tp': tp, 'hh': hh, 'swap': 0.0, 'tot': gia, 'hv': False, 'keo': False,
                     'rui_ro': abs(gia - sl) * lot * hd if sl is not None else 0.0}

    for i in range(n):
        # Nến không có lệnh mở và không có ý định vào lệnh: equity = balance, bỏ qua nhanh
        if vt[1] is None and vt[-1] is None and (i == 0 or y[i - 1] == 0):
            eq[i] = bal[i] = balance
            dang_2_chieu = False
            continue

        # 1. Swap
        if dv_swap[i] > 0:
            for p in vt.values():
                if p is not None:
                    p['swap'] += swap_usd[p['huong']] * p['lot'] * dv_swap[i]

        # 2. Mở lệnh ở giá mở theo ý định của nến trước
        if i > 0 and y[i - 1] != 0:
            huong = int(y[i - 1])
            if quan_ly == 'dao_chieu' and vt[-huong] is not None:
                p = vt[-huong]
                dong(p, i, o[i] if p['huong'] > 0 else o[i] + sp, 'dao_chieu')
            if vt[huong] is None:
                mo(i, huong, sl_pct[i - 1], tp_pct[i - 1])

        # 3. SL / TP trong nến, rồi cập nhật SL cho nến sau
        for huong in (1, -1):
            p = vt[huong]
            if p is None:
                continue
            ly_do_sl = 'trailing' if p['keo'] else ('hoa_von' if p['hv'] else 'sl')
            if huong > 0:
                if p['sl'] is not None and l[i] <= p['sl']:
                    dong(p, i, min(p['sl'], o[i]), ly_do_sl); continue
                if p['tp'] is not None and h[i] >= p['tp']:
                    dong(p, i, max(p['tp'], o[i]), 'tp'); continue
            else:
                if p['sl'] is not None and h[i] + sp >= p['sl']:
                    dong(p, i, max(p['sl'], o[i] + sp), ly_do_sl); continue
                if p['tp'] is not None and l[i] + sp <= p['tp']:
                    dong(p, i, min(p['tp'], o[i] + sp), 'tp'); continue
            if quan_ly == 'hoa_von' and not p['hv']:
                mot_r = abs(p['gia'] - p['sl0'])
                if (huong > 0 and h[i] >= p['gia'] + mot_r) or (huong < 0 and l[i] + sp <= p['gia'] - mot_r):
                    p['sl'], p['hv'] = p['gia'], True
            elif quan_ly == 'trailing':
                if huong > 0:
                    p['tot'] = max(p['tot'], h[i])
                    moi = p['tot'] * (1 - keo_pct / 100)
                    if moi > p['sl']:
                        p['sl'], p['keo'] = moi, True
                else:
                    p['tot'] = min(p['tot'], l[i] + sp)
                    moi = p['tot'] * (1 + keo_pct / 100)
                    if moi < p['sl']:
                        p['sl'], p['keo'] = moi, True

        # 4. Equity và Stop Out (đóng lệnh lỗ nhiều nhất trước)
        if vt[1] is not None or vt[-1] is not None:
            while True:
                e, m = balance + tha_noi(i, c[i]), ky_quy(c[i])
                if m <= 0 or 100 * e / m > ch.stop_out_phan_tram:
                    break
                mo_ds = [p for p in vt.values() if p is not None]
                lo = min(mo_ds, key=lambda p: ((c[i] if p['huong'] > 0 else c[i] + sp) - p['gia']) * p['huong'])
                dong(lo, i, c[i] if lo['huong'] > 0 else c[i] + sp, 'stop_out')
                if vt[1] is None and vt[-1] is None:
                    break
        hai = vt[1] is not None and vt[-1] is not None
        if hai:
            tk['so_nen_2_chieu'] += 1
            if not dang_2_chieu:
                tk['so_lan_2_chieu'] += 1
        dang_2_chieu = hai
        eq[i], bal[i] = balance + tha_noi(i, c[i]), balance

    for p in list(vt.values()):                             # hết dữ liệu
        if p is not None:
            dong(p, n - 1, c[n - 1] if p['huong'] > 0 else c[n - 1] + sp, 'het_du_lieu')
    eq[n - 1] = bal[n - 1] = balance
    return {'equity': eq, 'balance': bal, 'lenh': pd.DataFrame(lenh), 'thong_ke': tk}
'''

BAO_CAO_MD = r'''
BƯỚC 4: CHỈ SỐ ĐÁNH GIÁ

| Chỉ số | Ý nghĩa |
|---|---|
| Net Profit, Return, CAGR, Profit Factor, Sharpe, Sortino, Max DD, Recovery | như báo cáo MT5 (Sharpe/Sortino trên % thay đổi equity mỗi nến, quy năm) |
| Win rate · Win rate hòa vốn | tỉ lệ thắng thực tế so với mức cần để hòa vốn = 1 ÷ (1 + R:R) |
| Kỳ vọng (R) | trung bình mỗi lệnh lãi/lỗ bao nhiêu lần số tiền rủi ro — **> 0 là có lợi thế** |
| Số năm có lãi | trong các năm kiểm tra walk-forward, bao nhiêu năm kết thúc có lãi — **độ ổn định** |
| Cách đóng lệnh · Lãi BUY/SELL · Giữ 2 chiều | cách lệnh kết thúc, lãi tách theo chiều, mức độ hedge |
'''

BAO_CAO = r'''
def _sut_giam(chuoi):
    dinh = np.maximum.accumulate(chuoi)
    pct = (dinh - chuoi) / dinh
    return 100 * float(pct.max()), float((dinh - chuoi).max())


def _nen_moi_nam(ts):
    so_nam = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / (365.25 * 86400), 1e-9)
    return len(ts) / so_nam


def _sharpe(eq, nmn):
    r = np.diff(eq) / eq[:-1]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(nmn)) if sd > 0 else np.nan


def theo_nam(kq, ts, ch=None):
    """Return, Sharpe, Max DD và số lệnh của từng năm dương lịch."""
    ch = ch or CH
    eq = pd.Series(kq['equity'], index=pd.DatetimeIndex(ts))
    nmn = _nen_moi_nam(pd.Series(eq.index))
    ld = kq['lenh']
    so_lenh = pd.to_datetime(ld['entry_time']).dt.year.value_counts() if len(ld) else pd.Series(dtype=int)
    ra, dau = [], ch.so_du_ban_dau
    for nam, e in eq.groupby(eq.index.year):
        v = np.r_[dau, e.to_numpy()]
        ra.append({'Năm': int(nam), 'Return (%)': round(100 * (v[-1] / dau - 1), 2),
                   'Sharpe': round(_sharpe(v, nmn), 3), 'Max DD (%)': round(_sut_giam(v)[0], 2),
                   'Số lệnh': int(so_lenh.get(nam, 0))})
        dau = v[-1]
    return pd.DataFrame(ra)


def bao_cao(kq, ts, rr=0.0, ch=None):
    ch = ch or CH
    von0, eq, bal = ch.so_du_ban_dau, kq['equity'], kq['balance']
    nmn = _nen_moi_nam(ts)
    so_nam = len(eq) / nmn
    r = np.diff(eq) / eq[:-1]
    sd, sd_giam = (r.std(ddof=1) if len(r) > 1 else 0.0), np.sqrt((np.minimum(r, 0) ** 2).mean())
    mdd_pct, mdd_usd = _sut_giam(eq)
    ld = kq['lenh']
    pnl = ld['pnl'] if len(ld) else pd.Series(dtype=float)
    lai, lo = pnl[pnl > 0].sum(), pnl[pnl < 0].sum()
    net = bal[-1] - von0
    dem = ld['ly_do'].value_counts() if len(ld) else pd.Series(dtype=int)
    tk = kq['thong_ke']
    nbuy = int((ld['type'] == 'BUY').sum()) if len(ld) else 0
    tn = theo_nam(kq, ts, ch)
    return {
        'Net Profit': round(net, 2),
        'Return (%)': round(100 * net / von0, 2),
        'CAGR (%)': round(100 * ((max(bal[-1], 1e-9) / von0) ** (1 / so_nam) - 1), 2),
        'Profit Factor': round(lai / -lo, 3) if lo < 0 else np.nan,
        'Sharpe': round(r.mean() / sd * np.sqrt(nmn), 3) if sd > 0 else np.nan,
        'Sortino': round(r.mean() / sd_giam * np.sqrt(nmn), 3) if sd_giam > 0 else np.nan,
        'Max DD (%)': round(mdd_pct, 2),
        'Recovery Factor': round(net / mdd_usd, 3) if mdd_usd > 0 else np.nan,
        'Số năm có lãi': '%d / %d' % ((tn['Return (%)'] > 0).sum(), len(tn)),
        'Số lệnh': len(ld), 'BUY': nbuy, 'SELL': len(ld) - nbuy,
        'Win rate (%)': round(100 * (pnl > 0).mean(), 2) if len(ld) else np.nan,
        'Win rate hòa vốn (%)': round(100 / (1 + rr), 1) if rr > 0 else np.nan,
        'Kỳ vọng (R)': round(ld['R'].mean(), 3) if len(ld) and ld['R'].notna().any() else np.nan,
        **{'Đóng: ' + k: int(dem.get(k, 0)) for k in
           ['tp', 'sl', 'hoa_von', 'trailing', 'dao_chieu', 'stop_out', 'het_du_lieu']},
        'Lãi BUY': round(ld.loc[ld['type'] == 'BUY', 'pnl'].sum(), 2) if len(ld) else 0.0,
        'Lãi SELL': round(ld.loc[ld['type'] == 'SELL', 'pnl'].sum(), 2) if len(ld) else 0.0,
        'Giữ TB (nến)': round(ld['so_nen_giu'].mean(), 1) if len(ld) else np.nan,
        'Số lần 2 chiều': tk['so_lan_2_chieu'],
        '% nến 2 chiều': round(100 * tk['so_nen_2_chieu'] / len(eq), 2),
        'Bỏ lệnh (ký quỹ)': tk['bo_lenh_ky_quy'],
        'Total Commission': round(ld['commission'].sum(), 2) if len(ld) else 0.0,
        'Total Swap': round(ld['swap'].sum(), 2) if len(ld) else 0.0,
    }


def in_bao_cao(bc, ten):
    print('=' * 96)
    print('STRATEGY TESTER REPORT — %s | %s | %s → %s' % (ten, CH.ky_hieu, df_master['timestamp'].min(),
                                                          df_master['timestamp'].max()))
    print('=' * 96)
    k = list(bc.items())
    nua = (len(k) + 1) // 2
    for (a, x), (b, yv) in zip(k[:nua], k[nua:] + [('', '')]):
        print('  %-24s %-14s  %-24s %s' % (a, x, b, yv))
'''

CHAY_MD = r'''
BƯỚC 5: CHẠY 45 KỊCH BẢN VÀ MỐC CHUẨN B0

Mã kịch bản dạng `S03-XGB` (chiến lược S03 trên dự báo XGBoost). Ghi ra:
`backtest_comparison_report.csv` (toàn giai đoạn), `ket_qua_theo_nam.csv` (từng năm),
`lich_su_lenh_tat_ca.csv` (mọi lệnh, cột `lenh` 1 = BUY / 0 = SELL) và
`tin_hieu_lenh_NB3.csv` (ý định vào lệnh ở từng nến: 1 = BUY, 0 = SELL, trống = không vào).
'''

CHAY = r'''
DV_SWAP = don_vi_swap(df_master['timestamp'], CH)
TS = df_master['timestamp']
ket_qua, lich_su, theo_nam_ds, luu = [], [], [], {}
tin_hieu = pd.DataFrame({'timestamp': TS})
t0 = time.time()

def _ghi(kb, ma, ten, nhom, ng, nhanh, kq, rr):
    ket_qua.append({'Kịch bản': kb, 'Chiến lược': ma, 'Tên chiến lược': ten, 'Nhóm': nhom,
                    'Nguồn': ng, 'Nhánh': nhanh, **bao_cao(kq, TS, rr)})
    theo_nam_ds.append(theo_nam(kq, TS).assign(**{'Kịch bản': kb, 'Chiến lược': ma, 'Nguồn': ng, 'Nhánh': nhanh}))
    luu[kb] = kq

for ma, cl in CHIEN_LUOC.items():
    for ng in NGUON:
        y, slp, tpp = tao_lenh(ma, df_master['xh_' + ng].to_numpy(), df_master)
        kq = backtest(df_master, y, slp, tpp, cl['quan_ly'], cl.get('keo', 0.0), CH, DV_SWAP)
        kb = '%s-%s' % (ma, ng)
        _ghi(kb, ma, cl['ten'], cl['nhom'], ng, NGUON[ng][1], kq, cl['rr'])
        if len(kq['lenh']):
            lich_su.append(kq['lenh'].assign(kich_ban=kb))
        tin_hieu[kb] = pd.Series(y).map({1: 1, -1: 0}).values        # 1 = BUY, 0 = SELL, trống = không vào
    print('  %s xong (%.0f giây)' % (ma, time.time() - t0))

# Mốc chuẩn B0: mua ở nến đầu, giữ tới cuối, lot cố định 0,10, không SL/TP
y0 = np.zeros(len(df_master), dtype=int); y0[0] = 1
kq0 = backtest(df_master, y0, np.zeros(len(y0)), np.zeros(len(y0)), ch=CH, dv_swap=DV_SWAP, lot_co_dinh=0.10)
_ghi('B0', 'B0', 'Mua và giữ', 'Mốc chuẩn', '—', 'Mốc chuẩn', kq0, 0.0)

df_kq = pd.DataFrame(ket_qua)
df_nam = pd.concat(theo_nam_ds, ignore_index=True)
lich_su_lenh = pd.concat(lich_su, ignore_index=True) if lich_su else pd.DataFrame()
print('\n%d kịch bản xong sau %.0f giây.' % (len(df_kq), time.time() - t0))
cot_xem = ['Kịch bản', 'Nhánh', 'Net Profit', 'CAGR (%)', 'Profit Factor', 'Sharpe', 'Max DD (%)',
           'Số năm có lãi', 'Số lệnh', 'Win rate (%)', 'Win rate hòa vốn (%)', 'Kỳ vọng (R)']
print('\nToàn bộ kịch bản, xếp theo Sharpe:')
print(df_kq.sort_values('Sharpe', ascending=False)[cot_xem].to_string(index=False))
'''

NGAU_NHIEN_MD = r'''
BƯỚC 6: ĐỐI CHỨNG B1 — DỰ BÁO NGẪU NHIÊN

Với mỗi nguồn và mỗi chiến lược, chuỗi dự báo được **xáo trộn ngẫu nhiên** `N_NGAU_NHIEN`
lần (giữ nguyên số nến uptrend / sideway / downtrend) rồi backtest lại. **Phân vị** của
kịch bản thật = bao nhiêu % lần ngẫu nhiên có Sharpe thấp hơn. Phân vị ≥ 95 % → dự báo
tốt hơn ngẫu nhiên một cách đáng tin. Mất khoảng 10 phút với `N_NGAU_NHIEN = 50`; đặt 0 ở
Bước 0 để bỏ qua.
'''

NGAU_NHIEN = r'''
df_kq['Phân vị Sharpe (%)'] = np.nan
df_kq['Phân vị Net (%)'] = np.nan
ngau_nhien = pd.DataFrame()
if N_NGAU_NHIEN > 0:
    rng = np.random.default_rng(HAT_GIONG)
    nmn = _nen_moi_nam(TS)
    t0 = time.time()
    hang = []
    for ma, cl in CHIEN_LUOC.items():
        for ng in NGUON:
            x = df_master['xh_' + ng].to_numpy()
            sh, net = [], []
            for _ in range(N_NGAU_NHIEN):
                y, slp, tpp = tao_lenh(ma, rng.permutation(x), df_master)
                kq = backtest(df_master, y, slp, tpp, cl['quan_ly'], cl.get('keo', 0.0), CH, DV_SWAP,
                              ghi_lenh=False)
                sh.append(_sharpe(kq['equity'], nmn)); net.append(kq['balance'][-1] - CH.so_du_ban_dau)
            sh, net = np.nan_to_num(np.array(sh), nan=-np.inf), np.array(net)
            dong_kq = df_kq['Kịch bản'] == '%s-%s' % (ma, ng)
            that_sh = df_kq.loc[dong_kq, 'Sharpe'].iloc[0]
            that_net = df_kq.loc[dong_kq, 'Net Profit'].iloc[0]
            df_kq.loc[dong_kq, 'Phân vị Sharpe (%)'] = round(100 * np.mean(sh < (that_sh if pd.notna(that_sh) else -np.inf)), 1)
            df_kq.loc[dong_kq, 'Phân vị Net (%)'] = round(100 * np.mean(net < that_net), 1)
            hang.append({'Chiến lược': ma, 'Nguồn': ng,
                         'Sharpe ngẫu nhiên TB': np.nanmean(np.where(np.isfinite(sh), sh, np.nan)),
                         'Net ngẫu nhiên TB': net.mean(), 'Net ngẫu nhiên P95': np.percentile(net, 95)})
        print('  %s xong (%.0f giây)' % (ma, time.time() - t0))
    ngau_nhien = pd.DataFrame(hang)
    luu_tep(ngau_nhien, 'ket_qua_ngau_nhien.csv')
    print('\nSố chiến lược THẮNG NGẪU NHIÊN (phân vị Sharpe ≥ 95 %) theo nguồn:')
    thang = (df_kq[df_kq['Chiến lược'] != 'B0'].groupby('Nguồn')['Phân vị Sharpe (%)']
             .apply(lambda s: '%d / %d' % ((s >= 95).sum(), s.notna().sum())))
    print(thang.reindex(list(NGUON)).to_string())
else:
    print('Bỏ qua đối chứng ngẫu nhiên (N_NGAU_NHIEN = 0).')

duong_dan_bao_cao = luu_tep(df_kq, 'backtest_comparison_report.csv')
luu_tep(df_nam, 'ket_qua_theo_nam.csv')
luu_tep(lich_su_lenh, 'lich_su_lenh_tat_ca.csv')
luu_tep(tin_hieu, 'tin_hieu_lenh_NB3.csv')
'''

SO_SANH_MD = r'''
BƯỚC 7: SO SÁNH PHÂN TÍCH KỸ THUẬT VÀ XGBOOST

1. **Bảng chéo chiến lược × nguồn** (Sharpe, CAGR) trên toàn giai đoạn.
2. **Từng năm:** Return của mỗi kịch bản theo năm — nguồn nào ổn định qua các năm.
3. **Đối đầu trực tiếp XGBoost với từng luật kỹ thuật** trên **9 chiến lược × các năm**
   (mỗi cặp = cùng chiến lược, cùng năm): XGB thắng bao nhiêu cặp, và **kiểm định Wilcoxon**
   ghép cặp (p < 0,05 → khác biệt có ý nghĩa thống kê, không phải may rủi).
4. **Nhánh kỹ thuật (tốt nhất / trung bình 4 luật) và nhánh AI** theo từng chiến lược.
'''

SO_SANH = r'''
ds_ng = list(NGUON)
KT = [ng for ng in ds_ng if NGUON[ng][1] == 'Kỹ thuật']
AI_ = [ng for ng in ds_ng if NGUON[ng][1] == 'AI']
ten_cl = {ma: '%s %s' % (ma, cl['ten']) for ma, cl in CHIEN_LUOC.items()}
kq_cl = df_kq[df_kq['Chiến lược'] != 'B0']
bang_sharpe = kq_cl.pivot(index='Chiến lược', columns='Nguồn', values='Sharpe')[ds_ng].rename(index=ten_cl)
bang_cagr = kq_cl.pivot(index='Chiến lược', columns='Nguồn', values='CAGR (%)')[ds_ng].rename(index=ten_cl)


def hien_bang(b, tieu_de, dinh_dang='{:.2f}'):
    print('\n' + tieu_de)
    try:
        display(b.style.background_gradient(cmap='RdYlGn', axis=None).format(dinh_dang))
    except Exception:
        print(b.round(2).to_string())


hien_bang(bang_sharpe, 'SHARPE toàn giai đoạn — chiến lược × nguồn (kỹ thuật: %s | AI: %s)' % (' '.join(KT), ' '.join(AI_)))
hien_bang(bang_cagr, 'CAGR (%/năm) toàn giai đoạn — chiến lược × nguồn')
luu_tep(bang_sharpe.reset_index(), 'bang_sharpe_chien_luoc_x_nguon.csv')

# ── 2. Từng năm
nam_cl = df_nam[df_nam['Chiến lược'] != 'B0']
bang_nam = nam_cl.pivot_table(index=['Chiến lược', 'Nguồn'], columns='Năm', values='Return (%)')
bang_nam = bang_nam.reindex([(ma, ng) for ma in CHIEN_LUOC for ng in ds_ng])
bang_nam['Số năm lãi'] = (bang_nam > 0).sum(axis=1)
hien_bang(bang_nam, 'RETURN (%) TỪNG NĂM — mỗi dòng một kịch bản', '{:.1f}')
b0_nam = df_nam[df_nam['Chiến lược'] == 'B0'].set_index('Năm')['Return (%)']
print('\nMua và giữ (B0) theo năm: ' + ' | '.join('%d: %+.1f %%' % kv for kv in b0_nam.items()))

# ── 3. Đối đầu XGBoost (và AI khác nếu có) với từng luật kỹ thuật, trên cặp (chiến lược, năm)
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

def _p(dd):
    dd = dd[dd != 0]
    if wilcoxon is None or len(dd) < 6:
        return np.nan
    return wilcoxon(dd).pvalue

doi_dau = []
ret = nam_cl.pivot_table(index=['Chiến lược', 'Năm'], columns='Nguồn', values='Return (%)')
shp = nam_cl.pivot_table(index=['Chiến lược', 'Năm'], columns='Nguồn', values='Sharpe').fillna(0)  # năm không giao dịch → 0
for a in AI_:
    for k in KT:
        d_ret, d_shp = (ret[a] - ret[k]).dropna(), (shp[a] - shp[k]).dropna()
        d_toan = (bang_sharpe[a] - bang_sharpe[k]).dropna()
        doi_dau.append({'Cặp': '%s vs %s' % (a, k),
                        'Số chiến lược so được': len(d_toan),       # bỏ chiến lược mà một bên không có lệnh
                        'Thắng Sharpe toàn kỳ': int((d_toan > 0).sum()),
                        'Số cặp (chiến lược, năm)': len(d_ret),
                        'Thắng Return từng năm': int((d_ret > 0).sum()),
                        'Chênh Return TB (điểm %)': round(d_ret.mean(), 2),
                        'p Wilcoxon (Return năm)': round(_p(d_ret), 4),
                        'Thắng Sharpe từng năm': int((d_shp > 0).sum()),
                        'p Wilcoxon (Sharpe năm)': round(_p(d_shp), 4)})
doi_dau = pd.DataFrame(doi_dau)
print('\nĐỐI ĐẦU: %s với từng luật kỹ thuật (cặp = cùng chiến lược, cùng năm)' % ', '.join(AI_))
print(doi_dau.to_string(index=False))
print('p < 0,05 → khác biệt có ý nghĩa thống kê; p ≥ 0,05 → chưa đủ bằng chứng.')
luu_tep(doi_dau, 'doi_dau_xgb_vs_ky_thuat.csv')

# ── 4. Nhánh kỹ thuật và nhánh AI theo từng chiến lược
hang = []
for ma in CHIEN_LUOC:
    s = bang_sharpe.loc[ten_cl[ma]]
    hang.append({'Chiến lược': ten_cl[ma],
                 'KT tốt nhất': '%s %.2f' % (s[KT].idxmax(), s[KT].max()) if s[KT].notna().any() else '—',
                 'AI tốt nhất': '%s %.2f' % (s[AI_].idxmax(), s[AI_].max()) if s[AI_].notna().any() else '—',
                 'Sharpe KT max': s[KT].max(), 'Sharpe KT TB': s[KT].mean(),
                 'Sharpe AI max': s[AI_].max(), 'Sharpe AI TB': s[AI_].mean()})
so_sanh = pd.DataFrame(hang)
so_sanh['Thắng (so KT tốt nhất)'] = np.where(so_sanh['Sharpe AI max'] > so_sanh['Sharpe KT max'], 'AI', 'Kỹ thuật')
so_sanh['Thắng (so KT trung bình)'] = np.where(so_sanh['Sharpe AI max'] > so_sanh['Sharpe KT TB'], 'AI', 'Kỹ thuật')
print('\nTỪNG CHIẾN LƯỢC: nhánh Kỹ thuật và nhánh AI (Sharpe toàn giai đoạn)')
print(so_sanh.round(3).to_string(index=False))
print('\nAI thắng %d / %d chiến lược khi so với luật kỹ thuật TỐT NHẤT, %d / %d khi so với TRUNG BÌNH 4 luật.'
      % ((so_sanh['Thắng (so KT tốt nhất)'] == 'AI').sum(), len(so_sanh),
         (so_sanh['Thắng (so KT trung bình)'] == 'AI').sum(), len(so_sanh)))
luu_tep(so_sanh, 'so_sanh_ky_thuat_vs_ai.csv')

b0 = df_kq[df_kq['Kịch bản'] == 'B0'].iloc[0]
print('\nMốc chuẩn mua và giữ: Net %.2f USD | CAGR %.2f %% | Sharpe %.3f | Max DD %.2f %%'
      % (b0['Net Profit'], b0['CAGR (%)'], b0['Sharpe'], b0['Max DD (%)']))
'''

HINH_MD = r'''
BƯỚC 8: BIỂU ĐỒ VÀ BÁO CÁO CHI TIẾT MỘT KỊCH BẢN

- Bản đồ nhiệt Sharpe; Return từng năm của `CHIEN_LUOC_VE` theo nguồn.
- Equity của `CHIEN_LUOC_VE` trên cả 5 nguồn và mốc mua-và-giữ.
- Báo cáo kiểu MT5, lịch sử lệnh và biểu đồ khớp lệnh của `KICH_BAN_VE`
  (để trống: kịch bản XGBoost có Sharpe cao nhất).
'''

HINH = r'''
CHIEN_LUOC_VE = ''      # ví dụ 'S13'; để trống → chiến lược của kịch bản được chọn
KICH_BAN_VE = ''        # ví dụ 'S13-XGB'; để trống → kịch bản AI có Sharpe cao nhất
CHI_BAO_VE = 'rsi14'

fig = go.Figure(go.Heatmap(z=bang_sharpe.values, x=bang_sharpe.columns, y=bang_sharpe.index,
                           colorscale='RdYlGn', zmid=0, text=np.round(bang_sharpe.values, 2),
                           texttemplate='%{text}'))
fig.update_layout(title='Sharpe toàn giai đoạn: chiến lược × nguồn', height=520, template='plotly_white',
                  yaxis_autorange='reversed')
fig.show()

ung_vien = df_kq[df_kq['Nhánh'] == 'AI'] if (df_kq['Nhánh'] == 'AI').any() else df_kq[df_kq['Kịch bản'] != 'B0']
KICH_BAN_VE = KICH_BAN_VE or ung_vien.sort_values('Sharpe', ascending=False).iloc[0]['Kịch bản']
CHIEN_LUOC_VE = CHIEN_LUOC_VE or KICH_BAN_VE.split('-')[0]

d = df_nam[df_nam['Chiến lược'].isin([CHIEN_LUOC_VE, 'B0'])]
fig = go.Figure()
for ng in ds_ng + ['—']:
    x = d[d['Nguồn'] == ng]
    fig.add_bar(x=x['Năm'], y=x['Return (%)'], name='B0 mua và giữ' if ng == '—' else ng)
fig.update_layout(title='Return từng năm — %s' % ten_cl[CHIEN_LUOC_VE], barmode='group', height=480,
                  template='plotly_white', yaxis_title='%')
fig.show()

fig = go.Figure()
for ng in ds_ng:
    kb = '%s-%s' % (CHIEN_LUOC_VE, ng)
    fig.add_scatter(x=TS, y=luu[kb]['equity'], mode='lines', name=kb,
                    line=dict(dash='dot' if NGUON[ng][1] == 'Kỹ thuật' else 'solid',
                              width=1 if NGUON[ng][1] == 'Kỹ thuật' else 2.5))
fig.add_scatter(x=TS, y=luu['B0']['equity'], mode='lines', name='B0 mua và giữ', line=dict(dash='dash', color='black'))
fig.update_layout(title='Equity — %s (kỹ thuật: chấm · AI: liền)' % ten_cl[CHIEN_LUOC_VE],
                  height=550, template='plotly_white', yaxis_title='USD')
fig.show()

kq = luu[KICH_BAN_VE]
in_bao_cao(df_kq[df_kq['Kịch bản'] == KICH_BAN_VE].iloc[0].drop(['Kịch bản']).to_dict(), KICH_BAN_VE)
print('\nTheo năm:')
print(df_nam[df_nam['Kịch bản'] == KICH_BAN_VE].drop(columns=['Kịch bản', 'Chiến lược', 'Nguồn', 'Nhánh']).to_string(index=False))
ld = kq['lenh']
print('\nLịch sử lệnh (20 dòng đầu) — cột lenh: 1 = BUY, 0 = SELL:')
print(ld.head(20).round(3).to_string(index=False) if len(ld) else '(không có lệnh)')

df_chart = df_master[['timestamp', 'open', 'high', 'low', 'close', 'rsi14', 'macd_hist']].copy()
df_chart['equity'] = kq['equity']
plot_results(df_chart, ld if len(ld) else pd.DataFrame(columns=['type']), KICH_BAN_VE, CHI_BAO_VE)
'''

TAI_VE = r'''
# Tải bảng so sánh về máy (không bắt buộc nếu đã lưu trên Google Drive)
if TREN_COLAB and not duong_dan_bao_cao.startswith('/content/drive'):
    files.download(duong_dan_bao_cao)
'''
