# -*- coding: utf-8 -*-
"""Sinh notebook Colab — hệ thống trading hai chiều XAU/USD, bản nâng cấp v2.

    python src/build_notebook.py

Vì sao sinh notebook bằng script thay vì viết tay .ipynb:

  · CHUẨN .ipynb — mỗi phần tử của mảng "source" là MỘT dòng và phải kết thúc
    bằng "\\n", trừ phần tử cuối. Jupyter nối bằng "".join() chứ không phải
    "\\n".join(). Thiếu ký tự xuống dòng thì cả ô bị gộp thành một dòng và
    notebook hỏng trong MỌI môi trường.

  · Hàm kiem_tra() dưới đây xác thực bằng "".join() — đúng cách Jupyter làm.
    Trình kiểm tra tự viết hay dùng "\\n".join() và do đó che mất chính lỗi
    cần tìm.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
RA = GOC / "notebooks" / "XAUUSD_Dual_Direction_Model.ipynb"


def md(chu: str) -> dict:
    d = chu.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [x + "\n" for x in d[:-1]] + [d[-1]]}


def code(ma: str) -> dict:
    d = ma.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [x + "\n" for x in d[:-1]] + [d[-1]]}


def cac_o() -> list[dict]:
    o = []

    # ════════════════════════════════════════════════════ mở đầu
    o.append(md(r'''
# Hệ thống trading hai chiều XAU/USD — bản v2

Nâng cấp từ v1 sau khi chạy thật trên 65.125 nến H1 (2015–2025) và **thua lỗ
−4.281 USD**. Bản này sửa đúng ba nguyên nhân đã đo được.

## Ba lỗi của v1 và cách sửa

| Lỗi đo được ở v1 | Nguyên nhân | Cách sửa ở v2 |
|---|---|---|
| **Overtrading** — 2.052 lệnh, spread bào mòn vốn | Ngưỡng 0,55 quá thấp, không lọc phiên | Ngưỡng 0,65 / 0,68 + **bộ lọc ba lớp** |
| **Short lỗ −4.996 USD** trong khi Long lãi +715 | Mô hình đánh ngược xu hướng tăng dài hạn của vàng | **Bộ lọc HTF** — cấm Short khi giá trên EMA400 |
| Chốt lời quá ít cơ hội | Rào thời gian 8 nến quá chật cho TP = 1,5·ATR | Nới lên **16 nến**, hòa vốn sớm tại **0,8R** |

## Ma trận đa khung thời gian

Ba tầng thông tin, mỗi tầng trả lời một câu hỏi khác nhau:

| Tầng | Đặc trưng | Câu hỏi |
|---|---|---|
| **Dài hạn** (Macro) | EMA 400 / EMA 800 trên M15 ≈ H4 / Daily | Xu hướng nền nghiêng về đâu? |
| **Trung hạn** (Swing) | FVG, khoảng cách EMA chuẩn hóa ATR, BB/Keltner squeeze | Cấu trúc đang căng hay giãn? |
| **Ngắn hạn** (Micro) | Biên phiên Á, tỷ lệ râu nến, mã hóa chu kỳ giờ | Vi cấu trúc phiên hiện tại? |

## Bộ lọc ba lớp — lõi của bản nâng cấp

Một tín hiệu chỉ được thực thi khi vượt **cả ba** cửa:

```
Lớp 1  Xác suất ML   P(Long) >= 0,65   hoặc   P(Short) >= 0,68
Lớp 2  Xu hướng HTF  Long  chỉ khi giá TRÊN  EMA400
                     Short chỉ khi giá DƯỚI  EMA400
Lớp 3  Phiên         Chỉ mở lệnh 07:00–18:00 UTC (London + New York)
```

**Ngưỡng Short cao hơn Long (0,68 vs 0,65) là cố ý.** Vàng có xu hướng tăng dài
hạn, nên một mô hình trung lập sẽ tự nhiên sinh quá nhiều tín hiệu Short sai.
Đặt ngưỡng bất đối xứng là cách thừa nhận tính bất đối xứng của thị trường.

> **Cảnh báo về lọc quá tay.** Bộ lọc HTF biến hệ thống thành thuần thuận xu
> hướng: nó không còn kiếm được tiền ở điểm đảo chiều, và số lệnh Short sẽ giảm
> rất mạnh trên dữ liệu vàng. Mục 7 đo **mức thiệt hại của từng lớp lọc** để
> biết chính xác đã đánh đổi những gì — lọc bớt lệnh xấu và lọc mất lệnh tốt
> trông giống hệt nhau nếu chỉ nhìn tổng lợi nhuận.
'''))

    o.append(md("## 0. Cài đặt và cấu hình"))

    o.append(code(r'''
import subprocess, sys

def _cai(*goi):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *goi], check=False)

try:
    import lightgbm  # noqa: F401
except ImportError:
    _cai("lightgbm")

# Thư viện `ta` cài để tiện đối chiếu, nhưng MỌI chỉ báo trong notebook này đều
# tự cài đặt bằng pandas/numpy. Lý do: `ta` đổi API giữa các phiên bản, và một
# notebook nghiên cứu không nên hỏng vì phụ thuộc ngoài đổi chữ ký hàm.
try:
    import ta  # noqa: F401
except ImportError:
    _cai("ta")

print("Đã sẵn sàng.")
'''))

    o.append(code(r'''
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 80)
plt.rcParams.update({"figure.figsize": (13, 5), "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})

HAT_GIONG = 42
np.random.seed(HAT_GIONG)


@dataclass
class CauHinh:
    """Toàn bộ tham số hệ thống. Sửa ở đây, không rải số ma thuật trong code."""

    # ── Nguồn dữ liệu ────────────────────────────────────────────
    duong_dan: str | None = None       # None → sinh dữ liệu mô phỏng
    khung: str = "M15"                 # "M15" hoặc "H1"
    so_nen_mo_phong: int = 40_000
    gia_khoi_diem: float = 1900.0

    # ── Thị trường và chi phí ────────────────────────────────────
    spread: float = 0.30               # 30 points = 0,30 USD/oz
    oz_moi_lot: float = 100.0          # 1 lot XAU/USD = 100 oz

    # ── Quản trị vốn ─────────────────────────────────────────────
    von_ban_dau: float = 10_000.0
    rui_ro_moi_lenh: float = 0.005     # 0,5 % vốn = 50 USD ở mức vốn ban đầu
    lot_toi_thieu: float = 0.01
    lot_toi_da: float = 10.0

    # ── Rào cản Triple Barrier ───────────────────────────────────
    he_so_tp: float = 1.5              # TP = 1,5 × ATR(14)
    he_so_sl: float = 1.0              # SL = 1,0 × ATR(14)
    rao_thoi_gian: int = 16            # v1 dùng 8 — quá chật cho TP = 1,5·ATR
    kich_hoat_hoa_von: float = 0.8     # v1 dùng 1,0 — dời SL về hòa vốn sớm hơn

    # ── Bộ lọc ba lớp ────────────────────────────────────────────
    # Lớp 1: ngưỡng xác suất. Short khắt khe hơn Long vì vàng có xu hướng tăng
    # dài hạn, khiến mô hình trung lập sinh quá nhiều tín hiệu Short sai.
    nguong_mua: float = 0.65
    nguong_ban: float = 0.68
    # Lớp 2: EMA đại diện khung lớn, tính ngay trên chuỗi hiện tại
    chu_ky_htf: int = 400              # ≈ H4 nếu dữ liệu là M15
    chu_ky_macro: int = 800            # ≈ Daily
    bat_loc_htf: bool = True
    # Lớp 3: khung giờ thanh khoản cao (London mở → New York chiều)
    gio_mo_utc: int = 7
    gio_dong_utc: int = 18
    bat_loc_phien: bool = True

    # ── Phiên Á theo UTC (23:00 hôm trước → 07:00) = 06:00–14:00 giờ VN
    phien_a_bat_dau: int = 23
    phien_a_ket_thuc: int = 7

    # ── Mô hình ──────────────────────────────────────────────────
    so_fold: int = 5


CH = CauHinh()
print(pd.Series(asdict(CH)).to_string())
'''))

    # ═══════════════════════════════════════════ 1. Data engine
    o.append(md(r'''
## 1. `ForexDataEngine` — tải hoặc sinh dữ liệu

Nếu `duong_dan` là `None` hoặc tệp không tồn tại, lớp này sinh dữ liệu mô phỏng
mang bốn đặc tính thật của giá vàng:

- **Biến động theo phiên** — Á trầm, London sôi động, chồng lấn Âu–Mỹ
  (13:00–16:00 UTC) mạnh nhất.
- **Cụm biến động** — biến động hôm nay phụ thuộc hôm qua (GARCH giản lược).
- **Đuôi dày** — cú nhảy do tin vĩ mô, không phải phân phối chuẩn thuần túy.
- **Nghỉ cuối tuần** — bỏ thứ Bảy và Chủ nhật, tạo khoảng nhảy đầu tuần.

Nến OHLC dựng từ **đường đi trong nến** (8 bước con) chứ không bịa high/low
quanh close. Bịa sẽ làm sai quan hệ giữa biên độ nến và biến động thật, khiến
mọi kiểm định chạm rào cản sau đó đều lệch.

**Đọc tệp thật:** chỉ lấy đúng 6 cột `datetime, open, high, low, close, volume`.
Nếu tệp có sẵn cột đặc trưng hoặc nhãn từ pipeline khác, chúng bị **bỏ qua** —
nhận đặc trưng dựng sẵn từ nguồn ngoài là thừa hưởng luôn mọi rò rỉ của nguồn đó
mà không có cách nào kiểm chứng.
'''))

    o.append(code(r'''
class ForexDataEngine:
    """Tải dữ liệu OHLCV thật, hoặc sinh dữ liệu mô phỏng khi chưa có tệp."""

    COT = ["open", "high", "low", "close", "volume"]

    def __init__(self, ch: CauHinh):
        self.ch = ch
        self.buoc_phut = 15 if ch.khung.upper() == "M15" else 60

    # ── công khai ────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        p = self.ch.duong_dan
        if p:
            from pathlib import Path as _P
            if _P(p).exists():
                print("Đọc dữ liệu thật:", p)
                return self._chuan_hoa(self._doc_tep(p))
            print("Không thấy tệp %r — chuyển sang dữ liệu mô phỏng." % p)
        print("Sinh dữ liệu mô phỏng (%s nến %s)."
              % (format(self.ch.so_nen_mo_phong, ","), self.ch.khung))
        return self._chuan_hoa(self._sinh_mo_phong())

    # ── đọc tệp thật ─────────────────────────────────────────────
    def _doc_tep(self, p: str) -> pd.DataFrame:
        d = pd.read_parquet(p) if str(p).endswith((".parquet", ".pq")) \
            else pd.read_csv(p)
        d.columns = [str(c).strip().lower() for c in d.columns]
        cot_tg = next((c for c in ("datetime", "time", "date", "timestamp")
                       if c in d.columns), None)
        if cot_tg is None:
            raise ValueError("Không tìm thấy cột thời gian trong %s" % list(d.columns))
        d = d.rename(columns={cot_tg: "datetime"})
        if "volume" not in d.columns:
            # Nhiều nguồn Forex không có volume thật; đặt 1 thay vì bỏ cột để
            # các đặc trưng phía sau không phải rẽ nhánh.
            d["volume"] = 1.0
        thua = [c for c in d.columns if c not in ["datetime"] + self.COT]
        if thua:
            print("  bỏ qua %d cột không phải OHLCV: %s%s"
                  % (len(thua), ", ".join(thua[:6]), " ..." if len(thua) > 6 else ""))
        return d[["datetime"] + self.COT]

    # ── sinh dữ liệu mô phỏng ────────────────────────────────────
    def _he_so_phien(self, gio: np.ndarray) -> np.ndarray:
        """Bội số biến động theo giờ UTC."""
        hs = np.full(len(gio), 0.55)                # phiên Á: trầm
        hs[(gio >= 7) & (gio < 13)] = 1.00          # London
        hs[(gio >= 13) & (gio < 16)] = 1.45         # chồng lấn Âu-Mỹ
        hs[(gio >= 16) & (gio < 21)] = 1.10         # New York
        hs[(gio >= 21) & (gio < 23)] = 0.45         # cuối phiên Mỹ
        return hs

    def _sinh_mo_phong(self) -> pd.DataFrame:
        ch, rng = self.ch, np.random.default_rng(HAT_GIONG)
        n = ch.so_nen_mo_phong

        # Lưới thời gian bỏ cuối tuần: sinh dư rồi lọc cho đủ n nến
        moc = pd.date_range("2019-01-01", periods=int(n * 1.6),
                            freq="%dmin" % self.buoc_phut, tz="UTC")
        moc = moc[moc.dayofweek < 5][:n]

        gio = moc.hour.to_numpy()
        hs = self._he_so_phien(gio)

        # Biến động nền cộng thành phần dai dẳng (GARCH giản lược): biến động
        # hôm nay phụ thuộc hôm qua, tạo cụm biến động như thật
        sigma_nam = 0.16
        buoc_nam = (self.buoc_phut / 60.0) / (24.0 * 252.0)
        sigma_nen = sigma_nam * np.sqrt(buoc_nam)
        dai_dang = np.empty(n)
        dai_dang[0] = 1.0
        eps = rng.normal(0, 0.12, n)
        for i in range(1, n):
            dai_dang[i] = 0.94 * dai_dang[i - 1] + 0.06 + eps[i]
        dai_dang = np.clip(dai_dang, 0.45, 2.6)

        sigma = sigma_nen * hs * dai_dang
        loi_suat = rng.normal(0.0, 1.0, n) * sigma

        # Xu hướng nền rất nhẹ. Cố ý giữ gần 0: dữ liệu mô phỏng có xu hướng
        # mạnh sẽ khiến mô hình thiên vị Long một cách giả tạo, và bài toán hai
        # chiều mất ý nghĩa vì chiều Short không bao giờ có cơ hội.
        loi_suat += 3e-6

        # Cú nhảy do tin vĩ mô — nguồn của đuôi dày
        nhay = rng.random(n) < 0.0012
        loi_suat[nhay] += rng.normal(0, 6.0, nhay.sum()) * sigma[nhay]

        # Khoảng nhảy đầu tuần
        dau_tuan = np.r_[False, np.diff(moc.dayofweek.to_numpy()) < 0]
        loi_suat[dau_tuan] += rng.normal(0, 3.0, dau_tuan.sum()) * sigma[dau_tuan]

        dong = ch.gia_khoi_diem * np.exp(np.cumsum(loi_suat))
        mo = np.r_[ch.gia_khoi_diem, dong[:-1]]

        # High/Low dựng từ ĐƯỜNG ĐI trong nến (8 bước con)
        b = 8
        buoc_con = rng.normal(0, 1, (n, b)) * (sigma / np.sqrt(b))[:, None]
        duong_di = mo[:, None] * np.exp(np.cumsum(buoc_con, axis=1))
        cao = np.maximum(duong_di.max(axis=1), np.maximum(mo, dong))
        thap = np.minimum(duong_di.min(axis=1), np.minimum(mo, dong))

        klg = (rng.gamma(2.4, 380, n) * hs).round()
        return pd.DataFrame({"datetime": moc, "open": mo, "high": cao,
                             "low": thap, "close": dong, "volume": klg})

    # ── chuẩn hóa và kiểm định ───────────────────────────────────
    def _chuan_hoa(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["datetime"] = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
        d = d.dropna(subset=["datetime"]).sort_values("datetime")
        d = d.drop_duplicates("datetime", keep="last").set_index("datetime")
        d = d[self.COT].astype(float)

        n0 = len(d)
        d = d.dropna(subset=["open", "high", "low", "close"])
        d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
        # Chỉ loại bản ghi vi phạm CẤU TRÚC. Không cắt biên giá trị cực đoan:
        # một cú nhảy 3 % của vàng hoàn toàn có thể là phản ứng thật với tin vĩ
        # mô, và cắt nó đi là xóa mất chính thứ mô hình cần học.
        hop_le = ((d["high"] >= d[["open", "close"]].max(axis=1)) &
                  (d["low"] <= d[["open", "close"]].min(axis=1)) &
                  (d["high"] >= d["low"]))
        d = d[hop_le]

        print("  %s → %s | %s nến (loại %d bản ghi lỗi cấu trúc)"
              % (d.index.min(), d.index.max(), format(len(d), ","), n0 - len(d)))
        return d
'''))

    o.append(code(r'''
engine = ForexDataEngine(CH)
df = engine.load()
display(df.head())
print()
print(df.describe().T[["mean", "std", "min", "max"]].to_string())
'''))


    # ═══════════════════════════════════ 2. Feature extractor
    o.append(md(r'''
## 2. `ForexFeatureExtractor` — ma trận ba khung thời gian

Trả về **hai** thứ:

- `X` — ma trận đặc trưng đưa vào mô hình, **toàn bộ đã `.shift(1)`**
- `ngu_canh` — trạng thái xu hướng HTF và giờ UTC, dùng cho bộ lọc lớp 2 và 3

Tách riêng là chủ ý: bộ lọc HTF **không** phải đặc trưng của mô hình mà là luật
thực thi đặt **sau** mô hình. Gộp chung sẽ khiến mô hình học đè lên luật, và khi
đó không còn phân biệt được ưu thế đến từ mô hình hay từ luật.

### Ba tầng thời gian

| Tầng | Đặc trưng | Ý nghĩa |
|---|---|---|
| **Dài hạn** | `dist_htf_atr`, `dist_macro_atr`, `htf_macro_atr`, `trend_tang` | EMA400 ≈ H4, EMA800 ≈ Daily trên dữ liệu M15 |
| **Trung hạn** | `fvg_atr`, `dist_ema20_atr`, `dist_ema50_atr`, `squeeze_ty_le` | Cấu trúc swing và độ căng biến động |
| **Ngắn hạn** | `a_kc_dinh_atr`, `rau_tren_than`, `gio_sin/cos` | Vi cấu trúc phiên |

### Vì sao chuẩn hóa mọi khoảng cách theo ATR

`Close − EMA20` tính bằng USD có phân phối hoàn toàn khác giữa vàng ở 1.050 và
vàng ở 4.546 — đúng biên độ của bộ dữ liệu thật 2015–2025. Chia cho ATR biến nó
thành đại lượng **không thứ nguyên**, so sánh được qua mọi mức giá và mọi chế độ
biến động.

### Fair Value Gap

FVG là khoảng bất cân xứng ba nến: `low[t] > high[t−2]` là gap tăng chưa được
lấp, `high[t] < low[t−2]` là gap giảm. Đo bằng bội số ATR, dấu dương cho gap
tăng và âm cho gap giảm.

### Bollinger / Keltner squeeze

Tỷ lệ `độ_rộng_BB / độ_rộng_Keltner`. Nhỏ hơn 1 nghĩa là Bollinger nằm gọn trong
Keltner — biến động đang bị nén, thường đi trước một cú bung. Dùng **tỷ lệ liên
tục** thay vì cờ nhị phân để mô hình học được cả mức độ nén.

### Chống rò rỉ

Toàn bộ đặc trưng `.shift(1)`. Ngoại lệ hợp lệ duy nhất là nhóm thời gian — giờ
và thứ được biết trước cả khi nến bắt đầu.

Biên phiên Á chỉ công bố **từ 07:00 UTC**, khi phiên đã đóng. Công bố sớm hơn là
nhìn tương lai: lúc 01:00 chưa thể biết đỉnh của cả phiên kéo dài tới 07:00.
'''))

    o.append(code(r'''
class ForexFeatureExtractor:
    """Sinh đặc trưng ba tầng thời gian, đối xứng Long/Short, đã .shift(1)."""

    # Nhóm thời gian là ngoại lệ hợp lệ duy nhất KHÔNG cần dịch: giờ và thứ của
    # nến t được biết trước cả khi nến bắt đầu.
    COT_THOI_GIAN = ["gio_sin", "gio_cos", "thu", "phien_au", "phien_my",
                     "chong_lan"]

    def __init__(self, ch: CauHinh):
        self.ch = ch
        self.ten_dac_trung: list[str] = []

    # ── chỉ báo cơ sở ────────────────────────────────────────────
    @staticmethod
    def ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def atr(h, l, c, n: int = 14) -> pd.Series:
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                       axis=1).max(axis=1)
        return tr.ewm(alpha=1 / n, adjust=False).mean()

    @staticmethod
    def rsi(c: pd.Series, n: int = 14) -> pd.Series:
        d = c.diff()
        len_ = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        xuong = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + len_ / xuong.replace(0, np.nan))

    @classmethod
    def macd(cls, c: pd.Series, nhanh=12, cham=26, tin_hieu=9):
        duong = cls.ema(c, nhanh) - cls.ema(c, cham)
        tin = cls.ema(duong, tin_hieu)
        return duong, tin, duong - tin

    # ── phiên Á ─────────────────────────────────────────────────
    def _bien_do_phien_a(self, d: pd.DataFrame) -> pd.DataFrame:
        """High/Low phiên Á, CHỈ công bố từ khi phiên đóng (07:00 UTC)."""
        ch = self.ch
        gio = d.index.hour
        # Phiên Á vắt qua nửa đêm nên gán mã phiên theo mốc lùi 23 giờ
        ma_phien = (d.index - pd.Timedelta(hours=ch.phien_a_bat_dau)).normalize()
        trong_phien = (gio >= ch.phien_a_bat_dau) | (gio < ch.phien_a_ket_thuc)

        tam = pd.DataFrame({"ma": ma_phien, "h": d["high"], "l": d["low"]})
        gom = tam[trong_phien].groupby("ma")
        ra = pd.DataFrame(index=d.index)
        ra["a_cao"] = pd.Series(ma_phien, index=d.index).map(gom["h"].max())
        ra["a_thap"] = pd.Series(ma_phien, index=d.index).map(gom["l"].min())
        # Che toàn bộ thời gian còn TRONG phiên — chưa được phép biết
        ra.loc[trong_phien, ["a_cao", "a_thap"]] = np.nan
        return ra

    # ── xây dựng ────────────────────────────────────────────────
    def transform(self, d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ch = self.ch
        o_, h, l, c, v = (d["open"], d["high"], d["low"], d["close"], d["volume"])
        X = pd.DataFrame(index=d.index)

        atr14 = self.atr(h, l, c, 14)
        atr_an = atr14.replace(0, np.nan)

        # ══════════════ TẦNG DÀI HẠN — Macro proxy ══════════════
        ema_htf = self.ema(c, ch.chu_ky_htf)         # ≈ H4 nếu dữ liệu M15
        ema_macro = self.ema(c, ch.chu_ky_macro)     # ≈ Daily
        X["dist_htf_atr"] = (c - ema_htf) / atr_an
        X["dist_macro_atr"] = (c - ema_macro) / atr_an
        X["htf_macro_atr"] = (ema_htf - ema_macro) / atr_an
        # Nhãn xu hướng: giá trên EMA nhanh VÀ EMA nhanh trên EMA chậm
        trend_tang = (c > ema_htf) & (ema_htf > ema_macro)
        trend_giam = (c < ema_htf) & (ema_htf < ema_macro)
        X["trend_tang"] = trend_tang.astype(float)
        X["trend_giam"] = trend_giam.astype(float)
        # Độ dốc EMA macro: xu hướng nền đang mạnh lên hay yếu đi
        X["doc_htf_atr"] = (ema_htf - ema_htf.shift(50)) / atr_an

        # ══════════════ TẦNG TRUNG HẠN — Swing dynamics ═════════
        for n in (20, 50):
            X["dist_ema%d_atr" % n] = (c - self.ema(c, n)) / atr_an
        X["ema20_50_atr"] = (self.ema(c, 20) - self.ema(c, 50)) / atr_an

        # Fair Value Gap: khoảng bất cân xứng ba nến, chuẩn hóa theo ATR.
        # Dương = gap tăng chưa lấp, âm = gap giảm, 0 = không có gap.
        gap_tang = (l - h.shift(2)).clip(lower=0)
        gap_giam = (l.shift(2) - h).clip(lower=0)
        X["fvg_atr"] = (gap_tang - gap_giam) / atr_an
        X["fvg_co"] = ((gap_tang > 0) | (gap_giam > 0)).astype(float)

        # Bollinger / Keltner squeeze — tỷ lệ liên tục, không phải cờ nhị phân
        giua = c.rolling(20).mean()
        sd = c.rolling(20).std()
        rong_bb = 4 * sd                              # BB(20, 2): trên − dưới
        rong_kc = 4 * atr14                           # Keltner(20, 2)
        X["squeeze_ty_le"] = rong_bb / rong_kc.replace(0, np.nan)
        X["bb_vitri"] = (c - (giua - 2 * sd)) / rong_bb.replace(0, np.nan)
        X["bb_rong_atr"] = rong_bb / atr_an

        # Động lượng chuẩn hóa
        X["rsi14"] = self.rsi(c, 14) / 100.0
        _, _, hist = self.macd(c)
        X["macd_hist_atr"] = hist / atr_an
        for n in (5, 20, 60):
            X["roc%d_atr" % n] = (c - c.shift(n)) / atr_an

        # ══════════════ TẦNG NGẮN HẠN — Microstructure ══════════
        than = (c - o_).abs().replace(0, np.nan)
        bien_do = (h - l).replace(0, np.nan)
        X["rau_tren_than"] = (h - np.maximum(o_, c)) / than
        X["rau_duoi_than"] = (np.minimum(o_, c) - l) / than
        X["than_bien_do"] = (c - o_).abs() / bien_do
        X["huong_nen"] = np.sign(c - o_)
        X["bien_do_atr"] = (h - l) / atr_an
        X["gap_atr"] = (o_ - c.shift(1)) / atr_an

        pa = self._bien_do_phien_a(d)
        X["a_bien_do_atr"] = (pa["a_cao"] - pa["a_thap"]) / atr_an
        X["a_kc_dinh_atr"] = (c - pa["a_cao"]) / atr_an
        X["a_kc_day_atr"] = (c - pa["a_thap"]) / atr_an
        X["a_pha_vo"] = np.where(c > pa["a_cao"], 1.0,
                                 np.where(c < pa["a_thap"], -1.0, 0.0))

        # Biến động và khối lượng
        X["atr_pct"] = atr14 / c
        X["atr_z"] = (atr14 - atr14.rolling(200).mean()) / \
            atr14.rolling(200).std().replace(0, np.nan)
        X["kl_hang"] = v.rolling(200, min_periods=50).rank(pct=True)

        # ══════════════ Mã hóa chu kỳ ═══════════════════════════
        # Dùng `hour` thô thì 23 giờ và 0 giờ cách nhau 23 đơn vị dù thực tế
        # liền kề. Cặp sin/cos ánh xạ 24 giờ lên đường tròn nên khoảng cách
        # phản ánh đúng khoảng cách thời gian. Cần CẢ HAI hàm: chỉ dùng sin
        # thì 6 giờ và 18 giờ trùng giá trị.
        gio_le = d.index.hour + d.index.minute / 60.0
        X["gio_sin"] = np.sin(2 * np.pi * gio_le / 24)
        X["gio_cos"] = np.cos(2 * np.pi * gio_le / 24)
        X["thu"] = d.index.dayofweek.astype(float)
        X["phien_au"] = ((d.index.hour >= 7) & (d.index.hour < 13)).astype(float)
        X["phien_my"] = ((d.index.hour >= 13) & (d.index.hour < 21)).astype(float)
        X["chong_lan"] = ((d.index.hour >= 13) & (d.index.hour < 16)).astype(float)

        X = X.replace([np.inf, -np.inf], np.nan)

        # ══════════════ CHỐNG RÒ RỈ: dịch một nến ═══════════════
        cot_dich = [k for k in X.columns if k not in self.COT_THOI_GIAN]
        X[cot_dich] = X[cot_dich].shift(1)

        # ── Ngữ cảnh cho bộ lọc lớp 2 và lớp 3 ──────────────────
        # Cũng dịch một nến để giữ nguyên nguyên tắc: mọi quyết định chỉ dựa
        # trên thông tin có tại mốc mở nến.
        ngu_canh = pd.DataFrame(index=d.index)
        ngu_canh["tren_htf"] = (c > ema_htf).shift(1).fillna(False).astype(bool)
        ngu_canh["trend_tang"] = trend_tang.shift(1).fillna(False).astype(bool)
        ngu_canh["trend_giam"] = trend_giam.shift(1).fillna(False).astype(bool)
        ngu_canh["gio_utc"] = d.index.hour

        self.ten_dac_trung = list(X.columns)
        return X, ngu_canh

    # ── tự kiểm tra rò rỉ ───────────────────────────────────────
    def kiem_tra_ro_ri(self, X: pd.DataFrame, d: pd.DataFrame,
                       nguong: float = 0.10) -> pd.DataFrame:
        """So tương quan của mỗi đặc trưng với lợi suất nến HIỆN TẠI.

        Đặc trưng hợp lệ chỉ biết tới nến t−1 nên tương quan với lợi suất nến t
        phải gần 0. Giá trị cao bất thường nghĩa là cột đó chưa được dịch.
        """
        r = np.log(d["close"] / d["close"].shift(1))
        b = pd.DataFrame([{"dac_trung": k,
                           "tq_voi_loi_suat_hien_tai": round(float(X[k].corr(r)), 4)}
                          for k in X.columns])
        b["canh_bao"] = np.where(b["tq_voi_loi_suat_hien_tai"].abs() > nguong,
                                 "NGHI NGỜ", "")
        b = b.sort_values("tq_voi_loi_suat_hien_tai", key=abs, ascending=False)
        n = int((b["canh_bao"] != "").sum())
        print("Kiểm tra rò rỉ: %d/%d đặc trưng vượt ngưỡng |r| > %.2f"
              % (n, len(b), nguong))
        if n:
            print(b.head(n).to_string(index=False))
        return b
'''))

    o.append(code(r'''
fe = ForexFeatureExtractor(CH)
X_all, ngu_canh = fe.transform(df)
print("Số đặc trưng:", X_all.shape[1])
print()
_ = fe.kiem_tra_ro_ri(X_all, df)
print()
print("Ngữ cảnh cho bộ lọc:")
print("  giá trên EMA%d : %.1f %% số nến"
      % (CH.chu_ky_htf, 100 * ngu_canh["tren_htf"].mean()))
print("  xu hướng tăng   : %.1f %%" % (100 * ngu_canh["trend_tang"].mean()))
print("  xu hướng giảm   : %.1f %%" % (100 * ngu_canh["trend_giam"].mean()))
print("  trong phiên %02d–%02d UTC: %.1f %%"
      % (CH.gio_mo_utc, CH.gio_dong_utc,
         100 * ((ngu_canh["gio_utc"] >= CH.gio_mo_utc) &
                (ngu_canh["gio_utc"] < CH.gio_dong_utc)).mean()))
'''))

    # ═══════════════════════════════════════════ 3. Labeler
    o.append(md(r'''
## 3. `ForexTripleBarrierLabeler` — nhãn ba lớp hai chiều

Tại **mỗi** nến, dựng **hai kịch bản độc lập** rồi xem kịch bản nào thắng trước.

| | Long | Short |
|---|---|---|
| Vào lệnh | Ask = Close + spread | Bid = Close |
| TP | Entry + 1,5·ATR | Entry − 1,5·ATR |
| SL | Entry − 1,0·ATR − spread | Entry + 1,0·ATR + spread |
| Rào thời gian | 16 nến | 16 nến |

Kiểm TP/SL trên **giá đúng chiều**: Long thoát ở Bid (dùng `high`/`low` nguyên
bản), Short thoát ở Ask (dùng `high + spread` / `low + spread`).

### Hai quy ước bảo thủ, cố ý

**Trong cùng một nến, giả định SL đến trước.** Dữ liệu OHLC không cho biết
`high` hay `low` xảy ra trước. Giả định ngược lại sẽ thổi phồng tỷ lệ thắng một
cách có hệ thống — và sai lệch đó không bao giờ lộ ra vì backtest cũng dùng
chính dữ liệu đó.

**Cùng thắng thì gán 0.** Nếu cả Long lẫn Short đều chạm TP trong cửa sổ, giá đã
quét cả hai phía — vào chiều nào cũng có thể bị quét trước. Đây là môi trường
nên đứng ngoài, không phải cơ hội.
'''))

    o.append(code(r'''
class ForexTripleBarrierLabeler:
    """Gán nhãn 3 lớp (+1 / -1 / 0) bằng Triple Barrier hai chiều, có spread."""

    def __init__(self, ch: CauHinh):
        self.ch = ch

    @staticmethod
    def _quet(cao, thap, i0, n_toi_da, tp, sl, la_long, spread):
        """Trả về (chạm_tp, chạm_sl, chỉ_số_thoát) cho MỘT kịch bản.

        Long  thoát ở Bid → so trực tiếp với high/low.
        Short thoát ở Ask → so với high+spread / low+spread.
        Trong cùng một nến chạm cả hai, ưu tiên SL (quy ước bảo thủ).
        """
        for j in range(i0 + 1, i0 + 1 + n_toi_da):
            if la_long:
                if thap[j] <= sl:
                    return False, True, j
                if cao[j] >= tp:
                    return True, False, j
            else:
                if cao[j] + spread >= sl:
                    return False, True, j
                if thap[j] + spread <= tp:
                    return True, False, j
        return False, False, i0 + n_toi_da

    def transform(self, d: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
        ch = self.ch
        cao = d["high"].to_numpy(float)
        thap = d["low"].to_numpy(float)
        dong = d["close"].to_numpy(float)
        a = atr.to_numpy(float)
        n = len(d)
        sp, H = ch.spread, ch.rao_thoi_gian

        nhan = np.zeros(n, dtype=float)
        long_thang = np.zeros(n, dtype=bool)
        short_thang = np.zeros(n, dtype=bool)
        so_nen_giu = np.full(n, np.nan)

        for i in range(n):
            if i + H >= n or not np.isfinite(a[i]) or a[i] <= 0:
                nhan[i] = np.nan
                continue

            # Kịch bản Long: vào ở Ask, thoát ở Bid
            vao_l = dong[i] + sp
            l_tp, l_sl, l_j = self._quet(
                cao, thap, i, H,
                vao_l + ch.he_so_tp * a[i], vao_l - ch.he_so_sl * a[i] - sp,
                True, sp)

            # Kịch bản Short: vào ở Bid, thoát ở Ask
            vao_s = dong[i]
            s_tp, s_sl, s_j = self._quet(
                cao, thap, i, H,
                vao_s - ch.he_so_tp * a[i], vao_s + ch.he_so_sl * a[i] + sp,
                False, sp)

            long_thang[i], short_thang[i] = l_tp, s_tp
            if l_tp and not s_tp:
                nhan[i], so_nen_giu[i] = 1.0, l_j - i
            elif s_tp and not l_tp:
                nhan[i], so_nen_giu[i] = -1.0, s_j - i
            else:
                nhan[i] = 0.0
                so_nen_giu[i] = min(l_j, s_j) - i

        return pd.DataFrame({"nhan": nhan, "long_thang": long_thang,
                             "short_thang": short_thang,
                             "so_nen_giu": so_nen_giu}, index=d.index)
'''))

    o.append(code(r'''
atr14 = ForexFeatureExtractor.atr(df["high"], df["low"], df["close"], 14)

labeler = ForexTripleBarrierLabeler(CH)
lab = labeler.transform(df, atr14)

ten = {-1.0: "Short (-1)", 0.0: "Trung tính (0)", 1.0: "Long (+1)"}
print("Phân bố nhãn (rào thời gian %d nến):" % CH.rao_thoi_gian)
for k, v in lab["nhan"].value_counts(normalize=True, dropna=True).sort_index().items():
    print("  %-16s %6.2f %%   (%s mẫu)"
          % (ten.get(k, k), 100 * v, format(int(lab["nhan"].eq(k).sum()), ",")))

ca_hai = int((lab["long_thang"] & lab["short_thang"]).sum())
khong_ai = int((~lab["long_thang"] & ~lab["short_thang"]).sum())
print()
print("Trong nhóm trung tính:")
print("  cả hai chiều cùng chạm TP (giật hai đầu): %s" % format(ca_hai, ","))
print("  không chiều nào chạm TP (đi ngang/hết giờ): %s" % format(khong_ai, ","))
print()
print("Số nến giữ trung bình: %.2f / %d" % (lab["so_nen_giu"].mean(), CH.rao_thoi_gian))
'''))

    # ═══════════════════════════════════════════ 4. Model
    o.append(md(r'''
## 4. `ForexMLModel` — LightGBM + bộ lọc ba lớp

### Vì sao `TimeSeriesSplit` chứ không K-Fold ngẫu nhiên

K-Fold xáo trộn cho phép mô hình học từ **tương lai** để dự báo **quá khứ**. Trên
chuỗi thời gian tài chính, điều đó tạo kết quả đẹp giả và sụp đổ khi chạy thật.

### Purging — vùng đệm bắt buộc

Nhãn tại nến `t` dùng thông tin tới `t + 16`. Nếu tập huấn luyện kết thúc sát tập
kiểm định, những nhãn cuối **đã nhìn vào** vùng kiểm định. Lớp này cắt bỏ
`rao_thoi_gian` nến cuối mỗi tập huấn luyện.

### Bộ lọc ba lớp

Đây là điểm khác biệt lớn nhất so với v1. Ba cửa nối tiếp, **không** cửa nào có
thể bỏ qua:

**Lớp 1 — Xác suất.** `P(Long) ≥ 0,65` hoặc `P(Short) ≥ 0,68`. Nếu cả hai cùng
vượt ngưỡng thì mô hình đang mâu thuẫn với chính nó → đứng ngoài.

**Lớp 2 — Xu hướng HTF.** Long chỉ khi giá **trên** EMA400; Short chỉ khi giá
**dưới** EMA400. Đây là lời giải trực tiếp cho khoản lỗ −4.996 USD chiều Short
ở v1: mô hình khi đó liên tục bán trong một thị trường tăng dài hạn.

**Lớp 3 — Phiên.** Chỉ mở lệnh trong 07:00–18:00 UTC. Ngoài khung này thanh
khoản mỏng, spread giãn rộng hơn 0,30 USD giả định, nên mọi lệnh mở ban đêm đều
chịu chi phí thật cao hơn mô hình tính.

> Lưu ý: bộ lọc chỉ chặn **mở** vị thế. Lệnh đã mở vẫn được quản lý bình thường
> qua đêm — đóng lệnh cưỡng bức lúc 18:00 sẽ tạo ra một sai lệch khác.

Hàm `sinh_tin_hieu()` trả về cả **nhật ký từng lớp** để đo được mỗi cửa loại bao
nhiêu tín hiệu. Không có nhật ký này thì lọc bớt lệnh xấu và lọc mất lệnh tốt
trông giống hệt nhau.
'''))

    o.append(code(r'''
class ForexMLModel:
    """LightGBM đa lớp, kiểm định tịnh tiến có purging, bộ lọc tín hiệu ba lớp."""

    def __init__(self, ch: CauHinh):
        self.ch = ch
        self.mo_hinh: list = []
        self.tam_quan_trong: pd.Series | None = None
        self.bao_cao_fold: pd.DataFrame | None = None
        self.nhat_ky_loc: pd.DataFrame | None = None

    def _tao(self, trong_so: dict) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="multiclass", num_class=3,
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            max_depth=6, min_child_samples=60,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, class_weight=trong_so,
            random_state=HAT_GIONG, n_jobs=-1, verbose=-1)

    # ── huấn luyện tịnh tiến ─────────────────────────────────────
    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        ch = self.ch
        ok = X.notna().all(axis=1) & y.notna()
        Xv, yv = X[ok], y[ok].astype(int)
        print("Dữ liệu huấn luyện: %s mẫu × %d đặc trưng"
              % (format(len(Xv), ","), Xv.shape[1]))

        ma = {-1: 0, 0: 1, 1: 2}
        y3 = yv.map(ma).to_numpy()
        Xn = Xv.to_numpy(float)

        xac_suat = np.full((len(Xv), 3), np.nan)
        tqt = np.zeros(Xv.shape[1])
        hang = []

        for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=ch.so_fold).split(Xn), 1):
            # PURGING: cắt phần cuối tập huấn luyện vì nhãn ở đó đã nhìn vào
            # vùng kiểm định
            tr = tr[:-ch.rao_thoi_gian] if len(tr) > ch.rao_thoi_gian else tr
            if len(tr) < 500:
                continue

            dem = np.bincount(y3[tr], minlength=3)
            trong_so = {c: len(tr) / (3 * max(1, dem[c])) for c in range(3)}

            m = self._tao(trong_so)
            m.fit(Xn[tr], y3[tr])
            P = m.predict_proba(Xn[te])

            # Một fold có thể thiếu lớp; ánh xạ lại theo m.classes_ để cột xác
            # suất luôn đúng thứ tự [Short, Trung tính, Long]
            Q = np.zeros((len(te), 3))
            for j, cls in enumerate(m.classes_):
                Q[:, int(cls)] = P[:, j]
            xac_suat[te] = Q

            self.mo_hinh.append(m)
            tqt += m.feature_importances_
            hang.append({"fold": k, "n_train": len(tr), "n_test": len(te),
                         "tu": str(Xv.index[te[0]])[:16],
                         "den": str(Xv.index[te[-1]])[:16],
                         "acc": float((Q.argmax(1) == y3[te]).mean())})
            print("  fold %d: train %s | test %s (%s → %s) | acc %.4f"
                  % (k, format(len(tr), ","), format(len(te), ","),
                     hang[-1]["tu"], hang[-1]["den"], hang[-1]["acc"]))

        self.tam_quan_trong = pd.Series(tqt, index=Xv.columns).sort_values(ascending=False)
        self.bao_cao_fold = pd.DataFrame(hang)

        ra = pd.DataFrame(xac_suat, index=Xv.index,
                          columns=["p_short", "p_trung_tinh", "p_long"])
        ra["nhan_that"] = yv
        return ra.dropna(subset=["p_long"])

    # ── bộ lọc ba lớp ───────────────────────────────────────────
    def sinh_tin_hieu(self, du_bao: pd.DataFrame,
                      ngu_canh: pd.DataFrame) -> pd.Series:
        """Trả về +1 mua / -1 bán / 0 đứng ngoài, sau ba cửa lọc nối tiếp."""
        ch = self.ch
        nc = ngu_canh.reindex(du_bao.index)

        # ── Lớp 1: xác suất ML
        mua = du_bao["p_long"] >= ch.nguong_mua
        ban = du_bao["p_short"] >= ch.nguong_ban
        # Cả hai cùng vượt ngưỡng = mô hình mâu thuẫn với chính nó → đứng ngoài
        mua_1, ban_1 = mua & ~ban, ban & ~mua
        n1_mua, n1_ban = int(mua_1.sum()), int(ban_1.sum())

        # ── Lớp 2: xu hướng khung lớn
        if ch.bat_loc_htf:
            tren = nc["tren_htf"].fillna(False)
            mua_2 = mua_1 & tren           # chỉ Long khi giá TRÊN EMA_HTF
            ban_2 = ban_1 & ~tren          # chỉ Short khi giá DƯỚI EMA_HTF
        else:
            mua_2, ban_2 = mua_1, ban_1
        n2_mua, n2_ban = int(mua_2.sum()), int(ban_2.sum())

        # ── Lớp 3: phiên thanh khoản cao
        if ch.bat_loc_phien:
            trong_phien = ((nc["gio_utc"] >= ch.gio_mo_utc) &
                           (nc["gio_utc"] < ch.gio_dong_utc)).fillna(False)
            mua_3, ban_3 = mua_2 & trong_phien, ban_2 & trong_phien
        else:
            mua_3, ban_3 = mua_2, ban_2
        n3_mua, n3_ban = int(mua_3.sum()), int(ban_3.sum())

        self.nhat_ky_loc = pd.DataFrame([
            {"lop": "1. Xác suất ML", "mua": n1_mua, "ban": n1_ban,
             "tong": n1_mua + n1_ban, "con_lai_%": 100.0},
            {"lop": "2. Xu hướng HTF", "mua": n2_mua, "ban": n2_ban,
             "tong": n2_mua + n2_ban,
             "con_lai_%": round(100 * (n2_mua + n2_ban) / max(1, n1_mua + n1_ban), 1)},
            {"lop": "3. Phiên giao dịch", "mua": n3_mua, "ban": n3_ban,
             "tong": n3_mua + n3_ban,
             "con_lai_%": round(100 * (n3_mua + n3_ban) / max(1, n1_mua + n1_ban), 1)},
        ])

        return pd.Series(np.where(mua_3, 1, np.where(ban_3, -1, 0)),
                         index=du_bao.index, name="tin_hieu")
'''))

    o.append(code(r'''
model = ForexMLModel(CH)
du_bao = model.fit_predict(X_all, lab["nhan"])

ma = {-1: 0, 0: 1, 1: 2}
that = du_bao["nhan_that"].map(ma)
duoc = du_bao[["p_short", "p_trung_tinh", "p_long"]].to_numpy().argmax(1)
print()
print("Ma trận nhầm lẫn (hàng = thật, cột = dự báo):")
print(pd.DataFrame(confusion_matrix(that, duoc, labels=[0, 1, 2]),
                   index=["that_-1", "that_0", "that_+1"],
                   columns=["db_-1", "db_0", "db_+1"]).to_string())
print()
print(classification_report(that, duoc, labels=[0, 1, 2],
                           target_names=["Short", "Trung tính", "Long"],
                           digits=4, zero_division=0))
'''))

    o.append(code(r'''
tin_hieu = model.sinh_tin_hieu(du_bao, ngu_canh)

print("BỘ LỌC BA LỚP — số tín hiệu còn lại sau mỗi cửa")
print(model.nhat_ky_loc.to_string(index=False))
print()
print("Kết quả cuối: %s mua | %s bán | độ phủ %.2f %% trên %s nến"
      % (format(int((tin_hieu == 1).sum()), ","),
         format(int((tin_hieu == -1).sum()), ","),
         100 * (tin_hieu != 0).mean(), format(len(tin_hieu), ",")))
'''))

    # ═══════════════════════════════════════════ 5. Backtester
    o.append(md(r'''
## 5. `ForexCFDBacktester` — khớp lệnh Bid/Ask hai chiều

### Quy ước giá

Chuỗi OHLC được coi là **giá Bid**. Ask = Bid + spread.

| | Vào lệnh | Thoát lệnh | SL/TP kiểm trên |
|---|---|---|---|
| **BUY** | Ask = Close + 0,30 | Bid | `low` / `high` |
| **SELL** | Bid = Close | Ask | `high + 0,30` / `low + 0,30` |

Bất đối xứng này là **chi phí thật**: mua đắt hơn và bán rẻ hơn giá giữa. Mô
phỏng bằng một chuỗi giá duy nhất sẽ tính thiếu một nửa spread ở mọi lệnh.

### Position sizing động

```
Risk = Equity × 0,5 %
Lot  = Risk / (khoảng_cách_SL × 100 oz)      giới hạn [0,01 ; 10,0]
```

Khoảng cách SL là khoảng cách **hiệu dụng** đã gồm spread, nên số tiền mất khi
chạm SL đúng bằng 0,5 % vốn — không phải xấp xỉ. Ô kiểm chứng sau backtest xác
nhận R bội khi cắt lỗ đúng bằng −1,0000.

### Break-even tại 0,8R

Khi giá chạy được 0,8R theo hướng có lợi, SL dời về mức PnL bằng đúng 0 **sau
khi trừ spread**:

- **BUY** vào ở Ask = E, đóng bằng bán ở Bid → hòa vốn khi **Bid = E**
- **SELL** vào ở Bid = E, đóng bằng mua ở Ask → hòa vốn khi **Ask = E**

Nhiều bản cài đặt kéo SL về "giá vào lệnh" mà không phân biệt mức đó tính theo
Bid hay Ask. Sai một spread ở đây làm lệnh break-even hóa ra vẫn lãi hoặc lỗ
nhẹ — con số nhỏ nên rất khó thấy, nhưng nó chứng tỏ mô hình khớp lệnh chưa nhất
quán. Ô kiểm chứng ngay dưới xác nhận cả hai chiều đều cho đúng 0.
'''))

    o.append(code(r'''
class ForexCFDBacktester:
    """Mô phỏng khớp lệnh CFD hai chiều: Bid/Ask, sizing động, break-even."""

    def __init__(self, ch: CauHinh):
        self.ch = ch
        self.lenh: pd.DataFrame | None = None
        self.duong_von: pd.Series | None = None

    def run(self, d: pd.DataFrame, tin_hieu: pd.Series,
            atr: pd.Series) -> pd.DataFrame:
        ch = self.ch
        idx = d.index
        cao = d["high"].to_numpy(float)
        thap = d["low"].to_numpy(float)
        dong = d["close"].to_numpy(float)
        a = atr.reindex(idx).to_numpy(float)
        s = tin_hieu.reindex(idx).fillna(0).to_numpy(int)
        sp, H = ch.spread, ch.rao_thoi_gian
        n = len(d)

        von = ch.von_ban_dau
        moc_von = np.full(n, np.nan)
        ket_qua = []
        i = 0

        while i < n - 1:
            huong = s[i]
            if huong == 0 or not np.isfinite(a[i]) or a[i] <= 0:
                moc_von[i] = von
                i += 1
                continue

            la_long = huong > 0
            # Vào lệnh theo đúng phía của spread
            vao = dong[i] + sp if la_long else dong[i]
            if la_long:
                tp = vao + ch.he_so_tp * a[i]
                sl = vao - ch.he_so_sl * a[i] - sp
            else:
                tp = vao - ch.he_so_tp * a[i]
                sl = vao + ch.he_so_sl * a[i] + sp

            R = abs(vao - sl)                       # 1R tính theo giá
            if R <= 0:
                moc_von[i] = von
                i += 1
                continue

            # Sizing: rủi ro cố định theo % vốn HIỆN TẠI
            rui_ro = von * ch.rui_ro_moi_lenh
            lot = float(np.clip(rui_ro / (R * ch.oz_moi_lot),
                                ch.lot_toi_thieu, ch.lot_toi_da))

            sl_hien = sl
            da_hoa_von = False
            j_thoat = min(i + H, n - 1)
            thoat = dong[j_thoat] if la_long else dong[j_thoat] + sp
            ly_do = "hết giờ"

            for j in range(i + 1, min(i + 1 + H, n)):
                if la_long:
                    # Long thoát ở Bid → dùng high/low nguyên bản
                    if thap[j] <= sl_hien:          # SL ưu tiên (quy ước bảo thủ)
                        thoat, j_thoat = sl_hien, j
                        ly_do = "hòa vốn" if da_hoa_von else "cắt lỗ"
                        break
                    if cao[j] >= tp:
                        thoat, j_thoat, ly_do = tp, j, "chốt lời"
                        break
                    if (not da_hoa_von) and (cao[j] - vao) >= ch.kich_hoat_hoa_von * R:
                        sl_hien = vao               # Bid = Ask vào lệnh → PnL = 0
                        da_hoa_von = True
                else:
                    # Short thoát ở Ask → cộng spread vào high/low
                    h_, l_ = cao[j] + sp, thap[j] + sp
                    if h_ >= sl_hien:
                        thoat, j_thoat = sl_hien, j
                        ly_do = "hòa vốn" if da_hoa_von else "cắt lỗ"
                        break
                    if l_ <= tp:
                        thoat, j_thoat, ly_do = tp, j, "chốt lời"
                        break
                    if (not da_hoa_von) and (vao - l_) >= ch.kich_hoat_hoa_von * R:
                        # Vòng lặp so theo giá ASK nên mức dừng cũng phải theo
                        # ASK. Đóng lệnh Short là MUA ở Ask → hòa vốn khi
                        # Ask = giá bán ban đầu.
                        sl_hien = vao
                        da_hoa_von = True
            else:
                j_thoat = min(i + H, n - 1)
                thoat = dong[j_thoat] if la_long else dong[j_thoat] + sp

            lai_lo = (thoat - vao) * (1 if la_long else -1) * lot * ch.oz_moi_lot
            von += lai_lo
            moc_von[i:j_thoat + 1] = von

            ket_qua.append({
                "vao_luc": idx[i], "thoat_luc": idx[j_thoat],
                "huong": "LONG" if la_long else "SHORT",
                "gia_vao": vao, "gia_thoat": thoat, "sl_dau": sl, "tp": tp,
                "lot": lot, "so_nen_giu": j_thoat - i,
                "lai_lo": lai_lo, "R_boi": lai_lo / rui_ro if rui_ro else np.nan,
                "ly_do": ly_do, "von_sau": von})

            i = j_thoat + 1
            if von <= 0:
                print("  ⚠ Cháy tài khoản tại", idx[j_thoat])
                break

        self.duong_von = pd.Series(moc_von, index=idx).ffill().fillna(ch.von_ban_dau)
        self.lenh = pd.DataFrame(ket_qua)
        return self.lenh

    # ── thống kê ─────────────────────────────────────────────────
    def _chi_tieu(self, l: pd.DataFrame, nhan: str) -> dict:
        if l.empty:
            return {"nhom": nhan, "so_lenh": 0, "win_rate_%": np.nan,
                    "loi_nhuan_rong": 0.0, "profit_factor": np.nan,
                    "ky_vong_R": np.nan, "sharpe": np.nan, "mdd_%": np.nan}
        p = l["lai_lo"].to_numpy(float)
        lai, lo = p[p > 0].sum(), -p[p < 0].sum()
        von = np.r_[self.ch.von_ban_dau, l["von_sau"].to_numpy(float)]
        dinh = np.maximum.accumulate(von)
        sut = (von - dinh) / dinh
        # Sharpe quy năm theo TẦN SUẤT LỆNH thật, không giả định 252 ngày
        r = p / self.ch.von_ban_dau
        nam = max((l["thoat_luc"].max() - l["vao_luc"].min()).days / 365.25, 1e-9)
        sd = r.std(ddof=1) if len(r) > 1 else np.nan
        return {
            "nhom": nhan,
            "so_lenh": len(l),
            "win_rate_%": round(100 * float((p > 0).mean()), 2),
            "loi_nhuan_rong": round(float(p.sum()), 2),
            "profit_factor": round(float(lai / lo), 3) if lo > 0 else np.inf,
            "ky_vong_R": round(float(l["R_boi"].mean()), 3),
            "sharpe": round(float(r.mean() / sd * np.sqrt(len(l) / nam)), 3) if sd else np.nan,
            "mdd_%": round(float(-sut.min() * 100), 2),
        }

    def bao_cao(self) -> pd.DataFrame:
        l = self.lenh
        return pd.DataFrame([
            self._chi_tieu(l, "TỔNG THỂ"),
            self._chi_tieu(l[l["huong"] == "LONG"], "LONG"),
            self._chi_tieu(l[l["huong"] == "SHORT"], "SHORT"),
        ])
'''))

    o.append(code(r'''
bt = ForexCFDBacktester(CH)
lenh = bt.run(df, tin_hieu, atr14)

print("Tổng số lệnh:", format(len(lenh), ","))
print()
print(bt.bao_cao().to_string(index=False))
print()
print("Vốn: %s → %s USD  (%+.2f %%)"
      % (format(CH.von_ban_dau, ",.0f"), format(bt.duong_von.iloc[-1], ",.2f"),
         100 * (bt.duong_von.iloc[-1] / CH.von_ban_dau - 1)))
print()
print("Lý do thoát lệnh:")
print(pd.crosstab(lenh["huong"], lenh["ly_do"], margins=True).to_string())
'''))

    o.append(code(r'''
# ── Kiểm chứng tính nhất quán của mô hình khớp lệnh ─────────────
# Lệnh thoát bằng "hòa vốn" phải cho lãi/lỗ ĐÚNG BẰNG 0 ở CẢ HAI chiều. Lệch dù
# chỉ vài USD nghĩa là đâu đó nhầm giữa giá Bid và giá Ask.
hv = lenh[lenh["ly_do"] == "hòa vốn"]
print("Kiểm chứng break-even (phải bằng 0 ở cả hai chiều):")
if hv.empty:
    print("  (chưa có lệnh nào chạm hòa vốn)")
else:
    for h, g in hv.groupby("huong"):
        print("  %-6s %3d lệnh | trung bình %+.6f USD | |max| %.6f"
              % (h, len(g), g["lai_lo"].mean(), g["lai_lo"].abs().max()))
    lech = hv["lai_lo"].abs().max()
    print("  →", "ĐẠT" if lech < 1e-6
          else "SAI LỆCH %.6f USD — kiểm tra lại Bid/Ask" % lech)

# Lệnh cắt lỗ phải mất đúng mức rủi ro đã định
cl = lenh[lenh["ly_do"] == "cắt lỗ"]
if not cl.empty:
    print()
    print("Kiểm chứng sizing (cắt lỗ phải mất ≈ %.1f%% vốn):"
          % (100 * CH.rui_ro_moi_lenh))
    print("  R bội trung bình khi cắt lỗ: %.4f  (kỳ vọng −1,0)" % cl["R_boi"].mean())
    print("  Lot: min %.2f | trung vị %.2f | max %.2f"
          % (lenh["lot"].min(), lenh["lot"].median(), lenh["lot"].max()))
'''))

    # ═══════════════════════════════════════════ 6. Biểu đồ
    o.append(md(r'''
## 6. Trực quan hóa

Bốn biểu đồ, mỗi cái trả lời một câu hỏi khác nhau:

1. **Equity curve tách chiều** — lợi nhuận đến từ Long hay Short? Nếu chỉ một
   chiều sinh lợi, hệ thống đang cưỡi xu hướng chứ không có ưu thế hai chiều.
2. **Underwater plot** — sụt giảm sâu bao nhiêu và **kéo dài bao lâu**. Thời gian
   chìm dưới đỉnh cũ quan trọng không kém độ sâu.
3. **Feature importance** — đặc trưng nào chi phối quyết định.
4. **Phân bố R bội** — hệ thống sống nhờ vài lệnh lớn hay nhờ biên mỏng đều đặn.
'''))

    o.append(code(r'''
fig, ax = plt.subplots(2, 2, figsize=(15, 9))

# ── 1. Equity curve tách chiều
l = lenh.copy()
if len(l):
    ax[0, 0].plot(l["thoat_luc"], l["lai_lo"].cumsum() + CH.von_ban_dau,
                  lw=1.7, label="Tổng", color="#1f3a5f")
    for h_, mau in (("LONG", "#1b6b3a"), ("SHORT", "#a31d1d")):
        g = l[l["huong"] == h_]
        if len(g):
            ax[0, 0].plot(g["thoat_luc"], g["lai_lo"].cumsum() + CH.von_ban_dau,
                          lw=1.1, alpha=0.85, label=h_, color=mau)
ax[0, 0].axhline(CH.von_ban_dau, ls="--", lw=0.9, color="grey")
ax[0, 0].set_title("Equity curve — tổng và tách theo chiều")
ax[0, 0].legend()
ax[0, 0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

# ── 2. Underwater
von = bt.duong_von
sut = 100 * (von - von.cummax()) / von.cummax()
ax[0, 1].fill_between(sut.index, sut.values, 0, color="#a31d1d", alpha=0.35)
ax[0, 1].plot(sut.index, sut.values, lw=0.8, color="#a31d1d")
ax[0, 1].set_title("Underwater — sụt giảm so với đỉnh cũ (%)")
ax[0, 1].set_ylabel("%")

# ── 3. Feature importance
top = model.tam_quan_trong.head(10).iloc[::-1]
ax[1, 0].barh(top.index, top.values, color="#1f3a5f", alpha=0.85)
ax[1, 0].set_title("Top 10 đặc trưng chi phối quyết định")

# ── 4. Phân bố R bội
if len(l):
    ax[1, 1].hist(l["R_boi"], bins=40, color="#1f3a5f", alpha=0.8)
    ax[1, 1].axvline(0, color="grey", ls="--", lw=1)
    ax[1, 1].axvline(l["R_boi"].mean(), color="#a31d1d", lw=1.4,
                     label="trung bình %.3fR" % l["R_boi"].mean())
    ax[1, 1].legend()
ax[1, 1].set_title("Phân bố kết quả từng lệnh (bội số R)")

plt.tight_layout()
plt.show()
'''))

    # ═══════════════════════════════════════════ 7. Ablation
    o.append(md(r'''
## 7. Đo thiệt hại của từng lớp lọc

Đây là ô quan trọng nhất của bản v2, và cũng là thứ v1 hoàn toàn thiếu.

Bộ lọc luôn làm giảm số lệnh — điều đó chắc chắn. Câu hỏi thật là: nó **loại
lệnh xấu hay loại luôn cả lệnh tốt**? Chỉ nhìn tổng lợi nhuận thì hai trường hợp
trông giống hệt nhau.

Ô dưới chạy lại backtest với **từng tổ hợp lọc** trên cùng dự báo, cùng dữ liệu,
cùng tham số — khác biệt duy nhất là bật/tắt lớp lọc.
'''))

    o.append(code(r'''
def thu_cau_hinh(bat_htf: bool, bat_phien: bool, nguong_m: float, nguong_b: float,
                 nhan: str) -> dict:
    """Chạy lại toàn bộ khâu tín hiệu + backtest với một tổ hợp lọc."""
    import copy
    ch2 = copy.deepcopy(CH)
    ch2.bat_loc_htf, ch2.bat_loc_phien = bat_htf, bat_phien
    ch2.nguong_mua, ch2.nguong_ban = nguong_m, nguong_b

    m2 = ForexMLModel(ch2)
    sig = m2.sinh_tin_hieu(du_bao, ngu_canh)
    bt2 = ForexCFDBacktester(ch2)
    lenh2 = bt2.run(df, sig, atr14)
    if lenh2.empty:
        return {"cau_hinh": nhan, "so_lenh": 0}
    r = bt2._chi_tieu(lenh2, nhan)
    rL = bt2._chi_tieu(lenh2[lenh2["huong"] == "LONG"], "L")
    rS = bt2._chi_tieu(lenh2[lenh2["huong"] == "SHORT"], "S")
    return {"cau_hinh": nhan, "so_lenh": r["so_lenh"],
            "win_%": r["win_rate_%"], "lai_lo": r["loi_nhuan_rong"],
            "PF": r["profit_factor"], "mdd_%": r["mdd_%"],
            "n_long": rL["so_lenh"], "pnl_long": rL["loi_nhuan_rong"],
            "n_short": rS["so_lenh"], "pnl_short": rS["loi_nhuan_rong"]}


cau_hinh = [
    (False, False, 0.55, 0.55, "v1: ngưỡng 0,55, không lọc"),
    (False, False, CH.nguong_mua, CH.nguong_ban, "chỉ siết ngưỡng"),
    (True,  False, CH.nguong_mua, CH.nguong_ban, "+ lọc HTF"),
    (False, True,  CH.nguong_mua, CH.nguong_ban, "+ lọc phiên"),
    (True,  True,  CH.nguong_mua, CH.nguong_ban, "v2: đủ ba lớp"),
]
kq = pd.DataFrame([thu_cau_hinh(*c) for c in cau_hinh])
print("THIỆT HẠI / LỢI ÍCH CỦA TỪNG LỚP LỌC")
print(kq.to_string(index=False))
'''))

    o.append(md(r'''
## 8. Đọc kết quả cho đúng

**Trên dữ liệu mô phỏng**, mọi con số chỉ chứng minh **pipeline chạy đúng** —
không nói gì về khả năng sinh lợi. Dữ liệu sinh từ bước ngẫu nhiên có cụm biến
động nên về lý thuyết **không tồn tại** mẫu hình khai thác được. Backtest cho lãi
lớn trên dữ liệu này là dấu hiệu rò rỉ, không phải mô hình giỏi.

### Bốn dấu hiệu cần kiểm tra trước khi tin bất kỳ kết quả nào

| Dấu hiệu | Ngưỡng đáng ngờ | Ý nghĩa |
|---|---|---|
| Win rate | > 65 % với TP/SL = 1,5 | Thường là rò rỉ hoặc lỗi khớp lệnh |
| Long vs Short | một chiều lãi, chiều kia lỗ nặng | Chỉ cưỡi xu hướng, không có ưu thế thật |
| Số lệnh Short | gần bằng 0 sau lọc HTF | Hệ thống đã thành một chiều — nên nói rõ thay vì gọi là "hai chiều" |
| Độ phủ | > 20 % sau ba lớp lọc | Bộ lọc chưa siết đủ, spread vẫn bào mòn |

### Chạy dữ liệu thật

```python
CH = CauHinh(duong_dan="duong/dan/xauusd_m15.csv", khung="M15")
```

Tệp cần các cột `datetime, open, high, low, close, volume`. Cột thừa bị bỏ qua.

> **Lưu ý về `rao_thoi_gian`.** 16 nến trên M15 là 4 giờ, nhưng trên H1 là 16
> giờ. Đổi khung dữ liệu thì phải chỉnh lại tham số này, nếu không bài toán đã
> thay đổi mà mình không biết.

### Ba việc bắt buộc trước khi dùng tiền thật

1. **Kiểm định xáo trộn nhãn** — xáo trộn cột `nhan` rồi huấn luyện lại. Macro-F1
   phải rơi về mức đoán mò ≈ 0,333. Nếu vẫn cao thì đường ống có rò rỉ.
2. **Tập niêm phong** — cắt hẳn năm gần nhất, không đụng tới trong suốt quá trình
   phát triển, mở **đúng một lần** ở cuối.
3. **So với mua-và-giữ** — trong một thị trường tăng dài hạn, đây mới là baseline
   công bằng cho chiều Long. Thắng backtest mà thua mua-và-giữ thì hệ thống chưa
   tạo ra giá trị nào.
'''))

    return o

    return o


# ══════════════════════════════════════════════════════════ xuất
def kiem_tra(nb: dict) -> tuple[int, int]:
    ma = md_ = 0
    for i, o in enumerate(nb["cells"]):
        for j, dong in enumerate(o["source"][:-1]):
            assert dong.endswith("\n"), "ô %d dòng %d thiếu \\n: %r" % (i, j, dong)
        if o["cell_type"] == "code":
            ast.parse("".join(o["source"]))
            ma += 1
        else:
            md_ += 1
    return ma, md_


def chay():
    nb = {
        "cells": cac_o(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    ma, md_ = kiem_tra(nb)
    RA.parent.mkdir(parents=True, exist_ok=True)
    RA.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Đã tạo %s — %d ô mã, %d ô markdown" % (RA.name, ma, md_))


if __name__ == "__main__":
    chay()
