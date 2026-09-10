# -*- coding: utf-8 -*-
"""Tải XAUUSD (time + OHLCV) từ Dukascopy, giai đoạn 01/01/2015 → 31/12/2025.

Tải nến M1 của TỪNG NGÀY từ datafeed.dukascopy.com rồi tự gộp lên khung cần. Mọi
khung đều phủ trọn 2015–2025, đi ra từ cùng một nguồn, và VOLUME LÀ VOLUME THẬT của
Dukascopy. Khoảng 3.400 tệp ngày nhỏ; có bộ nhớ đệm nên bị ngắt thì chạy lại sẽ tải tiếp.
Tệp ngày để ở --bo-dem (đĩa cục bộ, ghi nhanh); mỗi tháng tải xong được gộp thành một tệp
trong --luu-thang (trên Colab để trong Google Drive: chỉ khoảng 130 tệp, mất phiên vẫn còn).

CHẠY
  python cao_du_lieu_dukascopy.py                               # M30 H1 H4 D1
  python cao_du_lieu_dukascopy.py --khung M15 H1 D1 --gia mid

  Trên Google Colab dùng notebook "00_Tai_du_lieu_Dukascopy.ipynb" (bấm Run all).

  Nếu mạng chặn Dukascopy (DNS trả 127.0.0.1 — gặp với một số nhà mạng ở Việt Nam),
  script dừng và báo; khi đó hãy chạy trên Google Colab.

ĐẦU RA
  <thu_muc>/dukascopy_XAUUSD_<khung>.csv — đúng 6 cột: time, open, high, low, close, volume
  time là giờ MỞ nến theo UTC. Giá là giá BID (giống biểu đồ MT5), đổi bằng --gia ask|mid.
  D1 gộp theo phiên ngoại hối 22:00 → 22:00 UTC (không sinh nến rác ngày Chủ nhật).

TỰ KIỂM TRA (dừng và báo nếu không đạt, để không lặng lẽ ghi dữ liệu sai)
  · Giá hợp lý: vàng 2015–2025 nằm trong khoảng 1.000–5.000 USD. Sai hệ số chia giá
    (ví dụ chia 100 thay vì 1.000) sẽ cho giá 10.000+ và bị chặn ngay.
  · Giờ UTC: thị trường vàng mở lại vào tối Chủ nhật lúc 21:00–23:00 UTC. Nếu nến đầu
    tuần rơi vào giờ khác thì thời gian đang bị lệch múi giờ.
"""
from __future__ import annotations

import argparse
import lzma
import os
import random
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GOC = Path(__file__).resolve().parent.parent
TU, DEN = date(2015, 1, 1), date(2025, 12, 31)
HE_SO_GIA = 1000            # XAUUSD: 3 chữ số thập phân → giá = số nguyên / 1000
GIA_HOP_LY = (1000, 5000)   # khoảng giá vàng giai đoạn 2015–2025 (USD/oz)
UA = {"User-Agent": "Mozilla/5.0"}
COT = ["time", "open", "high", "low", "close", "volume"]
# Chờ kết nối 10 giây, chờ dữ liệu 20 giây: kết nối treo thì bỏ sớm và thử lại, thay vì
# mất trọn 60 giây mỗi lần như bản trước (nguyên nhân chạy rất chậm trên Colab).
THOI_GIAN_CHO = (10, 20)
SO_LAN_THU = 8
NGHI_CO_SO = 1.5            # giây nghỉ trước lần thử lại đầu, sau đó gấp đôi (tối đa 30)
# Tối đa 10 yêu cầu/giây cho mọi luồng cộng lại: chạy thử trên Colab ở ~20 yêu cầu/giây thì
# sau ~3.200 tệp máy chủ bắt đầu từ chối liên tục.
TOC_DO_TOI_DA = 10
KIEM_SOM = 40               # sau 40 tệp tải về mà toàn rỗng → dừng và báo
IN_TIEN_DO = 20             # giây giữa hai lần in tiến độ

# Khung → quy tắc gộp của pandas. D1 gộp theo phiên ngoại hối 22:00 → 22:00 UTC:
# gộp theo mốc 00:00 sẽ sinh một nến D1 giả chỉ dài 1–2 giờ vào tối Chủ nhật.
KHUNG = {"M1": ("1min", None), "M5": ("5min", None), "M15": ("15min", None),
         "M30": ("30min", None), "H1": ("1h", None), "H4": ("4h", None),
         "D1": ("24h", "22h")}

DTYPE = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])
_luong = threading.local()


def _phien():
    if not hasattr(_luong, "s"):
        _luong.s = requests.Session()
        _luong.s.headers.update(UA)
    return _luong.s


# ══════════════════════════════════════════════════════════ tải và giải mã
def kiem_tra_mang():
    try:
        ip = socket.gethostbyname("datafeed.dukascopy.com")
    except OSError as e:
        raise SystemExit("Không phân giải được datafeed.dukascopy.com (%s). Hãy chạy trên Google Colab." % e)
    if ip.startswith("127.") or ip == "0.0.0.0":
        raise SystemExit("Mạng này đang CHẶN Dukascopy: tên miền phân giải về %s.\n"
                         "→ Chạy trên Google Colab bằng notebook 00_Tai_du_lieu_Dukascopy.ipynb." % ip)


def giai_ma_ngay(noi_dung: bytes, ngay: date) -> pd.DataFrame:
    """Tệp BID/ASK_candles_min_1.bi5: LZMA, mỗi nến 24 byte big-endian
    (giây kể từ 00:00 UTC, open, close, low, high, volume)."""
    if not noi_dung:
        return pd.DataFrame()
    raw = lzma.decompress(noi_dung)
    a = np.frombuffer(raw[:len(raw) - len(raw) % 24], dtype=DTYPE)
    t = pd.Timestamp(ngay) + pd.to_timedelta(a["t"].astype("int64"), unit="s")
    d = pd.DataFrame({"time": t.astype("datetime64[ns]"),
                      "open": a["o"] / HE_SO_GIA, "high": a["h"] / HE_SO_GIA,
                      "low": a["l"] / HE_SO_GIA, "close": a["c"] / HE_SO_GIA,
                      "volume": a["v"].astype(float)})
    # Phút không có tick được Dukascopy điền nến phẳng với volume 0 (cả cuối tuần) → bỏ
    return d[d["volume"] > 0]


def _doc_duoc(noi_dung: bytes) -> bool:
    try:
        if noi_dung:
            lzma.decompress(noi_dung)
        return True
    except (lzma.LZMAError, EOFError):
        return False


def _ghi_an_toan(p: Path, noi_dung: bytes) -> None:
    """Ghi ra tệp tạm rồi đổi tên: bị ngắt giữa lúc ghi cũng không để lại tệp hỏng."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tam = p.with_name(p.name + ".tam")
    tam.write_bytes(noi_dung)
    os.replace(tam, p)


class GioiHan:
    """Giãn đều các yêu cầu của mọi luồng để không vượt quá moi_giay yêu cầu/giây."""

    def __init__(self, moi_giay: float):
        self.khoang = 1.0 / moi_giay if moi_giay and moi_giay > 0 else 0.0
        self._khoa, self._ke = threading.Lock(), 0.0

    def cho(self, dung: threading.Event):
        if not self.khoang:
            return
        with self._khoa:
            bay_gio = time.monotonic()
            luot = max(bay_gio, self._ke)
            self._ke = luot + self.khoang
        if luot > bay_gio:
            dung.wait(luot - bay_gio)


class ThongKe:
    """Đếm số yêu cầu, số lần thử lại và thời gian chờ mạng (dùng chung giữa các luồng)."""

    def __init__(self, toc_do: float = 0):
        self.gioi_han = GioiHan(toc_do)
        self._khoa = threading.Lock()
        self.yeu_cau = self.thu_lai = self.da_tai = self.rong = 0
        self.giay_mang = 0.0
        self.nghi_van = set()           # ngày máy chủ trả 200 nhưng tệp rỗng cả 3 lần
        # Bật khi có lỗi hoặc bị ngắt: mọi luồng của lần chạy này đang nghỉ/thử lại dừng ngay
        self.dung = threading.Event()

    def cong(self, **kw):
        with self._khoa:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def them_nghi_van(self, ngay: date):
        with self._khoa:
            self.nghi_van.add(ngay)


def tai_ngay(ky_hieu: str, loai: str, ngay: date, bo_dem: Path, tk: ThongKe | None = None) -> bytes:
    tk = tk or ThongKe()
    p = bo_dem / ky_hieu / loai / ("%s.bi5" % ngay.isoformat())
    if p.exists():
        b = p.read_bytes()
        if _doc_duoc(b):
            return b
        p.unlink()                      # tệp đệm hỏng (bản cũ bị ngắt giữa lúc ghi) → tải lại
    url = ("https://datafeed.dukascopy.com/datafeed/%s/%04d/%02d/%02d/%s_candles_min_1.bi5"
           % (ky_hieu, ngay.year, ngay.month - 1, ngay.day, loai))      # tháng đếm từ 0
    so_rong, loi = 0, ""
    for lan in range(SO_LAN_THU):
        tk.gioi_han.cho(tk.dung)
        if tk.dung.is_set():
            raise RuntimeError("Đã dừng: %s" % url)
        t0 = time.monotonic()
        try:
            r = _phien().get(url, timeout=THOI_GIAN_CHO)
            loi = "HTTP %d" % r.status_code
        except requests.RequestException as e:
            r, loi = None, type(e).__name__
        tk.cong(yeu_cau=1, giay_mang=time.monotonic() - t0)
        if r is not None and r.status_code == 404:
            b = b""                                                      # ngày không có dữ liệu
            break
        if r is not None and r.status_code == 200:
            b = r.content
            if b:
                break
            # Ngày giao dịch luôn có tệp khác rỗng (cả phút không tick cũng được điền);
            # tệp rỗng thường do máy chủ đang giới hạn. Rỗng 3 lần thì chấp nhận nhưng đánh
            # dấu nghi vấn và KHÔNG lưu đệm, để lần chạy sau tải lại ngày này.
            so_rong += 1
            if so_rong >= 3:
                tk.them_nghi_van(ngay)
                tk.cong(da_tai=1, rong=1)
                return b
        # Lỗi mạng, 429/503 hoặc tệp rỗng: bỏ kết nối cũ (có thể đã chết), nghỉ rồi thử lại
        tk.cong(thu_lai=1)
        _luong.__dict__.pop("s", None)
        tk.dung.wait(min(30.0, NGHI_CO_SO * 2 ** lan) * (0.5 + random.random()))
    else:
        tk.dung.set()                   # báo các luồng khác dừng ngay, không tải thêm vòng nữa
        raise RuntimeError("Tải thất bại sau %d lần (lỗi cuối: %s): %s" % (SO_LAN_THU, loi, url))
    tk.cong(da_tai=1, rong=int(not b))
    _ghi_an_toan(p, b)
    return b


def _tep_thang(luu_thang: Path | None, ky_hieu: str, loai: str, ym) -> Path | None:
    return luu_thang / ("%s_%s_%04d-%02d.csv.gz" % (ky_hieu, loai, *ym)) if luu_thang else None


def _phut(giay: float) -> str:
    return "%d phút %02d giây" % divmod(int(giay), 60)


def m1_dukascopy(ky_hieu: str, loai: str, bo_dem: Path, so_luong: int,
                 luu_thang: Path | None = None, toc_do: float | None = None) -> pd.DataFrame:
    """Nến M1 cả giai đoạn. Tệp ngày lưu ở bo_dem; mỗi tháng tải xong được gộp thành
    một tệp trong luu_thang (nên để trên Google Drive: ít tệp, bị ngắt vẫn chạy tiếp)."""
    # Lấy thừa một ngày ở đầu để nến D1 phiên 22:00 đầu tiên đủ dữ liệu
    ngay_ds = [TU - timedelta(days=1) + timedelta(days=i)
               for i in range((DEN - TU).days + 2)]
    ngay_ds = [d for d in ngay_ds if d.weekday() != 5]                    # thứ Bảy không giao dịch
    luu_thang = Path(luu_thang) if luu_thang else None
    thang = {}
    for d in ngay_ds:
        thang.setdefault((d.year, d.month), []).append(d)

    phan, can_tai = {}, []
    for ym, ds in thang.items():
        p = _tep_thang(luu_thang, ky_hieu, loai, ym)
        if p is not None and p.exists():
            g = pd.read_csv(p, parse_dates=["time"])
            g["time"] = g["time"].astype("datetime64[ns]")         # cùng đơn vị với nến vừa tải
            phan[ym] = g
        else:
            can_tai += ds
    print("Tải %s %s M1: %d tệp ngày | %d/%d tháng đã có sẵn | cần tải %d ngày, %d luồng"
          % (ky_hieu, loai, len(ngay_ds), len(phan), len(thang), len(can_tai), so_luong), flush=True)

    tk, cho_gop = ThongKe(TOC_DO_TOI_DA if toc_do is None else toc_do), {}
    t0 = lan_in = time.monotonic()
    xong = 0
    ex = ThreadPoolExecutor(max_workers=so_luong)
    try:
        viec = {ex.submit(tai_ngay, ky_hieu, loai, d, bo_dem, tk): d for d in can_tai}
        for f in as_completed(viec):
            d = viec[f]
            ym = (d.year, d.month)
            cho_gop.setdefault(ym, {})[d] = giai_ma_ngay(f.result(), d)
            xong += 1
            if len(cho_gop[ym]) == len(thang[ym]):                       # đủ ngày của tháng
                g = [x for _, x in sorted(cho_gop.pop(ym).items()) if len(x)]
                phan[ym] = pd.concat(g, ignore_index=True) if g else pd.DataFrame(columns=COT)
                p = _tep_thang(luu_thang, ky_hieu, loai, ym)
                if p is not None and not tk.nghi_van.intersection(thang[ym]):
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tam = p.with_name(p.name + ".tam")               # ghi tạm rồi đổi tên
                    phan[ym].to_csv(tam, index=False, compression="gzip")
                    os.replace(tam, p)
            if tk.da_tai >= KIEM_SOM and tk.rong == tk.da_tai:
                raise SystemExit("Máy chủ Dukascopy trả về toàn tệp rỗng (%d tệp đầu) — có thể đang bị "
                                 "chặn hoặc giới hạn. Đợi vài phút rồi chạy lại với --luong nhỏ hơn."
                                 % tk.da_tai)
            bay_gio = time.monotonic()
            if bay_gio - lan_in >= IN_TIEN_DO or xong == len(can_tai):
                lan_in = bay_gio
                toc_do = xong / max(bay_gio - t0, 1e-9)
                print("  %4d / %d ngày | %.1f ngày/giây | còn khoảng %s | chờ mạng TB %.1f giây/yêu cầu"
                      " | thử lại %d | tệp rỗng %d"
                      % (xong, len(can_tai), toc_do, _phut((len(can_tai) - xong) / max(toc_do, 1e-9)),
                         tk.giay_mang / max(tk.yeu_cau, 1), tk.thu_lai, tk.rong), flush=True)
    except RuntimeError as e:
        tk.dung.set()                                          # dừng ngay, không chờ hàng đợi
        ex.shutdown(wait=False, cancel_futures=True)
        con = len(thang) - len(phan)
        noi_luu = ("đã lưu %d/%d tháng vào %s" % (len(phan), len(thang), luu_thang) if luu_thang
                   else "các ngày đã tải nằm trong %s" % bo_dem)
        raise SystemExit(
            "\n⚠ Máy chủ Dukascopy từ chối liên tục — %s\n  Tiến độ: %s, còn %d tháng chưa xong.\n"
            "  → Đợi 5–10 phút rồi chạy lại: chỉ tải phần còn thiếu. Nếu vẫn bị từ chối, chọn\n"
            "    Thời gian chạy → Ngắt kết nối và xoá thời gian chạy (đổi máy Colab) rồi Run all."
            % (e, noi_luu, con)) from e
    except BaseException:
        tk.dung.set()
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    if can_tai:
        print("  Xong phần tải sau %s." % _phut(time.monotonic() - t0))
    if tk.nghi_van:
        ds = sorted(tk.nghi_van)
        print("  ⚠ %d ngày máy chủ trả tệp rỗng cả 3 lần (có thể đang bị giới hạn): %s%s\n"
              "    Các tháng chứa những ngày này chưa được lưu — chạy lại để tải lại riêng các ngày đó."
              % (len(ds), ", ".join(map(str, ds[:10])), " …" if len(ds) > 10 else ""))

    co = [phan[ym] for ym in sorted(phan) if len(phan[ym])]
    if not co:
        raise SystemExit("Không tải được nến nào — kiểm tra lại kết nối tới Dukascopy.")
    m = pd.concat(co, ignore_index=True)
    return m.drop_duplicates("time").sort_values("time").reset_index(drop=True)


# ══════════════════════════════════════════════════════════ tự kiểm tra
def tu_kiem_tra(m1: pd.DataFrame) -> None:
    """Chặn hai lỗi âm thầm: sai hệ số chia giá và lệch múi giờ."""
    trung_vi = float(m1["close"].median())
    thap, cao = float(m1["low"].min()), float(m1["high"].max())
    print("  Giá: trung vị %.2f | thấp nhất %.2f | cao nhất %.2f USD" % (trung_vi, thap, cao))
    if not (GIA_HOP_LY[0] * 0.8 <= thap and cao <= GIA_HOP_LY[1] * 1.2):
        raise SystemExit("⚠ Giá nằm ngoài khoảng hợp lý %s USD — hệ số chia giá (%d) có thể sai."
                         % (GIA_HOP_LY, HE_SO_GIA))

    # Nến đầu tiên của mỗi tuần (sau khoảng nghỉ cuối tuần > 24 giờ) phải mở tối Chủ nhật UTC
    nghi = m1["time"].diff() > pd.Timedelta(hours=24)
    dau_tuan = m1.loc[nghi, "time"]
    gio = dau_tuan.dt.hour.value_counts(normalize=True)
    dung = float(gio.reindex([21, 22, 23]).fillna(0).sum())
    chu_nhat = float((dau_tuan.dt.dayofweek == 6).mean())
    print("  Giờ mở cửa đầu tuần (UTC): %s | %.1f%% vào Chủ nhật 21–23 giờ"
          % (", ".join("%dh:%.0f%%" % (h, 100 * v) for h, v in gio.head(4).items()), 100 * dung))
    if dung < 0.8 or chu_nhat < 0.8:
        raise SystemExit("⚠ Phiên đầu tuần không mở vào tối Chủ nhật 21–23 giờ UTC — thời gian "
                         "có thể đang lệch múi giờ.")
    print("  ✓ Giá hợp lý và thời gian đúng UTC.")


# ══════════════════════════════════════════════════════════ gộp và ghi
def gop(m: pd.DataFrame, khung: str) -> pd.DataFrame:
    quy_tac, lech = KHUNG[khung]
    if khung == "M1":
        return m.copy()
    g = m.set_index("time").resample(quy_tac, offset=lech) if lech else \
        m.set_index("time").resample(quy_tac)
    d = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return d.dropna(subset=["close"]).reset_index()


def cat_giai_doan(d: pd.DataFrame) -> pd.DataFrame:
    d = d[(d["time"] >= pd.Timestamp(TU)) & (d["time"] < pd.Timestamp(DEN) + pd.Timedelta(days=1))]
    return d.reset_index(drop=True)


def kiem_tra(d: pd.DataFrame, khung: str) -> None:
    sai = ((d["high"] < d[["open", "close"]].max(axis=1)) |
           (d["low"] > d[["open", "close"]].min(axis=1))).sum()
    thieu = d[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum()
    print("  %-3s %9d nến | %s → %s | sai hình học %d | thiếu giá trị %d | volume trung vị %.2f"
          % (khung, len(d), d["time"].min(), d["time"].max(), sai, thieu, d["volume"].median()))
    so_nam = d.groupby(d["time"].dt.year).size()
    print("      Số nến mỗi năm: " + ", ".join("%d:%d" % kv for kv in so_nam.items()))


def chay(khung_ds, gia="bid", thu_muc=None, bo_dem=None, so_luong=8, luu_thang=None, toc_do=None):
    thu_muc = Path(thu_muc) if thu_muc else GOC / "data" / "raw"
    thu_muc.mkdir(parents=True, exist_ok=True)
    bo_dem = Path(bo_dem) if bo_dem else thu_muc / "_bo_dem_dukascopy"
    kiem_tra_mang()

    if gia == "mid":
        b = m1_dukascopy("XAUUSD", "BID", bo_dem, so_luong, luu_thang, toc_do)
        a = m1_dukascopy("XAUUSD", "ASK", bo_dem, so_luong, luu_thang, toc_do)
        m1 = b.merge(a, on="time", suffixes=("_b", "_a"))
        m1 = pd.DataFrame({"time": m1["time"],
                           **{k: (m1[k + "_b"] + m1[k + "_a"]) / 2
                              for k in ("open", "high", "low", "close")},
                           "volume": m1["volume_b"]})
    else:
        m1 = m1_dukascopy("XAUUSD", gia.upper(), bo_dem, so_luong, luu_thang, toc_do)
    print("Đã có %d nến M1, %s → %s" % (len(m1), m1["time"].min(), m1["time"].max()))
    tu_kiem_tra(m1)

    print("\nGộp khung và ghi tệp (%s → %s):" % (TU, DEN))
    ra = []
    for k in khung_ds:
        d = cat_giai_doan(gop(m1, k))
        d["volume"] = d["volume"].round(4)
        kiem_tra(d, k)
        p = thu_muc / ("dukascopy_XAUUSD_%s.csv" % k)
        d[["time", "open", "high", "low", "close", "volume"]].to_csv(p, index=False)
        print("      → %s" % p)
        ra.append(p)
    return ra


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tải XAUUSD 2015–2025 (time + OHLCV) từ Dukascopy")
    ap.add_argument("--khung", nargs="+", default=["M30", "H1", "H4", "D1"], choices=list(KHUNG))
    ap.add_argument("--gia", choices=["bid", "ask", "mid"], default="bid")
    ap.add_argument("--thu-muc", default=None, help="thư mục ghi tệp, mặc định data/raw")
    ap.add_argument("--bo-dem", default=None, help="thư mục lưu tệp ngày đã tải (nên để ở đĩa cục bộ)")
    ap.add_argument("--luu-thang", default=None,
                    help="thư mục lưu mỗi tháng một tệp (trên Colab nên để trong Google Drive)")
    ap.add_argument("--luong", type=int, default=8, help="số luồng tải song song")
    ap.add_argument("--toc-do", type=float, default=TOC_DO_TOI_DA,
                    help="tối đa bao nhiêu yêu cầu/giây (cộng mọi luồng); 0 = không giới hạn")
    a = ap.parse_args(argv)
    chay(a.khung, a.gia, a.thu_muc, a.bo_dem, a.luong, a.luu_thang, a.toc_do)


if __name__ == "__main__":
    main()
