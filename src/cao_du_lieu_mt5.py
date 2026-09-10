# -*- coding: utf-8 -*-
"""Cào dữ liệu XAUUSD (time + OHLCV) từ terminal MetaTrader 5 — Exness, VT Markets, …

YÊU CẦU
  · Windows. Thư viện MetaTrader5 chỉ có cho Windows — KHÔNG chạy được trên Colab.
    Chạy script này trên máy tính, rồi tải tệp CSV lên Drive cho NB1 và NB2.
  · Terminal MT5 của sàn đã cài và ĐĂNG NHẬP SẴN (tài khoản demo cũng được).
    Script KHÔNG hỏi mật khẩu: nó đọc dữ liệu qua tài khoản đang đăng nhập trên terminal,
    và chỉ ĐỌC dữ liệu giá — không đặt lệnh.
  · pip install MetaTrader5 pandas

CHẠY
  python src/cao_du_lieu_mt5.py --san vtmarkets                     # H1, 2015-01-01 → hôm nay
  python src/cao_du_lieu_mt5.py --san exness --khung M15 H1 D1 --tu 2018-01-01
  python src/cao_du_lieu_mt5.py --terminal "D:\\MT5\\terminal64.exe" --mui-gio ny+7

ĐẦU RA
  data/raw/<san>_XAUUSD_<khung>.csv — đúng 6 cột: time, open, high, low, close, volume
  time là giờ MỞ nến theo UTC; volume là tick volume (sàn CFD không có khối lượng thật).

GIỜ MÁY CHỦ — điểm dễ sai nhất
  MT5 trả thời gian theo giờ MÁY CHỦ của sàn, và mỗi sàn một kiểu:
    · Exness     : GMT+0 quanh năm                        → --mui-gio utc
    · VT Markets : GMT+2 mùa đông / GMT+3 mùa hè, đổi giờ theo lịch MỸ
                   (giờ máy chủ = giờ New York + 7)       → --mui-gio ny+7
  Trừ cố định 2 hay 3 giờ sẽ làm lệch 1 giờ suốt nửa năm mà không có dấu hiệu gì.
  Script chuyển theo đúng lịch đổi giờ, rồi đối chiếu với độ lệch đo từ tick hiện tại.

ĐỂ LẤY ĐƯỢC DỮ LIỆU XA
  Trên MT5: Tools → Options → Charts → "Max. bars in chart" = Unlimited, rồi khởi động
  lại terminal. Để mặc định thì terminal chỉ giữ một số nến gần nhất.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent

# Sàn → đường dẫn terminal thường gặp + quy ước giờ máy chủ
SAN = {
    "exness": {
        "terminal": [r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe",
                     r"C:\Program Files\Exness MetaTrader 5\terminal64.exe"],
        "mui_gio": "utc",
    },
    "vtmarkets": {
        "terminal": [r"C:\Program Files\VT Markets (Pty) MT5 Terminal\terminal64.exe"],
        "mui_gio": "ny+7",
    },
}

# Khung → (hằng số MT5, độ dài mỗi lần tải). Tải từng đoạn để không vượt giới hạn
# mỗi lần gọi và để terminal kịp tải lịch sử từ máy chủ.
KHUNG = {
    "M1": ("TIMEFRAME_M1", timedelta(days=15)),
    "M5": ("TIMEFRAME_M5", timedelta(days=60)),
    "M15": ("TIMEFRAME_M15", timedelta(days=180)),
    "M30": ("TIMEFRAME_M30", timedelta(days=365)),
    "H1": ("TIMEFRAME_H1", timedelta(days=730)),
    "H4": ("TIMEFRAME_H4", timedelta(days=3650)),
    "D1": ("TIMEFRAME_D1", timedelta(days=36500)),
    "W1": ("TIMEFRAME_W1", timedelta(days=36500)),
    "MN": ("TIMEFRAME_MN1", timedelta(days=36500)),
}


# ══════════════════════════════════════════════════════ múi giờ
def ve_utc(t_may_chu: pd.Series, mui_gio: str) -> pd.Series:
    """Giờ máy chủ → UTC (không kèm múi giờ).

    utc   : máy chủ chạy GMT+0 (Exness)
    ny+7  : máy chủ = giờ New York + 7, tức GMT+2 mùa đông / GMT+3 mùa hè, đổi giờ
            đúng ngày Mỹ đổi giờ (VT Markets và phần lớn các sàn)
    số    : lệch cố định so với UTC, ví dụ "2" hoặc "-5"
    """
    if mui_gio == "utc":
        return t_may_chu
    if mui_gio == "ny+7":
        ny = (t_may_chu - pd.Timedelta(hours=7)).dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        return ny.dt.tz_convert("UTC").dt.tz_localize(None)
    return t_may_chu - pd.Timedelta(hours=float(mui_gio))


def lech_ky_vong_bay_gio(mui_gio: str) -> float:
    """Giờ máy chủ − UTC tại thời điểm hiện tại, theo quy ước đã chọn."""
    if mui_gio == "utc":
        return 0.0
    if mui_gio == "ny+7":
        return pd.Timestamp.now(tz="America/New_York").utcoffset().total_seconds() / 3600 + 7
    return float(mui_gio)


def lech_do_duoc(mt5, ky_hieu: str):
    """Đo giờ máy chủ − UTC từ tick mới nhất. Thị trường đóng (tick cũ) thì trả None."""
    tick = mt5.symbol_info_tick(ky_hieu)
    if tick is None or not getattr(tick, "time", 0):
        return None
    lech = (tick.time - datetime.now(timezone.utc).timestamp()) / 3600
    if abs(lech - round(lech)) > 0.1 or abs(round(lech)) > 14:   # tick cũ hơn vài phút
        return None
    return float(round(lech))


# ══════════════════════════════════════════════════════ kết nối
def chon_terminal(san: str | None, duong_dan: str | None) -> str | None:
    if duong_dan:
        return duong_dan
    for p in SAN.get(san, {}).get("terminal", []):
        if Path(p).exists():
            return p
    return None       # để MT5 tự tìm terminal đang mở


def ket_noi(mt5, duong_dan_terminal: str | None):
    """Kết nối tới terminal đang đăng nhập. Không nhận và không lưu mật khẩu."""
    ok = mt5.initialize(path=duong_dan_terminal) if duong_dan_terminal else mt5.initialize()
    if not ok:
        raise SystemExit(
            "Không kết nối được MT5: %s\n"
            "→ Mở terminal MT5 của sàn, đăng nhập, rồi chạy lại. Nếu máy có nhiều terminal,\n"
            "  truyền --terminal \"...\\terminal64.exe\" của đúng sàn." % (mt5.last_error(),))
    tk, tm = mt5.account_info(), mt5.terminal_info()
    cong_ty = (getattr(tk, "company", "") or getattr(tm, "company", "") or "").strip()
    print("Đã kết nối: %s | máy chủ %s | terminal %s"
          % (cong_ty or "?", getattr(tk, "server", "?"), getattr(tm, "path", "?")))
    return cong_ty


def tim_ky_hieu(mt5, ten: str) -> str:
    """Mỗi sàn đặt hậu tố riêng: XAUUSD, XAUUSDm, XAUUSDc, XAUUSD.s, XAUUSD-STD…"""
    if mt5.symbol_info(ten) is not None:
        chon = ten
    else:
        ung_vien = [s.name for s in (mt5.symbols_get(group="*%s*" % ten) or [])
                    if s.name.upper().startswith(ten.upper())]
        if not ung_vien:
            raise SystemExit("Không tìm thấy symbol %s trên tài khoản này." % ten)
        chon = sorted(ung_vien, key=len)[0]
        print("Symbol %s không có; dùng %s (các biến thể: %s)" % (ten, chon, ", ".join(ung_vien)))
    if not mt5.symbol_select(chon, True):              # đưa vào Market Watch
        raise SystemExit("Không bật được symbol %s: %s" % (chon, mt5.last_error()))
    return chon


# ══════════════════════════════════════════════════════ tải dữ liệu
def tai_mot_khung(mt5, ky_hieu: str, khung: str, tu: datetime, den: datetime,
                  mui_gio: str) -> pd.DataFrame:
    hang_so, buoc = KHUNG[khung]
    tf = getattr(mt5, hang_so)
    phan, a = [], tu
    while a < den:
        b = min(a + buoc, den)
        rates = None
        for lan in range(3):                           # terminal có thể đang tải lịch sử
            rates = mt5.copy_rates_range(ky_hieu, tf, a, b)
            if rates is not None and len(rates):
                break
            time.sleep(1.0 + lan)
        if rates is not None and len(rates):
            phan.append(pd.DataFrame(rates))
        print("  %s %s → %s : %6d nến" % (khung, a.date(), b.date(),
                                         0 if rates is None else len(rates)), flush=True)
        a = b

    if not phan:
        raise SystemExit("Không tải được nến %s nào: %s" % (khung, mt5.last_error()))
    d = pd.concat(phan, ignore_index=True)
    d["time"] = ve_utc(pd.to_datetime(d["time"], unit="s"), mui_gio)
    d = d.rename(columns={"tick_volume": "volume"}).dropna(subset=["time"])
    return (d[["time", "open", "high", "low", "close", "volume"]]
            .drop_duplicates("time").sort_values("time").reset_index(drop=True))


def kiem_tra(d: pd.DataFrame, khung: str, tu: datetime) -> None:
    sai = ((d["high"] < d[["open", "close"]].max(axis=1)) |
           (d["low"] > d[["open", "close"]].min(axis=1))).sum()
    buoc = d["time"].diff().dt.total_seconds().div(60)
    print("  %s: %d nến | %s → %s | nến sai hình học: %d | khoảng trống lớn nhất: %.1f ngày"
          % (khung, len(d), d["time"].min(), d["time"].max(), sai, buoc.max() / 1440))
    if d["time"].min() > pd.Timestamp(tu.replace(tzinfo=None)) + pd.Timedelta(days=7):
        print("  ⚠ Dữ liệu bắt đầu muộn hơn ngày yêu cầu (%s). Tăng 'Max. bars in chart' "
              "lên Unlimited, hoặc sàn không có lịch sử xa hơn." % tu.date())
    so_nam = d.groupby(d["time"].dt.year).size()
    print("  Số nến mỗi năm: " + ", ".join("%d:%d" % kv for kv in so_nam.items()))


def cao(khung_ds, tu, den, san=None, ky_hieu="XAUUSD", terminal=None, thu_muc=None,
        mui_gio=None):
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise SystemExit("Chưa cài thư viện: pip install MetaTrader5  (chỉ có trên Windows)")

    thu_muc = Path(thu_muc) if thu_muc else GOC / "data" / "raw"
    thu_muc.mkdir(parents=True, exist_ok=True)
    cong_ty = ket_noi(mt5, chon_terminal(san, terminal))
    try:
        if san is None:                                  # đoán sàn từ tên công ty
            ten = cong_ty.lower().replace(" ", "")
            san = next((k for k in SAN if k in ten), "mt5")
        elif san in SAN and san not in cong_ty.lower().replace(" ", ""):
            print("⚠ Chọn --san %s nhưng terminal đang kết nối là '%s'." % (san, cong_ty))
        mui_gio = mui_gio or SAN.get(san, {}).get("mui_gio")

        sym = tim_ky_hieu(mt5, ky_hieu)

        # Đối chiếu quy ước giờ máy chủ với độ lệch đo từ tick hiện tại
        do = lech_do_duoc(mt5, sym)
        if mui_gio is None:
            if do is None:
                raise SystemExit("Không biết giờ máy chủ của sàn này và thị trường đang đóng nên "
                                 "không đo được. Truyền --mui-gio utc | ny+7 | <số giờ>.")
            mui_gio = str(do)
            print("Giờ máy chủ đo được: UTC%+g → dùng lệch cố định %s giờ" % (do, mui_gio))
        ky_vong = lech_ky_vong_bay_gio(mui_gio)
        if do is None:
            print("Quy ước giờ máy chủ: %s (hiện tại UTC%+g). Thị trường đang đóng nên chưa "
                  "đối chiếu được với tick." % (mui_gio, ky_vong))
        elif do == ky_vong:
            print("✓ Giờ máy chủ UTC%+g khớp quy ước '%s'." % (do, mui_gio))
        else:
            raise SystemExit("Giờ máy chủ đo được UTC%+g KHÁC quy ước '%s' (UTC%+g). "
                             "Truyền --mui-gio cho đúng rồi chạy lại." % (do, mui_gio, ky_vong))

        ra = []
        for khung in khung_ds:
            print("\nTải %s %s từ %s đến %s" % (sym, khung, tu.date(), den.date()))
            d = tai_mot_khung(mt5, sym, khung, tu, den, mui_gio)
            kiem_tra(d, khung, tu)
            p = thu_muc / ("%s_%s_%s.csv" % (san, ky_hieu.upper(), khung))
            d.to_csv(p, index=False)
            print("  → Đã lưu %s" % p)
            ra.append(p)
        return ra
    finally:
        mt5.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cào dữ liệu XAUUSD (time + OHLCV) từ MT5")
    ap.add_argument("--san", choices=list(SAN), default=None,
                    help="exness | vtmarkets; bỏ trống thì đoán theo terminal đang kết nối")
    ap.add_argument("--ky-hieu", default="XAUUSD")
    ap.add_argument("--khung", nargs="+", default=["H1"], choices=list(KHUNG))
    ap.add_argument("--tu", default="2015-01-01", help="ngày bắt đầu (YYYY-MM-DD)")
    ap.add_argument("--den", default=None, help="ngày kết thúc, mặc định là hôm nay")
    ap.add_argument("--terminal", default=None, help="đường dẫn terminal64.exe")
    ap.add_argument("--thu-muc", default=None, help="thư mục ghi tệp, mặc định data/raw")
    ap.add_argument("--mui-gio", default=None,
                    help="giờ máy chủ: utc | ny+7 | <số giờ lệch>; mặc định theo --san")
    a = ap.parse_args(argv)

    # MT5 yêu cầu datetime có múi giờ UTC
    tu = datetime.fromisoformat(a.tu).replace(tzinfo=timezone.utc)
    den = (datetime.fromisoformat(a.den).replace(tzinfo=timezone.utc) if a.den
           else datetime.now(timezone.utc))
    cao(a.khung, tu, den, a.san, a.ky_hieu, a.terminal, a.thu_muc, a.mui_gio)


if __name__ == "__main__":
    main()
