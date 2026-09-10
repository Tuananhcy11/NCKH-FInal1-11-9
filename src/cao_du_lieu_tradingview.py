# -*- coding: utf-8 -*-
"""Cào dữ liệu XAUUSD (time + OHLCV) từ TradingView — mặc định OANDA:XAUUSD.

    python src/cao_du_lieu_tradingview.py                        # OANDA, H1, tối đa số nến
    python src/cao_du_lieu_tradingview.py --khung M15 H1 D1
    python src/cao_du_lieu_tradingview.py --san FX --khung D1     # FXCM, D1 từ 1970

Chạy được cả trên máy tính lẫn Google Colab:  pip install pandas python-dateutil
  pip install https://github.com/rongardF/tvdatafeed/archive/refs/heads/main.zip

ĐẦU RA
  data/raw/tradingview_<SAN>_XAUUSD_<khung>.csv — đúng 6 cột: time, open, high, low,
  close, volume. time là giờ MỞ nến theo UTC; volume là tick volume của nguồn.

GIỚI HẠN CẦN BIẾT
  · TradingView KHÔNG có API chính thức. Thư viện tvDatafeed kết nối vào websocket
    của trang web; điều khoản của TradingView không cho phép tải dữ liệu tự động.
    Chỉ nên dùng cho nghiên cứu cá nhân.
  · Không đăng nhập: mỗi khung chỉ lấy được khoảng 10.000 NẾN GẦN NHẤT.
    H1 ≈ 1,7 năm, M15 ≈ 5 tháng, D1 ≈ vài chục năm. Muốn xa hơn thì cần tài khoản
    TradingView trả phí: đặt biến môi trường TV_USERNAME và TV_PASSWORD trước khi
    chạy (script chỉ đọc biến môi trường, không in và không lưu mật khẩu).
  · tvDatafeed trả thời gian theo GIỜ CỦA MÁY ĐANG CHẠY (máy ở Việt Nam là GMT+7,
    Colab là UTC). Script chuyển về UTC theo múi giờ của máy, để tệp giống nhau dù
    chạy ở đâu.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dateutil import tz

GOC = Path(__file__).resolve().parent.parent

# Khung → tên hằng số trong tvDatafeed.Interval
KHUNG = {"M1": "in_1_minute", "M5": "in_5_minute", "M15": "in_15_minute",
         "M30": "in_30_minute", "H1": "in_1_hour", "H2": "in_2_hour", "H4": "in_4_hour",
         "D1": "in_daily", "W1": "in_weekly", "MN": "in_monthly"}


def ket_noi():
    try:
        from tvDatafeed import TvDatafeed
    except ImportError:
        raise SystemExit("Chưa cài thư viện:\n  pip install "
                         "https://github.com/rongardF/tvdatafeed/archive/refs/heads/main.zip")
    logging.getLogger("tvDatafeed").setLevel(logging.CRITICAL)
    ten, mat_khau = os.environ.get("TV_USERNAME"), os.environ.get("TV_PASSWORD")
    if ten and mat_khau:
        print("Đăng nhập TradingView bằng tài khoản trong biến môi trường TV_USERNAME.")
        return TvDatafeed(ten, mat_khau)
    print("Không đăng nhập — mỗi khung tối đa khoảng 10.000 nến gần nhất.")
    return TvDatafeed()


def ve_utc(chi_so: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Giờ của máy (tvDatafeed dùng datetime.fromtimestamp) → UTC không kèm múi giờ.

    Dùng múi giờ hệ thống có tính cả giờ mùa hè, nên đúng cả với máy ở nước có đổi giờ.
    """
    if chi_so.tz is not None:
        return chi_so.tz_convert("UTC").tz_localize(None)
    return (chi_so.tz_localize(tz.tzlocal(), ambiguous="NaT", nonexistent="shift_forward")
            .tz_convert("UTC").tz_localize(None))


def tai_mot_khung(tv, san: str, ky_hieu: str, khung: str, so_nen: int) -> pd.DataFrame:
    from tvDatafeed import Interval
    iv = getattr(Interval, KHUNG[khung])
    n = so_nen
    for lan in range(6):
        try:
            d = tv.get_hist(symbol=ky_hieu, exchange=san, interval=iv, n_bars=n)
        except OSError:
            # Trên Windows, mốc thời gian trước năm 1970 làm fromtimestamp báo lỗi
            # "Invalid argument" → xin ít nến hơn.
            d, n = None, n // 2
            print("  %s: nguồn có dữ liệu trước 1970, xin lại %d nến" % (khung, n))
            continue
        if d is not None and not d.empty:
            break
        print("  %s: chưa có dữ liệu (lần %d), thử lại..." % (khung, lan + 1))
        time.sleep(3 + 2 * lan)
    else:
        raise SystemExit("Không tải được %s:%s khung %s — kiểm tra lại mã sàn/symbol."
                         % (san, ky_hieu, khung))

    d = d.copy()
    d.index = ve_utc(pd.DatetimeIndex(d.index))
    d = d.rename_axis("time").reset_index()
    return (d[["time", "open", "high", "low", "close", "volume"]]
            .dropna(subset=["time", "close"]).drop_duplicates("time")
            .sort_values("time").reset_index(drop=True))


def kiem_tra(d: pd.DataFrame, khung: str) -> None:
    sai = ((d["high"] < d[["open", "close"]].max(axis=1)) |
           (d["low"] > d[["open", "close"]].min(axis=1))).sum()
    print("  %s: %d nến | %s → %s (UTC) | nến sai hình học: %d"
          % (khung, len(d), d["time"].min(), d["time"].max(), sai))
    so_nam = d.groupby(d["time"].dt.year).size()
    print("  Số nến mỗi năm: " + ", ".join("%d:%d" % kv for kv in so_nam.tail(12).items()))


def cao(khung_ds, san="OANDA", ky_hieu="XAUUSD", so_nen=20000, thu_muc=None):
    thu_muc = Path(thu_muc) if thu_muc else GOC / "data" / "raw"
    thu_muc.mkdir(parents=True, exist_ok=True)
    tv = ket_noi()
    ra = []
    for khung in khung_ds:
        print("\nTải %s:%s khung %s" % (san, ky_hieu, khung))
        d = tai_mot_khung(tv, san, ky_hieu, khung, so_nen)
        kiem_tra(d, khung)
        p = thu_muc / ("tradingview_%s_%s_%s.csv" % (san.upper(), ky_hieu.upper(), khung))
        d.to_csv(p, index=False)
        print("  → Đã lưu %s" % p)
        ra.append(p)
    return ra


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cào dữ liệu XAUUSD (time + OHLCV) từ TradingView")
    ap.add_argument("--san", default="OANDA", help="mã nguồn trên TradingView: OANDA, FX, VANTAGE…")
    ap.add_argument("--ky-hieu", default="XAUUSD")
    ap.add_argument("--khung", nargs="+", default=["H1"], choices=list(KHUNG))
    ap.add_argument("--so-nen", type=int, default=20000,
                    help="số nến muốn lấy; không đăng nhập thì nguồn tự giới hạn khoảng 10.000")
    ap.add_argument("--thu-muc", default=None, help="thư mục ghi tệp, mặc định data/raw")
    a = ap.parse_args(argv)
    cao(a.khung, a.san, a.ky_hieu, a.so_nen, a.thu_muc)


if __name__ == "__main__":
    main()
