# -*- coding: utf-8 -*-
"""Ghép 5 notebook Colab thành MỘT notebook chạy trọn quy trình.

    python src/tao_nb_toan_bo.py

    00 Tải Dukascopy → 01 Làm sạch → dukascopy_XAUUSD_<khung>_sach.csv → NB1 ∥ NB2 → NB3

Đọc các notebook đã sinh trong "Google colab/" (chạy tao_nb_lam_sach.py và
sua_notebook_colab.py trước nếu vừa sửa chúng) và ghi
"Google colab/NCKH_XAUUSD_Toan_bo_quy_trinh.ipynb". Ô của từng phần được giữ nguyên
văn, chỉ thay các chỗ nối đầu vào. Giữa hai phần có một ô xóa biến của phần trước để
mỗi phần chạy như một notebook độc lập (giống khi mở riêng từng notebook).
"""
import copy
import io
import json
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
TM = GOC / "Google colab"
DICH = TM / "NCKH_XAUUSD_Toan_bo_quy_trinh.ipynb"


def dong(s):
    s = s.strip("\n").split("\n")
    return [x + "\n" for x in s[:-1]] + [s[-1]]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": dong(s)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dong(s)}


def doc(ten):
    return json.load(io.open(TM / ten, encoding="utf-8"))["cells"]


def sach(o):
    o = copy.deepcopy(o)
    if o["cell_type"] == "code":
        o["outputs"], o["execution_count"] = [], None
    return o


def nguon(o):
    return "".join(o["source"])


def thay(o, cu, moi):
    s = nguon(o)
    assert s.count(cu) == 1, "Không tìm thấy đúng một chỗ %r — notebook gốc đã đổi?" % cu
    o = sach(o)
    o["source"] = dong(s.replace(cu, moi))
    return o


GIOI_THIEU = r'''
# NCKH XAU/USD — Toàn bộ quy trình trong một notebook

```
00 Tải Dukascopy → 01 Làm sạch → dukascopy_XAUUSD_<khung>_sach.csv → NB1 ∥ NB2 → NB3
```

| Phần | Nội dung | Đầu ra (trong `MyDrive/Data_NghienCuu`) |
|---|---|---|
| 00 | Tải XAUUSD 2015–2025 từ Dukascopy (time + OHLCV, volume thật) | `dukascopy_XAUUSD_<khung>.csv` |
| 01 | Làm sạch: ngày giờ, trùng lặp, khuyết thiếu, sai cấu trúc OHLC, kiểm tra chéo | `dukascopy_XAUUSD_<khung>_sach.csv` + báo cáo |
| NB1 | Chỉ báo kỹ thuật → **dự báo xu hướng** uptrend / sideway / downtrend (1 / 0 / −1) | `gold_price_technical_signal.csv` |
| NB2 | **XGBoost walk-forward 2020 → 2025** (mỗi năm dự báo bằng mô hình chỉ học dữ liệu trước năm đó) → **dự báo xu hướng** | `gold_model_signals.csv`, `danh_gia_walk_forward.csv` |
| NB3 | **9 chiến lược** × 5 nguồn (MA, RSI, MACD, TH, XGB) → lệnh **BUY (1) / SELL (0)** với SL:TP theo R:R; backtest kiểu MT5 2020–2025, kết quả từng năm, đối chứng ngẫu nhiên, đối đầu XGB với từng luật kỹ thuật | `backtest_comparison_report.csv`, `ket_qua_theo_nam.csv`, `doi_dau_xgb_vs_ky_thuat.csv` |

**Cách chạy:** chỉnh ô *Cấu hình chung* bên dưới rồi bấm **Thời gian chạy → Chạy tất cả**
(Run all) và cấp quyền Google Drive. NB1 và NB2 vẫn **độc lập** (cùng đọc tệp sạch, không
dùng kết quả của nhau) — trong bản gộp chúng chỉ chạy lần lượt.

- Đã tải dữ liệu từ trước thì giữ `BO_QUA_TAI_NEU_DA_CO = True`: phần 00 tự bỏ qua.
- `KHUNG_CHAY` là khung dùng cho NB1, NB2, NB3 (phải nằm trong `KHUNG_TAI`).
- Mỗi phần bắt đầu bằng một ô "ranh giới" xóa biến của phần trước, nên có thể chạy lại
  riêng từ đầu một phần bất kỳ (sau khi đã chạy ô *Cấu hình chung*).
'''

CAU_HINH = r'''
#@title Cấu hình chung
KHUNG_TAI = 'M30 H1 H4 D1'       #@param {type:'string'}
KHUNG_CHAY = 'H1'                #@param ['M30', 'H1', 'H4', 'D1']
BO_QUA_TAI_NEU_DA_CO = True      #@param {type:'boolean'}

import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    CAU_HINH_THU_MUC = '/content/drive/MyDrive/Data_NghienCuu'
except ImportError:                                  # chạy trên máy tính
    CAU_HINH_THU_MUC = os.environ.get('NCKH_THU_MUC', '.')
os.makedirs(CAU_HINH_THU_MUC, exist_ok=True)
assert KHUNG_CHAY in KHUNG_TAI.split(), 'KHUNG_CHAY phải nằm trong KHUNG_TAI'
print('Thư mục dữ liệu:', CAU_HINH_THU_MUC)
print('Tải/làm sạch:', KHUNG_TAI, '| NB1–NB3 chạy trên khung:', KHUNG_CHAY)
'''

RANH_GIOI = r'''
# ── Ranh giới phần: xóa biến của phần trước (như mở notebook mới), giữ cấu hình chung
import matplotlib.pyplot as _plt
_plt.close('all')
_GIU = {'KHUNG_TAI', 'KHUNG_CHAY', 'BO_QUA_TAI_NEU_DA_CO', 'CAU_HINH_THU_MUC',
        'In', 'Out', 'get_ipython', 'exit', 'quit', 'display'}
for _t in [t for t in list(globals()) if not t.startswith('_') and t not in _GIU]:
    del globals()[_t]
print('Bắt đầu %s' % {TIEU_DE})
'''

CHAY_TAI = r'''
import os
KHUNG = KHUNG_TAI
THU_MUC = CAU_HINH_THU_MUC
tep_can = [os.path.join(THU_MUC, 'dukascopy_XAUUSD_%s.csv' % k) for k in KHUNG.split()]
if BO_QUA_TAI_NEU_DA_CO and all(os.path.exists(p) for p in tep_can):
    print('Đã có đủ tệp %s trong %s → bỏ qua bước tải.' % (KHUNG, THU_MUC))
else:
    BO_DEM = '/content/_bo_dem_dukascopy'
    get_ipython().system('mkdir -p "%s" && cp -rn "%s/_bo_dem_dukascopy/." "%s/" 2>/dev/null; true'
                         % (BO_DEM, THU_MUC, BO_DEM))
    get_ipython().system('python cao_du_lieu_dukascopy.py --khung %s --thu-muc "%s" --bo-dem "%s" '
                         '--luu-thang "%s/_luu_thang_dukascopy" --luong 8 --toc-do 10'
                         % (KHUNG, THU_MUC, BO_DEM, THU_MUC))
    thieu = [os.path.basename(p) for p in tep_can if not os.path.exists(p)]
    if thieu:
        raise RuntimeError('Tải chưa xong (%s). Xem thông báo phía trên, đợi vài phút rồi chạy lại '
                           'ô này — chỉ tải phần còn thiếu.' % thieu)
'''


def phan(tieu_de, mo_ta):
    return [md("---\n# %s\n\n%s" % (tieu_de, mo_ta)), code(RANH_GIOI.replace("{TIEU_DE}", repr(tieu_de)))]


def tao():
    c00, c01 = doc("00_Tai_du_lieu_Dukascopy.ipynb"), doc("01_Lam_sach_du_lieu.ipynb")
    nb1, nb2 = doc("XAY_DUNG_CAC_CHI_SO_KY_THUAT.ipynb"), doc("3_model.ipynb")
    nb3 = doc("nckh_ptt(muc3).ipynb")

    # 00: bỏ ô gắn Drive riêng và ô chạy riêng, thay bằng ô chạy theo cấu hình chung
    assert nguon(c00[2]).startswith("%%writefile cao_du_lieu_dukascopy.py")
    assert "!python cao_du_lieu_dukascopy.py" in nguon(c00[3])
    p00 = [sach(c00[0]), sach(c00[2]), code(CHAY_TAI), sach(c00[4])]

    # 01: thư mục và tệp đầu vào lấy từ cấu hình chung
    p01 = []
    for o in c01:
        s = nguon(o)
        if o["cell_type"] == "code" and "THU_MUC = os.environ.get('NCKH_DU_LIEU', 'data/raw')" in s:
            o = thay(o, "THU_MUC = os.environ.get('NCKH_DU_LIEU', 'data/raw')", "THU_MUC = CAU_HINH_THU_MUC")
        elif o["cell_type"] == "code" and "#@title Chọn dữ liệu cần làm sạch" in s:
            o = thay(o, "NGUON = 'tai_len'        #@param ['tai_len', 'drive']",
                     "NGUON = 'drive'          # bản gộp: dùng tệp phần 00 vừa tải")
            o = thay(o, "TEN_TEP_DRIVE = ''       #@param {type:'string'}",
                     "TEN_TEP_DRIVE = ', '.join('dukascopy_XAUUSD_%s.csv' % k for k in KHUNG_TAI.split())")
        p01.append(sach(o))

    # NB1, NB2: đầu vào là tệp sạch của khung KHUNG_CHAY
    def noi_dau_vao(cells):
        ra, dem = [], 0
        for o in cells:
            if o["cell_type"] == "code" and nguon(o).lstrip().startswith("DUONG_DAN_DU_LIEU = ''"):
                o = thay(o, "DUONG_DAN_DU_LIEU = ''",
                         "DUONG_DAN_DU_LIEU = os.path.join(CAU_HINH_THU_MUC, "
                         "'dukascopy_XAUUSD_%s_sach.csv' % KHUNG_CHAY)   # tệp sạch từ phần 01")
                dem += 1
            ra.append(sach(o))
        assert dem == 1, "Không tìm thấy ô nhập dữ liệu"
        return ra

    cells = [md(GIOI_THIEU), code(CAU_HINH)]
    cells += phan("PHẦN 00 — Tải dữ liệu Dukascopy",
                  "Bỏ qua tự động nếu đã có đủ tệp và `BO_QUA_TAI_NEU_DA_CO = True`.") + p00
    cells += phan("PHẦN 01 — Làm sạch dữ liệu",
                  "Làm sạch mọi khung trong `KHUNG_TAI`, ghi `dukascopy_XAUUSD_<khung>_sach.csv`.") + p01
    cells += phan("PHẦN NB1 — Chỉ báo kỹ thuật → dự báo xu hướng",
                  "Đọc tệp sạch của `KHUNG_CHAY`. Đầu ra: uptrend / sideway / downtrend "
                  "(1 / 0 / −1), chưa phải lệnh.") + noi_dau_vao(nb1)
    cells += phan("PHẦN NB2 — XGBoost walk-forward 2020–2025 → dự báo xu hướng",
                  "Độc lập với NB1: tự đọc lại tệp sạch, không dùng kết quả NB1. Mỗi năm 2020–2025 "
                  "dự báo bằng mô hình chỉ học dữ liệu trước năm đó. Đầu ra: uptrend / sideway / downtrend.") + noi_dau_vao(nb2)
    cells += phan("PHẦN NB3 — 9 chiến lược → BUY / SELL → backtest 2020–2025",
                  "Ghép dự báo của NB1 và NB2 theo thời gian, áp 9 chiến lược (SL:TP theo R:R, đánh 2 chiều) "
                  "cho 4 luật kỹ thuật và XGBoost, xem kết quả từng năm, đối chứng ngẫu nhiên, rồi đối đầu "
                  "XGBoost với từng luật kỹ thuật.") + [sach(o) for o in nb3]

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": []}, "accelerator": "GPU",
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    with io.open(DICH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return nb


if __name__ == "__main__":
    nb = tao()
    lai = json.load(io.open(DICH, encoding="utf-8"))
    for o in lai["cells"]:
        s = nguon(o)
        if o["cell_type"] == "code" and not s.startswith("%%"):
            compile("\n".join(x for x in s.split("\n") if not x.lstrip().startswith("!")), "o", "exec")
    print("Đã ghi %s — %d ô (%d ô mã)" % (DICH.name, len(lai["cells"]),
                                          sum(o["cell_type"] == "code" for o in lai["cells"])))
