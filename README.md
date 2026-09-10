# Hệ thống trading hai chiều XAU/USD — Long & Short (v2)

Pipeline nghiên cứu định lượng cho thị trường **Forex / CFD margin**: ma trận đa
khung thời gian → gán nhãn Triple Barrier hai chiều → LightGBM đa lớp → **bộ lọc
ba lớp** → backtest khớp lệnh Bid/Ask có position sizing động và break-even.

## Ba lỗi của v1 mà v2 sửa

Bản v1 chạy trên 65.125 nến H1 thật (2015–2025) và **lỗ −4.281 USD**. Ba nguyên
nhân đo được, và cách sửa tương ứng:

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| Overtrading — 2.052 lệnh | Ngưỡng 0,55 quá thấp, không lọc phiên | Ngưỡng 0,65 / 0,68 + bộ lọc ba lớp |
| Short lỗ −4.996 USD (Long lãi +715) | Đánh ngược xu hướng tăng dài hạn của vàng | Bộ lọc HTF — cấm Short khi giá trên EMA400 |
| Ít cơ hội chạm TP | Rào 8 nến quá chật cho TP = 1,5·ATR | Nới lên 16 nến, hòa vốn sớm tại 0,8R |

## Bộ lọc ba lớp

```
Lớp 1  Xác suất ML   P(Long) >= 0,65   hoặc   P(Short) >= 0,68
Lớp 2  Xu hướng HTF  Long  chỉ khi giá TRÊN  EMA400
                     Short chỉ khi giá DƯỚI  EMA400
Lớp 3  Phiên         Chỉ mở lệnh 07:00–18:00 UTC
```

Ngưỡng Short cao hơn Long là **cố ý**: vàng có xu hướng tăng dài hạn nên mô hình
trung lập tự nhiên sinh quá nhiều tín hiệu Short sai.

Mục 7 của notebook đo **thiệt hại của từng lớp lọc** bằng cách chạy lại backtest
với từng tổ hợp. Không có phép đo này thì "lọc bớt lệnh xấu" và "lọc mất cả lệnh
tốt" trông giống hệt nhau.

## Chạy

Mở [`notebooks/XAUUSD_Dual_Direction_Model.ipynb`](notebooks/XAUUSD_Dual_Direction_Model.ipynb)
trên Google Colab và bấm **Run all**. Không cần chuẩn bị gì: nếu chưa có tệp dữ
liệu, `ForexDataEngine` tự sinh dữ liệu mô phỏng theo phân phối giá vàng.

Chạy dữ liệu thật — sửa đúng một dòng ở ô cấu hình:

```python
CH = CauHinh(duong_dan="duong/dan/xauusd_m15.csv")
```

Tệp cần các cột `datetime, open, high, low, close, volume` (`.csv` hoặc
`.parquet`). Thiếu `volume` thì lớp dữ liệu tự điền 1.

Sinh lại notebook sau khi sửa mã:

```bash
python src/build_notebook.py
```

## Kiến trúc — năm lớp độc lập

| Lớp | Trách nhiệm |
|---|---|
| `ForexDataEngine` | Tải hoặc sinh dữ liệu, chuẩn hóa UTC, kiểm định hình học OHLC |
| `ForexFeatureExtractor` | Ma trận 3 tầng thời gian, toàn bộ `.shift(1)`; trả thêm `ngu_canh` cho bộ lọc |
| `ForexTripleBarrierLabeler` | Nhãn 3 lớp `+1 / -1 / 0`, hai kịch bản độc lập có spread |
| `ForexMLModel` | LightGBM đa lớp, `TimeSeriesSplit` có purging, **bộ lọc ba lớp** kèm nhật ký |
| `ForexCFDBacktester` | Khớp lệnh Bid/Ask, sizing động [0,01–10,0 lot], break-even tại 0,8R |

## Bốn quyết định thiết kế đáng chú ý

**Bất đối xứng Bid/Ask được mô phỏng đúng.** Lệnh BUY vào ở Ask nhưng thoát ở
Bid; lệnh SELL vào ở Bid nhưng thoát ở Ask. Mô phỏng bằng một chuỗi giá duy nhất
sẽ tính thiếu một nửa spread ở mọi lệnh và mọi kết quả đều lạc quan giả.

**Quy ước bảo thủ khi mơ hồ trong nến.** Khi một nến chạm cả SL lẫn TP, dữ liệu
OHLC không cho biết cái nào đến trước. Cả khâu gán nhãn lẫn khâu backtest đều
**giả định SL đến trước**. Giả định ngược lại sẽ thổi phồng tỷ lệ thắng một cách
có hệ thống mà không bao giờ lộ ra.

**Break-even có bù spread.** Kéo SL về đúng "giá vào lệnh" là sai — lệnh vẫn lỗ
nhẹ. BUY vào ở Ask = E thì hòa vốn khi Bid = E; SELL vào ở Bid = E thì hòa vốn
khi Ask = E, tức SL đặt tại `E − spread`.

**Purging giữa train và test.** Nhãn tại nến `t` dùng thông tin tới `t + 8`, nên
`ForexMLModel` cắt bỏ 8 nến cuối mỗi tập huấn luyện. Không cắt thì những nhãn
cuối đã nhìn vào vùng kiểm định.

## Chống rò rỉ dữ liệu

Mọi đặc trưng `.shift(1)`. Ngoại lệ hợp lệ duy nhất là nhóm thời gian (giờ, thứ,
phiên) — chúng được biết trước cả khi nến bắt đầu.

Đặc trưng phiên Á chỉ được công bố **từ 07:00 UTC**, khi phiên đã đóng. Công bố
sớm hơn là nhìn tương lai: lúc 01:00 chưa thể biết đỉnh của cả phiên kéo dài tới
07:00.

`ForexFeatureExtractor.kiem_tra_ro_ri()` tự đối chiếu lại: so tương quan của từng
đặc trưng với lợi suất nến **hiện tại**. Đặc trưng hợp lệ chỉ biết tới nến `t−1`
nên tương quan phải gần 0.

## Đọc kết quả cho đúng

Trên dữ liệu mô phỏng, mọi con số chỉ chứng minh **pipeline chạy đúng** — không
nói gì về khả năng sinh lợi. Dữ liệu được sinh từ bước ngẫu nhiên có cụm biến
động, về lý thuyết **không tồn tại** mẫu hình khai thác được. Backtest cho lãi
lớn trên dữ liệu này là dấu hiệu rò rỉ, không phải mô hình giỏi.

Ba ngưỡng đáng ngờ khi chạy dữ liệu thật:

| Dấu hiệu | Ngưỡng | Ý nghĩa |
|---|---|---|
| Win rate | > 65 % với TP/SL = 1,5 | Thường là rò rỉ hoặc lỗi khớp lệnh |
| Long vs Short | một chiều lãi, một chiều lỗ nặng | Chỉ cưỡi xu hướng, không có ưu thế thật |
| Độ phủ tín hiệu | > 50 % | Ngưỡng quá thấp, trả spread cho quá nhiều lệnh |

## Cấu trúc

```
src/build_notebook.py     trình sinh notebook (sửa mã ở đây)
notebooks/                notebook Colab
```
