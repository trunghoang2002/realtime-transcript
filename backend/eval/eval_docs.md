# 📊 Evaluation Script Documentation

Script đánh giá chất lượng transcription bằng cách so sánh kết quả dự đoán với ground truth, sử dụng các metrics WER (Word Error Rate) và CER (Character Error Rate).

## 📋 Mục đích

Script `eval.py` được sử dụng để:
- Đánh giá độ chính xác của các model transcription (Whisper, SenseVoice)
- So sánh hiệu suất giữa các model khác nhau
- So sánh hiệu suất giữa realtime và upload transcription
- Đánh giá chất lượng trên các ngôn ngữ khác nhau (tiếng Nhật, tiếng Anh, ...)

## 🔧 Dependencies

Script yêu cầu thư viện `jiwer` để tính toán WER và CER:

```bash
pip install jiwer
```

## 📁 Cấu trúc thư mục

```
eval/
├── eval.py                    # Script đánh giá chính
├── eval.md                    # Tài liệu này
├── ja/                        # Test cases tiếng Nhật
│   ├── ground_truth_full.txt  # Ground truth (full transcript)
│   ├── ground_truth.txt       # Ground truth (segments)
│   ├── realtime/             # Kết quả realtime transcription
│   │   ├── whisper-small/
│   │   │   ├── predict_full.txt
│   │   │   └── predict.txt
│   │   ├── whisper-medium/
│   │   └── sensevoice-small/
│   └── upload/                # Kết quả upload transcription
│       ├── whisper-small/
│       └── ...
└── en/                        # Test cases tiếng Anh
    ├── ground_truth_full.txt
    ├── realtime/
    └── upload/
```

## 📊 Metrics

### WER (Word Error Rate)
- **Định nghĩa**: Tỷ lệ lỗi từ (số từ bị thay thế, xóa, hoặc thêm vào)
- **Công thức**: `WER = (S + D + I) / N`
  - S: Số từ bị thay thế (Substitutions)
  - D: Số từ bị xóa (Deletions)
  - I: Số từ bị thêm vào (Insertions)
  - N: Tổng số từ trong ground truth
- **Giá trị**: 0.0 = hoàn hảo, càng cao càng kém

### CER (Character Error Rate)
- **Định nghĩa**: Tỷ lệ lỗi ký tự (số ký tự bị thay thế, xóa, hoặc thêm vào)
- **Công thức**: `CER = (S + D + I) / N`
  - Tương tự WER nhưng tính theo ký tự thay vì từ
- **Giá trị**: 0.0 = hoàn hảo, càng cao càng kém
- **Hữu ích cho**: Ngôn ngữ không có khoảng trắng giữa từ (như tiếng Nhật, tiếng Trung)

## 🔄 Quy trình Normalize

Trước khi tính toán metrics, text được normalize theo các bước sau:

1. **Lowercase**: Chuyển tất cả về chữ thường
   ```python
   ground_truth = ground_truth.lower()
   prediction = prediction.lower()
   ```

2. **Loại bỏ punctuation**: Xóa các dấu câu
   - Tiếng Anh: `,.~!?`
   - Tiếng Nhật: `・、。「」『』（）【】〈〉《》！？〜～…‥―：；※`
   ```python
   pattern = r"[,.~!?・、。「」『』（）【】〈〉《》！？〜～…‥―：；※]"
   ```

3. **Normalize whitespace**: Chuẩn hóa khoảng trắng (nhiều space thành 1 space)
   ```python
   re.sub(r"\s+", " ", text).strip()
   ```

## 🚀 Cách sử dụng

### 1. Chuẩn bị dữ liệu

Đảm bảo bạn có:
- File `ground_truth_full.txt` trong thư mục `{lang}/`
- File `predict_full.txt` trong thư mục `{lang}/{type}/{model}/`

### 2. Cấu hình script

Chỉnh sửa các biến trong `eval.py`:

```python
lang = "ja"              # Ngôn ngữ: "ja" | "en"
type = "realtime"        # Loại: "realtime" | "upload"
model = "sensevoice-small"  # Model: "sensevoice-small" | "whisper-small" | "whisper-medium"
```

### 3. Chạy script

```bash
cd backend/eval
python eval.py
```

### 4. Xem kết quả

Script sẽ in ra console:
```
WER: 0.1234
CER: 0.0567
```

## 📝 Ví dụ

### Ví dụ 1: Đánh giá Whisper Small cho realtime transcription tiếng Nhật

```python
lang = "ja"
type = "realtime"
model = "whisper-small"
```

### Ví dụ 2: Đánh giá SenseVoice Small cho upload transcription tiếng Anh

```python
lang = "en"
type = "upload"
model = "sensevoice-small"
```

## 📄 Format file

### ground_truth_full.txt
File chứa transcript chính xác (ground truth), mỗi dòng là một đoạn hoặc toàn bộ transcript.

**Ví dụ:**
```
これはテストです。
音声認識の精度を評価します。
```

### predict_full.txt
File chứa kết quả dự đoán từ model, format tương tự ground_truth_full.txt.

**Ví dụ:**
```
これはテストです。
音声認識の精度を評価します。
```

## 🔍 Giải thích kết quả

### WER và CER thấp (< 0.1)
- ✅ Chất lượng transcription rất tốt
- Hầu hết các từ/ky tự được nhận diện chính xác

### WER và CER trung bình (0.1 - 0.3)
- ⚠️ Chất lượng transcription ở mức chấp nhận được
- Có một số lỗi nhưng vẫn có thể sử dụng được

### WER và CER cao (> 0.3)
- ❌ Chất lượng transcription kém
- Nhiều lỗi, cần cải thiện model hoặc cấu hình

## 💡 Tips

1. **So sánh models**: Chạy script với các model khác nhau để so sánh hiệu suất
2. **So sánh realtime vs upload**: So sánh cùng một model nhưng khác loại (realtime vs upload)
3. **Đa ngôn ngữ**: Test trên nhiều ngôn ngữ để đánh giá khả năng đa ngôn ngữ
4. **Full transcript**: Sử dụng `predict_full.txt` thay vì `predict.txt` để đánh giá toàn bộ transcript (không chỉ segments)

## 🔗 Tham khảo

- [jiwer documentation](https://github.com/jitsi/jiwer)
- [WER explanation](https://en.wikipedia.org/wiki/Word_error_rate)
- [CER explanation](https://en.wikipedia.org/wiki/Character_error_rate)

