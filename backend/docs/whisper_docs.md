## 🧩 1. Tổng quan về `WhisperModel`

`WhisperModel` là lớp chính dùng để:

* **Tải model Whisper** (các biến thể: tiny, base, small, medium, large, large-v2/v3).
* **Chạy inference** với input là mảng âm thanh (`numpy`).
* **Trả về kết quả dạng segment (đoạn có thời gian + text)**.

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    model_size_or_path="medium",
    device="cuda",          # hoặc "cpu"
    compute_type="float16", # hoặc "int8" để giảm RAM
)
```

---

## ⚙️ 2. Các tham số chính khi khởi tạo

| Tham số                | Mô tả chi tiết                                                                                                              |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **model_size_or_path** | Tên model hoặc đường dẫn local. <br>Ví dụ: `"tiny"`, `"small"`, `"medium"`, `"large-v3"`, hoặc `"./models/whisper-medium"`. |
| **device**             | `"cpu"`, `"cuda"`, `"auto"`. <br>Nếu có GPU, nên dùng `"cuda"`.                                                             |
| **compute_type**       | Kiểu tính toán: <br>- `"float16"` → nhanh, GPU<br>- `"int8"` / `"int8_float16"` → tiết kiệm RAM, dùng CPU.                  |
| **cpu_threads**        | Số luồng CPU (mặc định = all). Giới hạn để tránh quá tải CPU.                                                               |
| **num_workers**        | Số tiến trình xử lý batch audio song song. Hữu ích nếu dùng nhiều GPU hoặc nhiều file.                                      |
| **download_root**      | Thư mục chứa model tải về. Mặc định `~/.cache/whisper`.                                                                     |

---

## 🔍 3. Phương thức `transcribe()`

Cấu trúc cơ bản:

```python
segments, info = model.transcribe(
    audio,                     # numpy array float32 (mono 16kHz)
    beam_size=5,               # decoding beam width
    language="vi",             # "auto" hoặc mã ISO (en, ja, vi, ...)
    temperature=0.0,           # 0.0–1.0, thấp -> ít lỗi ngẫu nhiên
    best_of=5,                 # lấy best-of-N candidate
    vad_filter=True,           # bật lọc im lặng (Voice Activity Detection)
    vad_parameters=dict(min_silence_duration_ms=200),
    condition_on_previous_text=True,  # giữ ngữ cảnh giữa các chunk
    initial_prompt=None,       # prompt gợi ý nội dung ban đầu
    word_timestamps=False,     # trả timestamp cho từng từ
    no_speech_threshold=0.6,   # ngưỡng im lặng
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    patience=1.0,              # beam search early stopping
    suppress_tokens=[-1],      # token bị loại bỏ
)
```

### 📘 Giải thích các tham số quan trọng

| Nhóm                             | Tham số                                                                    | Giải thích                                                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Ngôn ngữ & khởi tạo**          | `language`, `initial_prompt`                                               | Có thể ép model dùng tiếng cụ thể, giúp nhanh và ổn định hơn so với auto detect.                                            |
| **Chính xác vs tốc độ**          | `beam_size`, `best_of`, `temperature`                                      | - Tăng `beam_size` → chính xác hơn, nhưng chậm.<br>- `best_of=1`, `beam_size=1` → realtime hơn.                             |
| **Streaming / segment rời nhau** | `condition_on_previous_text`                                               | Nếu `False`, mỗi chunk được dịch độc lập (thích hợp realtime). Nếu `True`, giữ ngữ cảnh giữa các đoạn (thích hợp file dài). |
| **Xử lý im lặng**                | `vad_filter`, `vad_parameters`                                             | Loại bỏ phần im lặng trước khi decode, giúp giảm latency.                                                                   |
| **Cắt ngắn lỗi**                 | `no_speech_threshold`, `compression_ratio_threshold`, `log_prob_threshold` | Dùng để loại bỏ segment bị sai (noise).                                                                                     |
| **Thông tin đầu ra**             | `word_timestamps=True`                                                     | Trả thêm timestamp từng từ (phù hợp làm karaoke hoặc highlight UI).                                                         |

---

## 🧠 4. Output trả về

### Dạng 1 — Generator segments

```python
for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
```

Mỗi `segment` có:

* `.start`, `.end` — thời gian (giây)
* `.text` — chuỗi đã nhận dạng
* `.words` — nếu bật `word_timestamps`

### Dạng 2 — Thông tin chung

`info` chứa:

* `language`: mã ngôn ngữ phát hiện
* `language_probability`: xác suất
* `duration`: độ dài audio
* `transcription_time`: thời gian xử lý

---

## ⚡ 5. Các chế độ tối ưu hiệu năng

| Trường hợp                   | Cấu hình gợi ý                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| **Realtime, latency thấp**   | `beam_size=1`, `best_of=1`, `temperature=0.0`, `condition_on_previous_text=False`    |
| **Chính xác cao (file dài)** | `beam_size=5`, `best_of=5`, `temperature=0.2–0.5`, `condition_on_previous_text=True` |
| **Server CPU yếu**           | `compute_type="int8"`, `vad_filter=True`, `model="small"`                            |
| **GPU inference**            | `device="cuda"`, `compute_type="float16"`, `model="medium"` hoặc `large-v3`          |

---

## 🧩 6. So sánh nhanh các model

| Model    | Kích thước | Ngôn ngữ | RAM cần (FP16) | Tốc độ (real-time) | Độ chính xác |
| -------- | ---------- | -------- | -------------- | ------------------ | ------------ |
| tiny     | 39 MB      | ~50      | <1 GB          | 8× nhanh           | thấp         |
| base     | 74 MB      | ~50      | 1 GB           | 5× nhanh           | trung bình   |
| small    | 244 MB     | ~50      | 2 GB           | 2× nhanh           | tốt          |
| medium   | 769 MB     | ~50      | 5 GB           | 1×                 | cao          |
| large-v3 | 1.5 GB     | ~100     | 10 GB          | 0.5×               | rất cao      |

---

## 🧪 7. Ví dụ đầy đủ

```python
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

model = WhisperModel("small", device="cuda", compute_type="float16")

audio, sr = sf.read("sample.wav")
assert sr == 16000

segments, info = model.transcribe(
    audio,
    language="vi",
    beam_size=1,
    vad_filter=True,
    condition_on_previous_text=False
)

for seg in segments:
    print(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
```

---

## ✅ 8. Khi dùng cho realtime transcript

| Mục tiêu                          | Khuyến nghị                                                        |
| --------------------------------- | ------------------------------------------------------------------ |
| Tốc độ phản hồi nhanh             | `beam_size=1`, `best_of=1`, `vad_filter=True`                      |
| Giảm drift văn bản giữa các chunk | `condition_on_previous_text=False`                                 |
| Tiếng Việt ổn định                | `language="vi"`                                                    |
| Nhiễu mic                         | Bật `vad_filter`, `min_silence_duration_ms=150`                    |
| Âm thanh dài (offline)            | Chia thành các block 30–60s, giữ `condition_on_previous_text=True` |

---
