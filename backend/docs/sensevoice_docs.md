## 🧩 1. Tổng quan về `SenseVoice`

`SenseVoice` là mô hình **đa nhiệm (multi-task)** trong hệ sinh thái **FunAudioLLM**, được thiết kế để không chỉ nhận dạng tiếng nói (ASR) mà còn **hiểu giọng nói** — bao gồm:

* **ASR** – Automatic Speech Recognition (chuyển lời nói thành văn bản)
* **LID** – Language Identification (phát hiện ngôn ngữ)
* **SER** – Speech Emotion Recognition (nhận diện cảm xúc)
* **AED** – Audio Event Detection (phát hiện sự kiện âm thanh: vỗ tay, cười, ho, nhạc, v.v.)

Phiên bản phổ biến nhất là `SenseVoiceSmall`, có ưu điểm **độ trễ rất thấp (~70 ms cho 10 s audio)**, hỗ trợ hơn **50 ngôn ngữ**, hoạt động tốt cả trên **CPU** và **GPU**.

```python
from funasr import AutoModel

model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    vad_model="fsmn-vad",                    # Tích hợp VAD (phát hiện giọng nói)
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda"                          # hoặc "cpu"
)
```

---

## ⚙️ 2. Các tham số chính khi khởi tạo

| Tham số               | Mô tả chi tiết                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| **model**             | Tên hoặc đường dẫn model. <br>Ví dụ: `"FunAudioLLM/SenseVoiceSmall"` hoặc model cục bộ.                |
| **vad_model**         | Bật/tắt VAD (Voice Activity Detection). <br>`"fsmn-vad"` để bật hoặc `None` để tắt.                    |
| **vad_kwargs**        | Dict cấu hình cho VAD, ví dụ:<br>`{"max_single_segment_time": 30000}` (ms) – thời lượng tối đa 1 đoạn. |
| **device**            | `"cuda:0"` / `"cpu"`. Nên dùng GPU để đạt tốc độ realtime.                                             |
| **hub**               | (Tuỳ chọn) Nguồn tải model, `"hf"` (HuggingFace) hoặc `"ms"` (ModelScope).                             |
| **trust_remote_code** | `True` nếu muốn cho phép model tải code tuỳ chỉnh từ repo gốc.                                         |

---

## 🔍 3. Phương thức `generate()`

Phương thức chính của SenseVoice để xử lý audio.

```python
results = model.generate(
    input="sample.wav",          # Hoặc numpy.ndarray float32
    cache={},                    # Dùng khi streaming
    language="auto",             # "vi", "en", "ja", "auto"...
    use_itn=False,               # Không chuyển đổi số → chữ
    batch_size_s=10,             # Số giây audio mỗi batch
    merge_vad=True,              # Gom đoạn sau khi VAD
    merge_length_s=5,            # Giới hạn merge
)
```

### 📘 Giải thích các tham số quan trọng

| Nhóm                   | Tham số                              | Giải thích                                                                 |
| ---------------------- | ------------------------------------ | -------------------------------------------------------------------------- |
| **Input**              | `input`                              | Đường dẫn file audio hoặc mảng `numpy.float32` mono 16kHz.                 |
| **Ngôn ngữ & VAD**     | `language`, `vad_model`, `merge_vad` | `language="auto"` cho tự động, `merge_vad=True` để ghép các đoạn gần nhau. |
| **Batch & hiệu năng**  | `batch_size_s`, `merge_length_s`     | Điều chỉnh kích thước batch (giây). Realtime nên dùng 5–10 s.              |
| **Text normalization** | `use_itn`                            | `True`: đổi “123” → “một trăm hai mươi ba”, `False`: giữ nguyên số.        |
| **Streaming**          | `cache`                              | Duy trì ngữ cảnh khi xử lý liên tục qua các chunk audio.                   |
| **Output detail**      | `rich` (nội bộ)                      | Một số phiên bản có thể trả thêm `emotion`, `event`, `lang`.               |

---

## 🧠 4. Output trả về

Hàm `generate()` trả về **list dict**, mỗi phần tử mô tả một đoạn audio.

```python
[
  {
    "text": "Xin chào, tôi là trợ lý ảo.",
    "key": "rand_key_gUe52RvEJgwBu"
  },
  ...
]
```

Mỗi phần tử có:

* `text` — nội dung nhận dạng
* `key`  — key

---

## ⚡ 5. Các chế độ tối ưu hiệu năng

| Trường hợp                   | Cấu hình gợi ý                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------ |
| **Realtime, latency thấp**   | `batch_size_s=5`, `vad_model="fsmn-vad"`, `merge_vad=False`, `language="auto"` |
| **Chính xác cao (file dài)** | `batch_size_s=60`, `merge_vad=True`, `merge_length_s=10`, `language="vi"`      |
| **CPU-only**                 | `device="cpu"`, `batch_size_s=10`, `vad_model=None` để bỏ phân đoạn tự động.   |
| **GPU mạnh (RTX/V100)**      | `device="cuda:0"`, `batch_size_s=20`, `merge_vad=True`.                        |
| **Streaming WebSocket**      | Sử dụng `cache={}` và chunk audio 1–2 s để inference nối tiếp.                 |

---

## 🧩 6. So sánh nhanh các model SenseVoice

| Model             | Kích thước | Ngôn ngữ | RAM cần (FP16) | Latency (10 s audio) | Tác vụ hỗ trợ      |
| ----------------- | ---------- | -------- | -------------- | -------------------- | ------------------ |
| `SenseVoiceSmall` | ~150 MB    | > 50     | ~1 GB          | ~70 ms               | ASR, LID, SER, AED |
| `SenseVoiceBase`  | ~350 MB    | > 50     | ~2 GB          | ~120 ms              | ASR, LID, SER, AED |
| `SenseVoiceLarge` | ~800 MB    | > 50     | ~5 GB          | ~250 ms              | ASR, LID, SER, AED |

---

## 🧪 7. Ví dụ đầy đủ

```python
from funasr import AutoModel
import soundfile as sf

model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 15000},
    device="cuda:0"
)

audio, sr = sf.read("sample.wav")
assert sr == 16000

res = model.generate(
    input=audio,
    language="vi",
    use_itn=False,
    batch_size_s=10,
    merge_vad=True
)

for r in res:
    print(f"{r['text']}")
```

---

## ✅ 8. Khi dùng cho realtime transcript

| Mục tiêu              | Khuyến nghị                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| Tốc độ phản hồi nhanh | Dùng `SenseVoiceSmall`, bật `vad_model`, `batch_size_s=5`                |
| Giảm trễ              | Gửi chunk audio 1 s qua WebSocket và gọi `model.generate()` liên tục     |
| Tiếng Việt ổn định    | Ép `language="vi"` để tránh nhận sai                                     |
| Chống nhiễu mic       | Bật `fsmn-vad` và giảm `max_single_segment_time` xuống 10 s              |
| Kết hợp emotion/event | Bật xuất `emotion`, `event` để thêm cảm xúc hoặc nhạc nền vào transcript |

---