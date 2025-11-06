# 🎙️ Realtime Transcript

Ứng dụng web chuyển đổi giọng nói thành văn bản (Speech-to-Text) theo thời gian thực và từ file audio/video, sử dụng OpenAI Whisper model qua `faster-whisper`.

## ✨ Tính năng

- **Realtime Transcription**: Chuyển đổi giọng nói thành văn bản theo thời gian thực qua WebSocket
- **File Upload**: Upload và transcribe file audio/video (mp3, wav, m4a, mp4, avi, mov, ...)
- **Đa ngôn ngữ**: Hỗ trợ nhiều ngôn ngữ với tự động phát hiện ngôn ngữ
- **Timestamps**: Hiển thị timestamp cho từng đoạn transcript
- **Drag & Drop**: Kéo thả file để upload dễ dàng
- **Progress Tracking**: Theo dõi tiến trình upload và xử lý

## 🏗️ Kiến trúc

- **Backend**: FastAPI với WebSocket cho realtime transcription
- **Frontend**: HTML/JavaScript với WebSocket client
- **Model**: OpenAI Whisper (qua `faster-whisper`)

## 📋 Yêu cầu hệ thống

### Phần mềm
- Python 3.8+
- ffmpeg (để xử lý video files)

### Phần cứng
- **CPU**: Máy tính có CPU đủ mạnh (khuyến nghị: 4+ cores)
- **GPU**: Tùy chọn, nhưng khuyến nghị nếu muốn xử lý nhanh hơn (CUDA compatible)

### Cài đặt ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Tải từ [ffmpeg.org](https://ffmpeg.org/download.html) và thêm vào PATH

**Conda:**
```bash
conda install -c conda-forge ffmpeg
```

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/trunghoang2002/realtime-transcript.git
cd realtime-transcript
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies
```bash
cd backend
pip install -r requirements.txt
```

**Lưu ý**: Nếu có GPU và muốn sử dụng CUDA:
- Đảm bảo đã cài đặt CUDA toolkit
- Sử dụng `onnxruntime-gpu` (đã có trong requirements.txt)

### 4. Cấu hình model (tùy chọn)

Chỉnh sửa `backend/main.py` để thay đổi model và device:

```python
MODEL_NAME = "small"   # "small" (nhanh), "medium" (chính xác), "large-v3" (nặng)
DEVICE = "cpu"         # "cuda" nếu có GPU, "cpu" nếu không
COMPUTE_TYPE = "int8"  # "float16" trên GPU, "int8" hoặc "int8_float16" trên CPU
```

**Khuyến nghị:**
- **CPU**: `MODEL_NAME = "small"`, `DEVICE = "cpu"`, `COMPUTE_TYPE = "int8"`
- **GPU**: `MODEL_NAME = "medium"`, `DEVICE = "cuda"`, `COMPUTE_TYPE = "float16"`

## ▶️ Chạy ứng dụng

### Khởi động server
```bash
cd backend
python main.py
```

Server sẽ chạy tại: `http://localhost:8917`

### Truy cập ứng dụng
Mở trình duyệt và truy cập: `http://localhost:8917`

## 📡 API Endpoints

### WebSocket: `/ws`

Kết nối WebSocket để realtime transcription.

**Protocol:**
1. Client gửi message bắt đầu:
```json
{
  "event": "start",
  "sample_rate": 16000,
  "format": "pcm16",
  "language": "vi"  // hoặc "auto", "en", "ja", ...
}
```

2. Client gửi audio chunks dưới dạng binary (PCM16 mono 16kHz)

3. Server trả về:
```json
{"type": "ready"}  // Khi sẵn sàng
{"type": "partial", "text": "..."}  // Transcript từng đoạn
{"type": "final", "text": ""}  // Khi stop
{"type": "error", "message": "..."}  // Nếu có lỗi
```

4. Client gửi để dừng:
```json
{"event": "stop"}
```

### REST API: `POST /api/transcribe`

Upload file audio/video để transcribe.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Form fields:
  - `file`: File audio/video
  - `language`: (optional) Ngôn ngữ ("vi", "en", "auto", ...)

**Response:**
```json
{
  "success": true,
  "filename": "audio.mp3",
  "text": "Full transcript text...",
  "full_text": "Full transcript with line breaks...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Đoạn text đầu tiên"
    },
    ...
  ],
  "language": "vi",
  "language_probability": 0.95
}
```

**Example với curl:**
```bash
curl -X POST http://localhost:8917/api/transcribe \
  -F "file=@audio.mp3" \
  -F "language=vi"
```

## 📁 Cấu trúc thư mục

```
realtime-transcript/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Python dependencies
│   └── __pycache__/         # Python cache
├── frontend/
│   └── index.html           # Frontend UI
└── README.md                # Tài liệu này
```

## 🎯 Cách sử dụng

### Realtime Transcription

1. Mở tab **"Realtime"**
2. Chọn ngôn ngữ (hoặc "Tự động")
3. Nhấn **"Start"** và cho phép truy cập microphone
4. Bắt đầu nói, transcript sẽ hiển thị theo thời gian thực
5. Nhấn **"Stop"** để dừng

### Upload File

1. Mở tab **"Upload File"**
2. Chọn ngôn ngữ (hoặc "Tự động")
3. Kéo thả file vào vùng upload hoặc click **"Chọn File"**
4. Chờ xử lý (progress bar sẽ hiển thị)
5. Xem kết quả transcript với timestamps

## ⚙️ Cấu hình nâng cao

### Thay đổi port

Sửa trong `backend/main.py`:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8917, reload=True)
```

### Thay đổi WebSocket URL

Sửa trong `frontend/index.html`:
```html
<input id="wsUrl" type="text" value="ws://localhost:8917/ws">
```

### Tối ưu hóa cho realtime

Trong `run_transcribe_on_buffer()`:
- `beam_size=1`: Tối ưu tốc độ
- `best_of=1`: Tối ưu tốc độ
- `condition_on_previous_text=False`: Giảm độ trễ

### Tối ưu hóa cho file upload

Trong `transcribe_file()`:
- `beam_size=5`: Tăng độ chính xác
- `best_of=5`: Tăng độ chính xác
- `condition_on_previous_text=True`: Sử dụng ngữ cảnh

## 🐛 Troubleshooting

### Lỗi: "max() arg is an empty sequence"
- **Nguyên nhân**: Audio quá ngắn hoặc hoàn toàn im lặng
- **Giải pháp**: Đã được xử lý tự động, chỉ cần thử lại

### Lỗi: "ffmpeg not found"
- **Nguyên nhân**: Chưa cài đặt ffmpeg
- **Giải pháp**: Cài đặt ffmpeg theo hướng dẫn ở trên

### Realtime transcription chậm
- **Giải pháp**: 
  - Giảm `MODEL_NAME` xuống "small"
  - Sử dụng GPU nếu có
  - Tăng `CHUNK_TARGET_BYTES` để giảm số lần transcribe

### File upload không hoạt động
- **Kiểm tra**: Đảm bảo file không quá lớn (giới hạn bộ nhớ)
- **Giải pháp**: Sử dụng file nhỏ hơn hoặc tăng bộ nhớ

### WebSocket connection failed
- **Kiểm tra**: Đảm bảo server đang chạy
- **Kiểm tra**: URL WebSocket đúng (ws://localhost:8917/ws)

## 📝 Ghi chú

- Model Whisper được tải tự động lần đầu chạy
- File tạm sẽ tự động xóa sau khi xử lý xong
- Realtime transcription sử dụng buffer ~1 giây để giảm độ trễ
- Với video files, audio sẽ được extract tự động nếu có ffmpeg

## 🔧 Dependencies

Xem `backend/requirements.txt` để biết danh sách đầy đủ.

**Chính:**
- `fastapi`: Web framework
- `faster-whisper`: Whisper model implementation
- `uvicorn`: ASGI server
- `numpy`: Xử lý audio

## 📄 License

Dự án này sử dụng các thư viện mã nguồn mở. Vui lòng xem license của từng thư viện.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📧 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên repository.

