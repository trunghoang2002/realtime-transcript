# 🎙️ Realtime Transcript

Ứng dụng web chuyển đổi giọng nói thành văn bản (Speech-to-Text) theo thời gian thực và từ file audio/video.

- Hỗ trợ 2 backend model:
  - Whisper (qua `faster-whisper`) — mặc định, tối ưu realtime
  - SenseVoice (qua `funasr`) — thay thế, có fallback timestamp cho file upload

## ✨ Tính năng

- **Realtime Transcription**: Chuyển đổi giọng nói thành văn bản theo thời gian thực qua WebSocket (stream Full Transcript + Segments có timestamp)
- **File Upload**: Upload và transcribe file audio/video (mp3, wav, m4a, mp4, avi, mov, ...), trả về full transcript và danh sách segments có timestamp
- **Speaker Detection (Nhận diện người nói)**: Tùy chọn nhận diện và phân biệt nhiều người nói trong audio, hỗ trợ cấu hình số lượng người nói tối đa (mặc định: 2)
- **Đa ngôn ngữ**: Hỗ trợ nhiều ngôn ngữ với tự động phát hiện ngôn ngữ
- **Timestamps**: Realtime có timestamp theo buffer; Upload có timestamp từ model hoặc được suy đoán (fallback) theo độ dài nội dung
- **RTF (Real-Time Factor)**: Tính toán và hiển thị RTF cho file upload để đánh giá hiệu suất xử lý
- **Auto-detect WebSocket URL**: Tự động phát hiện và cấu hình WebSocket URL từ port của backend
- **Error Handling**: Hệ thống thông báo lỗi/thành công với tự động dừng khi có lỗi
- **UI Protection**: Tự động disable các tùy chọn cấu hình khi đang xử lý để tránh thay đổi không mong muốn
- **Drag & Drop**: Kéo thả file để upload dễ dàng
- **Progress Tracking**: Theo dõi tiến trình upload và xử lý

## 🏗️ Kiến trúc

- **Backend**: FastAPI (WebSocket cho realtime, REST cho upload)
- **Frontend**: HTML/JavaScript với WebSocket client
- **Model**: Whisper (`faster-whisper`) hoặc SenseVoice (`funasr`)

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

**GPU (CUDA)**
- Khuyến nghị CUDA 12.1 + cuDNN 9
- Dự án có sẵn script để set môi trường CUDA/cuDNN và chạy server:
  - Whisper: `backend/run_main_with_cuda.sh`
  - SenseVoice: `backend/run_main_sensevoice_with_cuda.sh`

### 4. Cấu hình model (tùy chọn)

Chỉnh sửa `backend/main.py` (Whisper) hoặc `backend/main_sensevoice.py` (SenseVoice) để thay đổi model và device:

```python
MODEL_NAME = "small"   # "small" (nhanh), "medium" (chính xác), "large-v3" (nặng)
DEVICE = "cpu"         # "cuda" nếu có GPU, "cpu" nếu không
COMPUTE_TYPE = "int8"  # "float16" trên GPU, "int8" hoặc "int8_float16" trên CPU
```

**Khuyến nghị:**
- Whisper + CPU: `MODEL_NAME = "small"`, `DEVICE = "cpu"`, `COMPUTE_TYPE = "int8"`
- Whisper + GPU: `MODEL_NAME = "medium"`, `DEVICE = "cuda"`, `COMPUTE_TYPE = "float16"`
- SenseVoice + GPU: `DEVICE = "cuda"`
- SenseVoice + CPU: `DEVICE = "cpu"`

## ▶️ Chạy ứng dụng

### Khởi động server
```bash
cd backend
python main.py
```

Server sẽ chạy tại: `http://localhost:8917`

### Chạy với CUDA (nếu có GPU)

**Yêu cầu:**
- CUDA 12.1 đã được cài đặt
- File `~/activate_cuda121.sh` để activate CUDA 12.1
- Conda environment `v2t` đã được tạo và có các dependencies

**Chạy server:**
```bash
cd backend
./run_main_with_cuda.sh          # Whisper (port 8917)
# hoặc
./run_main_sensevoice_with_cuda.sh  # SenseVoice (port 8918)
```

**Với auto-reload (development):**
```bash
./run_main_with_cuda.sh --reload          # Whisper
./run_main_sensevoice_with_cuda.sh --reload  # SenseVoice
```

**Lưu ý:**
- Scripts tự động:
  - Activate conda environment `v2t`
  - Setup CUDA paths và cuDNN libraries từ conda env
  - Cấu hình `LD_LIBRARY_PATH` cho cuDNN và PyTorch libraries
  - Sử dụng GPU device 1 (`CUDA_VISIBLE_DEVICES=1`)
- Whisper chạy trên port **8917**, SenseVoice chạy trên port **8918**
- Nếu gặp lỗi, kiểm tra:
  - Conda env `v2t` đã được tạo chưa
  - File `~/activate_cuda121.sh` có tồn tại không
  - CUDA 12.1 đã được cài đặt đúng chưa

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
  "language": "vi",  // hoặc "auto", "en", "ja", ...
  "detect_speaker": false,  // (optional) Bật/tắt nhận diện người nói
  "max_speakers": 2  // (optional) Số lượng người nói tối đa (chỉ khi detect_speaker=true)
}
```

2. Client gửi audio chunks dưới dạng binary (PCM16 mono 16kHz)

3. Server trả về:
```json
{"type": "ready"}
{"type": "partial", "text": "...", "speaker_id": "spk_01", "language": "vi", "language_probability": 0.95, "segments": [{"start": 0.0, "end": 1.0, "text": "...", "speaker_id": "spk_01"}]}
{"type": "final", "text": ""}
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
  - `file`: File audio/video (required)
  - `language`: (optional) Ngôn ngữ ("vi", "en", "auto", ...)
  - `detect_speaker`: (optional) "true" hoặc "false" - Bật/tắt nhận diện người nói (mặc định: "false")
  - `max_speakers`: (optional) Số lượng người nói tối đa (chỉ khi detect_speaker="true", mặc định: 2)

**Response:**
```json
{
  "success": true,
  "filename": "audio.mp3",
  "text": "Full transcript text...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Đoạn text đầu tiên",
      "speaker_id": "spk_01"  // Chỉ có khi detect_speaker=true
    },
    ...
  ],
  "language": "vi",
  "language_probability": 0.95,
  "rtf": 0.234  // Real-Time Factor: processing_time / audio_duration
}
```

**Example với curl:**
```bash
# Upload cơ bản
curl -X POST http://localhost:8917/api/transcribe \
  -F "file=@audio.mp3" \
  -F "language=vi"

# Upload với speaker detection
curl -X POST http://localhost:8917/api/transcribe \
  -F "file=@audio.mp3" \
  -F "language=vi" \
  -F "detect_speaker=true" \
  -F "max_speakers=3"
```

## 📁 Cấu trúc thư mục

```
realtime-transcript/
├── backend/
│   ├── main.py                          # FastAPI server với Whisper model
│   ├── main_sensevoice.py               # FastAPI server với SenseVoice model
│   ├── requirements.txt                 # Python dependencies
│   ├── constraints.txt                  # Version constraints cho dependencies
│   │
│   ├── run_main_with_cuda.sh            # Script chạy Whisper với CUDA support
│   ├── run_main_sensevoice_with_cuda.sh # Script chạy SenseVoice với CUDA support
│   ├── activate_cuda_env.sh             # Script activate CUDA environment
│   │
│   ├── get_audio.py                     # Utilities để decode audio/video files
│   ├── silero_vad.py                    # VAD (Voice Activity Detection) sử dụng Silero VAD
│   ├── fix_speechbrain.py               # Patch compatibility cho SpeechBrain với huggingface_hub
│   │
│   ├── check_cuda.py                    # Script kiểm tra CUDA availability
│   │
│   ├── whisper_docs.md                  # Tài liệu về Whisper model
│   ├── sensevoice_docs.md               # Tài liệu về SenseVoice model
│   ├── whisper_vs_sensevoice.md         # So sánh Whisper vs SenseVoice
│   │
│   ├── eval/                            # Thư mục evaluation/testing
│   │   ├── eval.py                      # Script đánh giá model
│   │   ├── en/                          # Test cases tiếng Anh
│   │   └── ja/                          # Test cases tiếng Nhật
│   │
│   ├── note.txt                         # Ghi chú phát triển
│   └── __pycache__/                     # Python cache (tự động tạo)
│
├── frontend/
│   └── index.html                       # Frontend UI (HTML + JavaScript)
│
└── README.md                            # Tài liệu này
```

### Mô tả các thành phần chính

#### Backend Core Files
- **`main.py`**: Server chính sử dụng Whisper model (`faster-whisper`) cho realtime và file transcription
- **`main_sensevoice.py`**: Server thay thế sử dụng SenseVoice model (`funasr`) với khả năng fallback timestamp tốt hơn

#### Utility Modules
- **`get_audio.py`**: Xử lý decode audio/video files thành numpy array (16kHz mono) sử dụng `av` (PyAV)
- **`silero_vad.py`**: Voice Activity Detection để phát hiện các đoạn có giọng nói, loại bỏ im lặng
- **`fix_speechbrain.py`**: Patch để fix compatibility issue giữa SpeechBrain và huggingface_hub (cần cho speaker detection)

#### CUDA Scripts
- **`run_main_with_cuda.sh`**: Script tự động setup CUDA/cuDNN và chạy Whisper server trên port 8917
- **`run_main_sensevoice_with_cuda.sh`**: Script tự động setup CUDA/cuDNN và chạy SenseVoice server trên port 8918
- **`activate_cuda_env.sh`**: Script helper để activate CUDA environment
- **`check_cuda.py`**: Script kiểm tra CUDA có sẵn và hoạt động không

#### Documentation
- **`whisper_docs.md`**: Tài liệu chi tiết về cách sử dụng Whisper model
- **`sensevoice_docs.md`**: Tài liệu chi tiết về cách sử dụng SenseVoice model
- **`whisper_vs_sensevoice.md`**: So sánh ưu/nhược điểm giữa 2 models

#### Evaluation
- **`eval/`**: Thư mục chứa scripts và test cases để đánh giá chất lượng transcription
  - `eval.py`: Script chạy evaluation
  - `en/`, `ja/`: Test cases cho các ngôn ngữ khác nhau

## 🎯 Cách sử dụng

### Realtime Transcription

1. Mở tab **"Realtime"**
2. Cấu hình:
   - **WebSocket URL**: Tự động detect từ backend (có thể chỉnh sửa nếu cần)
   - **Ngôn ngữ**: Chọn ngôn ngữ (hoặc "Tự động")
   - **Detect speaker**: Bật/tắt nhận diện người nói (tùy chọn)
   - **Max speaker**: Số lượng người nói tối đa (chỉ hiện khi bật Detect speaker, mặc định: 2)
3. Nhấn **"Start"** và cho phép truy cập microphone
   - Các tùy chọn cấu hình sẽ tự động bị disable khi đang xử lý
4. Bắt đầu nói, transcript sẽ hiển thị theo thời gian thực theo 2 phần:
   - **Full Transcript**: Nối liên tục nội dung (có speaker ID nếu bật detect speaker)
   - **Segments**: Danh sách các đoạn có timestamp (start → end) và speaker ID (nếu có)
5. Nhấn **"Stop"** để dừng
   - Các tùy chọn cấu hình sẽ được enable lại

**Lưu ý**: Nếu có lỗi xảy ra, hệ thống sẽ tự động dừng và hiển thị thông báo lỗi.

### Upload File

1. Mở tab **"Upload File"**
2. Cấu hình:
   - **Ngôn ngữ**: Chọn ngôn ngữ (hoặc "Tự động")
   - **Detect speaker**: Bật/tắt nhận diện người nói (tùy chọn)
   - **Max speaker**: Số lượng người nói tối đa (chỉ hiện khi bật Detect speaker, mặc định: 2)
3. Kéo thả file vào vùng upload hoặc click **"Chọn File"**
   - Các tùy chọn cấu hình sẽ tự động bị disable khi đang xử lý
4. Chờ xử lý:
   - Progress bar sẽ hiển thị tiến trình upload
   - Thông báo thành công/lỗi sẽ xuất hiện ở góc trên bên phải
5. Xem kết quả:
   - **Full Transcript**: Toàn bộ nội dung (có speaker ID nếu bật detect speaker)
   - **Segments**: Danh sách các đoạn có timestamp và speaker ID (nếu có)
   - **RTF**: Real-Time Factor (hiệu suất xử lý) - RTF < 1.0 nghĩa là xử lý nhanh hơn thời gian thực

## ⚙️ Cấu hình nâng cao

### Thay đổi port

Sửa trong `backend/main.py`:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8917, reload=True)
```

### Thay đổi WebSocket URL

WebSocket URL tự động detect từ port của backend. Nếu cần chỉnh sửa thủ công, sửa trong `frontend/index.html`:
```html
<input id="wsUrl" type="text" value="ws://localhost:8917/ws">
```

Hoặc chỉnh sửa function `getWebSocketUrl()` trong JavaScript để thay đổi logic auto-detect.

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

### CUDA/cuDNN không tìm thấy (libcudnn_ops.so.*)
- Dùng script:
  - Whisper: `backend/run_main_with_cuda.sh`
  - SenseVoice: `backend/run_main_sensevoice_with_cuda.sh`
- Hoặc tự set:
  ```bash
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
  ```

## 📝 Ghi chú

- Model Whisper/SenseVoice được tải tự động lần đầu chạy
- File tạm sẽ tự động xóa sau khi xử lý xong
- Realtime transcription sử dụng buffer ~1 giây để giảm độ trễ
- Với video files, audio sẽ được extract tự động nếu có ffmpeg
- **Speaker Detection**: Sử dụng SpeechBrain ECAPA-TDNN model để nhận diện người nói
  - Speaker ID được gán dạng `spk_01`, `spk_02`, ...
  - Hệ thống tự động học và cập nhật embedding của từng speaker
  - Khi vượt quá `max_speakers`, hệ thống sẽ gán audio mới cho speaker gần nhất
- **RTF (Real-Time Factor)**: 
  - RTF = processing_time / audio_duration
  - RTF < 1.0: Xử lý nhanh hơn thời gian thực (tốt)
  - RTF = 1.0: Xử lý bằng thời gian thực
  - RTF > 1.0: Xử lý chậm hơn thời gian thực
- Các tùy chọn cấu hình tự động bị disable khi đang xử lý để tránh thay đổi không mong muốn
- Hệ thống tự động hiển thị/ẩn các phần tử UI dựa trên trạng thái (chỉ hiện transcript khi đang ghi)

## 🔧 Dependencies

Xem `backend/requirements.txt` để biết danh sách đầy đủ.

**Chính:**
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `numpy`: Xử lý audio
- `faster-whisper`: Whisper backend
- `funasr`: SenseVoice backend

## 📄 License

Dự án này sử dụng các thư viện mã nguồn mở. Vui lòng xem license của từng thư viện.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📧 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên repository.

