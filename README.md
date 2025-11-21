# 🎙️ Realtime Transcript

Ứng dụng web chuyển đổi giọng nói thành văn bản (Speech-to-Text) theo thời gian thực và từ file audio/video.

- Hỗ trợ nhiều backend model:
  - **Whisper** (qua `faster-whisper`) — mặc định, tối ưu realtime, mã nguồn mở
  - **SenseVoice** (qua `funasr`) — thay thế Whisper, có fallback timestamp
  - **Gemini** (qua Gemini API) — Google Gemini 2.5 Flash/Pro, chính xác cao
  - **Qwen Audio** (qua Modal/vLLM) — Alibaba Qwen2-Audio-7B, hỗ trợ nhiều ngôn ngữ
  - **Qwen Omni** (qua API) — Qwen3-Omni-30B, multimodal audio understanding
  - **Hybrid** - Kết hợp Whisper (realtime) + Qwen Omni (full transcription)

## ✨ Tính năng

- **Realtime Transcription**: Chuyển đổi giọng nói thành văn bản theo thời gian thực qua WebSocket với 2 loại messages:
  - **Partial messages**: Transcript tạm thời từ buffer ~1s (độ trễ thấp)
  - **Full messages**: Transcript chính xác hơn từ toàn bộ đoạn speech khi phát hiện kết thúc (độ chính xác cao)
- **Smart Buffer Management**: Tự động tích lũy audio và transcribe lại với cấu hình tốt hơn khi phát hiện kết thúc đoạn speech
- **File Upload**: Upload và transcribe file audio/video (mp3, wav, m4a, mp4, avi, mov, ...), trả về full transcript và danh sách segments có timestamp
- **Speaker Detection (Nhận diện người nói)**: Tùy chọn nhận diện và phân biệt nhiều người nói trong audio
  - **Whisper-based:**
    - **main_v3.py** (khuyến nghị ⭐): 2-tier matching (EMA + cluster centroid), persistent memory
    - **main_v2.py**: Bootstrap clustering với K-means, tự động phân cụm
    - **main.py**: Chỉ hỗ trợ trong file upload
  - **API-based:** 3-tier matching (EMA → centroid → verification model) cho độ chính xác cao
  - Hỗ trợ cấu hình số lượng người nói tối đa (mặc định: 2)
  - Speaker ID format: `spk_01`, `spk_02` (Whisper) hoặc `SPEAKER_00`, `SPEAKER_01` (v3)
- **Đa ngôn ngữ**: Hỗ trợ nhiều ngôn ngữ với tự động phát hiện ngôn ngữ
- **Timestamps**: Realtime có timestamp theo buffer; Upload có timestamp từ model hoặc được suy đoán (fallback) theo độ dài nội dung
- **RTF (Real-Time Factor)**: Tính toán và hiển thị RTF cho file upload để đánh giá hiệu suất xử lý
- **Auto-detect WebSocket URL**: Tự động phát hiện và cấu hình WebSocket URL từ port của backend
- **Error Handling**: Hệ thống thông báo lỗi/thành công với tự động dừng khi có lỗi
- **UI Protection**: Tự động disable các tùy chọn cấu hình khi đang xử lý để tránh thay đổi không mong muốn
- **Drag & Drop**: Kéo thả file để upload dễ dàng
- **Progress Tracking**: Theo dõi tiến trình upload và xử lý

## 📊 So sánh các Versions

### Whisper-based Versions (Mã nguồn mở, tối ưu realtime)

| Feature | main.py | main_v2.py | main_v3.py | main_v4.py | main_v5.py ⭐ | main_sensevoice.py |
|---------|---------|------------|------------|------------|---------------|-------------------|
| **Model** | Whisper | Whisper | Whisper | Whisper | Whisper | SenseVoice |
| **Realtime Transcription** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **File Upload** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Speaker Detection (Realtime)** | ❌ | ✅ Bootstrap | ✅ 2-tier | ✅ 2-tier | ✅ Fusion | ❌ |
| **Speaker Detection (File)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Embedding System** | - | SpeechBrain | SpeechBrain | Pyannote | **Fusion** | - |
| **Độ phức tạp** | Đơn giản | Phức tạp | Trung bình | Trung bình | Cao | Đơn giản |
| **Hiệu suất** | Tốt | Tốt | Tốt nhất | Rất tốt | **Tốt nhất** | Tốt |
| **Bootstrap Phase** | - | ~1 phút | Không cần | Không cần | Không cần | - |
| **Speaker Tracking** | File only | Persistent | Persistent | Persistent | Persistent | File only |
| **Chi phí** | Miễn phí | Miễn phí | Miễn phí | Miễn phí | Miễn phí | Miễn phí |
| **Khuyến nghị** | Testing | Tự động phân cụm | SpeechBrain | Pyannote | **Fusion - Tốt nhất** | Thay thế Whisper |

### API-based Versions (Chính xác cao, yêu cầu API key/server)

| Feature | main_gemini.py | main_qwenaudio.py | main_qwenomni.py | main_whispersmall_qwenomni.py |
|---------|----------------|-------------------|------------------|-------------------------------|
| **Model** | Gemini 2.5 | Qwen2-Audio-7B | Qwen3-Omni-30B | Whisper + Qwen Omni |
| **Provider** | Google API | Modal/vLLM | Custom API | Local + API |
| **Realtime Transcription** | ✅ Whisper | ✅ Whisper | ✅ Whisper | ✅ Whisper |
| **Full Transcription** | ✅ Gemini | ✅ Qwen Audio | ✅ Qwen Omni | ✅ Qwen Omni |
| **Speaker Detection** | ✅ | ✅ | ✅ | ✅ |
| **Độ chính xác** | Rất cao | Cao | Rất cao | Rất cao |
| **Latency** | Cao | Trung bình | Trung bình | Trung bình |
| **Chi phí** | Có phí | Có phí | Có phí | Có phí |
| **Đa ngôn ngữ** | ✅✅✅ | ✅✅ | ✅✅✅ | ✅✅✅ |
| **Khuyến nghị** | Chính xác nhất | Self-hosted | API endpoint | Hybrid tốt |

**Chọn version nào?**

**Mã nguồn mở (Miễn phí):**
- 🏁 **Mới bắt đầu**: `main.py` - Đơn giản, dễ hiểu
- ⭐ **Tốt nhất**: `main_v5.py` - Fusion diarization (Pyannote + SpeechBrain), chính xác cao nhất
- 🎯 **SpeechBrain**: `main_v3.py` - SpeechBrain embeddings, nhanh và ổn định
- 🎨 **Pyannote**: `main_v4.py` - Pyannote embeddings, chất lượng cao
- 🔬 **Nghiên cứu**: `main_v2.py` - Bootstrap clustering, tự động phân cụm
- 🔄 **Thay thế**: `main_sensevoice.py` - Model SenseVoice

**API-based (Chính xác cao, có phí):**
- 🌟 **Chính xác nhất**: `main_gemini.py` - Google Gemini, hỗ trợ đa ngôn ngữ tốt nhất
- 🏢 **Self-hosted**: `main_qwenaudio.py` - Deploy trên Modal/vLLM, kiểm soát data
- 🔌 **API endpoint**: `main_qwenomni.py` - Qwen3-Omni qua API
- ⚡ **Hybrid**: `main_whispersmall_qwenomni.py` - Kết hợp tốc độ + chính xác

### 💰 So sánh Chi phí & Performance

| Version | Chi phí | Latency | Chính xác | GPU Required | Use Case |
|---------|---------|---------|-----------|--------------|----------|
| **main_v5.py** | Miễn phí | Thấp (~150ms) | Rất cao | Optional | **Production miễn phí tốt nhất ⭐** |
| **main_v3.py** | Miễn phí | Thấp (~100ms) | Tốt | Optional | Production SpeechBrain |
| **main_v4.py** | Miễn phí | Thấp (~120ms) | Cao | Optional | Production Pyannote |
| **main_gemini.py** | ~$0.01/min | Cao (~1-2s) | Rất cao | No | Chất lượng cao nhất (API) ⭐ |
| **main_qwenaudio.py** | ~$0.5-1/hour | Trung bình (~500ms) | Cao | Modal GPU | Self-hosted |
| **main_whispersmall_qwenomni.py** | ~$0.005/min | Thấp+Cao | Rất cao | Optional | Hybrid tối ưu ⭐ |

**Lưu ý về chi phí:**
- Whisper versions: Hoàn toàn miễn phí, chạy local
- **main_v5.py**: Miễn phí, kết hợp tốt nhất của Pyannote và SpeechBrain
- Gemini: Free tier 15 requests/phút, sau đó có phí
- Qwen Audio Modal: Tính theo GPU hours (~$0.5-1/hour trên L4/H100)
- Hybrid: Chi phí thấp hơn vì chỉ call API cho full transcription

**Performance Tips:**
- **Tốt nhất (Miễn phí)**: Dùng `main_v5.py` - Fusion diarization với score-level fusion
- **Độ trễ thấp**: Dùng `main_v3.py` - SpeechBrain nhanh nhất
- **Chính xác cao (Miễn phí)**: Dùng `main_v4.py` hoặc `main_v5.py`
- **Chính xác cao (API)**: Dùng Gemini hoặc Hybrid versions
- **Data privacy**: Dùng Whisper local versions (v3, v4, v5)

## 🏗️ Kiến trúc

- **Backend**: FastAPI (WebSocket cho realtime, REST cho upload)
- **Frontend**: HTML/JavaScript với WebSocket client
- **Models**: 
  - **Mã nguồn mở**: Whisper (`faster-whisper`), SenseVoice (`funasr`)
  - **API-based**: Google Gemini, Qwen Audio (Modal/vLLM), Qwen Omni
  - **Hybrid**: Whisper (realtime) + API (full transcription)
- **Speaker Diarization**: 
  - **Fusion**: Kết hợp Pyannote + SpeechBrain với nhiều fusion methods (main_v5.py ⭐)
  - **SpeechBrain**: ECAPA-TDNN với 2-tier matching (EMA + cluster centroid) (main_v3.py)
  - **Pyannote**: pyannote.audio với 2-tier matching (main_v4.py)
  - Bootstrap clustering với K-means (main_v2.py)
  - 3-tier matching với verification model (API versions)
- **Infrastructure**:
  - Local: CUDA 12.1 + cuDNN 9 cho GPU inference
  - Cloud: Modal platform cho Qwen Audio deployment
  - API: Google Gemini API, Custom endpoints cho Qwen Omni

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
pip install -r constraints.txt
pip install -r requirements.txt
```

**GPU (CUDA)**
- Khuyến nghị CUDA 12.1 + cuDNN 9
- Dự án có sẵn script để set môi trường CUDA/cuDNN và chạy server:
  - `backend/scripts/run_main_with_cuda.sh` - Whisper (main.py)
  - `backend/scripts/run_main_v2_with_cuda.sh` - Whisper với bootstrap speaker detection (main_v2.py)
  - `backend/scripts/run_main_v3_with_cuda.sh` - Whisper với SpeechBrain diarization (main_v3.py)
  - `backend/scripts/run_main_v4_with_cuda.sh` - Whisper với Pyannote diarization (main_v4.py)
  - `backend/scripts/run_main_v5_with_cuda.sh` - Whisper với Fusion diarization (main_v5.py, khuyến nghị ⭐)
  - `backend/scripts/run_main_sensevoice_with_cuda.sh` - SenseVoice

### 4. Cấu hình model (tùy chọn)

Chỉnh sửa file tương ứng để thay đổi model và device:
- `backend/main.py` - Whisper version cơ bản
- `backend/main_v2.py` - Whisper với bootstrap speaker detection
- `backend/main_v3.py` - Whisper với RealtimeSpeakerDiarization (khuyến nghị)
- `backend/main_sensevoice.py` - SenseVoice

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

**Chọn version:**
- `main.py`: Version cơ bản, không có speaker detection trong realtime
- `main_v2.py`: Version với bootstrap clustering speaker detection
- `main_v3.py`: Version với RealtimeSpeakerDiarization class (khuyến nghị - đơn giản và hiệu quả)

### 5. Cấu hình Pyannote (cho main_v4.py và main_v5.py)

**Yêu cầu HuggingFace Token:**
```bash
# Tạo .env file trong thư mục backend
echo "HF_TOKEN=your_huggingface_token_here" > backend/.env
```
- Lấy token từ: https://huggingface.co/settings/tokens
- Accept điều khoản của model: https://huggingface.co/pyannote/speaker-diarization-community-1
- File sử dụng: `main_v4.py`, `main_v5.py`

**Lưu ý:**
- Pyannote model yêu cầu accept user agreement trước
- Token cần có quyền read
- main_v5.py có thể chạy với chỉ SpeechBrain nếu không có token (set `use_pyannote=False`)

### 6. Cấu hình API-based versions (tùy chọn)

**Gemini API:**
```bash
# Tạo .env file trong thư mục backend
echo "GEMINI_API_KEY=your_api_key_here" > backend/.env
```
- Lấy API key từ: https://aistudio.google.com/apikey
- File sử dụng: `main_gemini.py`, `main_v2_gemini.py`

**Qwen Audio (Modal):**
```bash
# Deploy lên Modal
cd backend
modal deploy qwen_audio_modal.py

# Cập nhật API_URL trong main_qwenaudio.py
API_URL = "https://your-modal-url/v1/chat/completions"
```
- Yêu cầu: Modal account và API token
- Chi phí: Theo GPU usage trên Modal platform

**Qwen Omni API:**
```bash
# Cập nhật API_URL trong main_qwenomni.py hoặc main_whispersmall_qwenomni.py
API_URL = "https://your-qwen-endpoint/v1/chat/completions"
```
- Yêu cầu: Custom API endpoint hoặc ngrok tunnel
- File sử dụng: `main_qwenomni.py`, `main_whispersmall_qwenomni.py`

**Lưu ý:**
- API-based versions có chi phí sử dụng
- Gemini: Miễn phí tier có giới hạn requests/day
- Qwen Audio: Chi phí GPU trên Modal (~$0.5-1/hour)
- Qwen Omni: Tùy thuộc vào hosting solution

## ▶️ Chạy ứng dụng

### 1. Khởi động server (CPU)
```bash
cd backend
python main.py          # Version cơ bản
# hoặc
python main_v2.py       # Version với bootstrap speaker detection
# hoặc
python main_v3.py       # Version với RealtimeSpeakerDiarization (khuyến nghị)
```

Server sẽ chạy tại: `http://localhost:8917`

### 2. Chạy với CUDA (nếu có GPU)

**Yêu cầu:**
- CUDA 12.1 đã được cài đặt
- File `~/activate_cuda121.sh` để activate CUDA 12.1
- Conda environment `v2t` đã được tạo và có các dependencies

**Chạy server (Whisper-based):**
```bash
cd backend/scripts

# Whisper versions (miễn phí, mã nguồn mở)
./run_main_with_cuda.sh          # main.py - Whisper cơ bản (port 8917)
./run_main_v2_with_cuda.sh       # main_v2.py - Bootstrap speaker detection (port 8917)
./run_main_v3_with_cuda.sh       # main_v3.py - SpeechBrain diarization (port 8917)
./run_main_v4_with_cuda.sh       # main_v4.py - Pyannote diarization (port 8917)
./run_main_v5_with_cuda.sh       # main_v5.py - Fusion diarization (port 8917, khuyến nghị ⭐)
./run_main_sensevoice_with_cuda.sh  # SenseVoice (port 8918)
```

**Chạy server (API-based):**
```bash
cd backend/scripts

# API-based versions (chính xác cao, có phí)
./run_main_gemini_with_cuda.sh              # Gemini 2.5 (khuyến nghị cho chính xác ⭐)
./run_main_v2_gemini_with_cuda.sh           # Gemini + Bootstrap clustering
./run_main_qwenaudio_with_cuda.sh           # Qwen Audio (self-hosted)
./run_main_qwenomni_with_cuda.sh            # Qwen Omni (API endpoint)
./run_main_whispersmall_qwenomni_with_cuda.sh  # Hybrid: Whisper + Qwen (khuyến nghị cho hybrid ⭐)
```

**Với auto-reload (development):**
```bash
# Thêm --reload flag vào bất kỳ script nào
./run_main_with_cuda.sh --reload
./run_main_v3_with_cuda.sh --reload
./run_main_gemini_with_cuda.sh --reload
./run_main_whispersmall_qwenomni_with_cuda.sh --reload
```

**Lưu ý:**
- Scripts tự động:
  - Activate conda environment `v2t`
  - Setup CUDA paths và cuDNN libraries từ conda env
  - Cấu hình `LD_LIBRARY_PATH` cho cuDNN và PyTorch libraries
  - Sử dụng GPU device 1 (`CUDA_VISIBLE_DEVICES=1`)
- **Ports:**
  - Whisper versions: port **8917**
  - SenseVoice: port **8918**
  - Tất cả API-based versions: port **8917**
- **Khuyến nghị:**
  - **Miễn phí tốt nhất**: `main_v5.py` - Fusion diarization (Pyannote + SpeechBrain) ⭐
  - **Miễn phí nhanh**: `main_v3.py` - SpeechBrain diarization
  - **Chính xác cao (API)**: `main_gemini.py` - Google Gemini API ⭐
  - **Hybrid**: `main_whispersmall_qwenomni.py` - Cân bằng tốc độ & chính xác ⭐
- **API-based versions yêu cầu:**
  - Gemini: `GEMINI_API_KEY` trong `.env` file
  - Qwen Audio: Modal deployment hoặc local vLLM server
  - Qwen Omni: Custom API endpoint
- Nếu gặp lỗi, kiểm tra:
  - Conda env `v2t` đã được tạo chưa
  - File `~/activate_cuda121.sh` có tồn tại không
  - CUDA 12.1 đã được cài đặt đúng chưa
  - API keys/endpoints đã được cấu hình chưa (cho API versions)

### 3. Truy cập ứng dụng
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
{"type": "ready"}  // Khi sẵn sàng nhận audio
{"type": "partial", "text": "...", "speaker_id": "spk_01", "language": "vi", "language_probability": 0.95, "segments": [{"start": 0.0, "end": 1.0, "text": "...", "speaker_id": "spk_01"}]}  // Transcript tạm thời từ buffer ~1s
{"type": "full", "text": "...", "speaker_id": "spk_01", "language": "vi", "language_probability": 0.95, "segments": [...]}  // Transcript chính xác từ toàn bộ đoạn speech (khi phát hiện kết thúc)
{"type": "final", "text": ""}  // Khi session kết thúc
{"type": "error", "message": "..."}  // Nếu có lỗi
{"type": "pong"}  // Phản hồi cho ping
```

**Lưu ý về Partial vs Full messages:**
- **Partial**: Được gửi liên tục từ buffer ~1s, có độ trễ thấp nhưng độ chính xác có thể chưa tối ưu
- **Full**: Được gửi khi phát hiện kết thúc đoạn speech (silence hoặc repeated substring), transcribe lại với cấu hình tốt hơn (beam_size=5, best_of=5 cho Whisper hoặc batch_size_s=20 cho SenseVoice) để có độ chính xác cao hơn
- Frontend tự động thay thế phần partial tương ứng bằng full text khi nhận được full message

4. Client có thể gửi ping để kiểm tra kết nối:
```json
{"event": "ping"}
```
Server sẽ trả về `{"type": "pong"}`

5. Client gửi để dừng:
```json
{"event": "stop"}
```
Khi nhận được stop, server sẽ:
- Flush phần còn lại trong buffer
- Transcribe lại full_buffer nếu còn (nếu chưa được gửi)
- Gửi final message và đóng kết nối

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
│   ├── main.py                          # FastAPI server với Whisper model (version cơ bản)
│   ├── main_v2.py                       # FastAPI server với Whisper + bootstrap speaker detection
│   ├── main_v3.py                       # FastAPI server với Whisper + SpeechBrain diarization
│   ├── main_v4.py                       # FastAPI server với Whisper + Pyannote diarization
│   ├── main_v5.py                       # FastAPI server với Whisper + Fusion diarization (khuyến nghị ⭐)
│   ├── main_sensevoice.py               # FastAPI server với SenseVoice model
│   ├── main_gemini.py                   # FastAPI server với Gemini model
│   ├── main_qwenaudio.py                # FastAPI server với Qwen Audio
│   ├── main_qwenomni.py                 # FastAPI server với Qwen Omni
│   ├── main_whispersmall_qwenomni.py    # FastAPI server với Whisper Small + Qwen Omni
│   │
│   ├── requirements.txt                 # Python dependencies
│   ├── constraints.txt                  # Version constraints cho dependencies
│   │
│   ├── scripts/                         # Scripts để chạy server với CUDA
│   │   ├── activate_cuda_env.sh         # Script activate CUDA environment
│   │   ├── run_main_with_cuda.sh        # Chạy main.py với CUDA
│   │   ├── run_main_v2_with_cuda.sh     # Chạy main_v2.py với CUDA
│   │   ├── run_main_v3_with_cuda.sh     # Chạy main_v3.py với CUDA
│   │   ├── run_main_v4_with_cuda.sh     # Chạy main_v4.py với CUDA
│   │   ├── run_main_v5_with_cuda.sh     # Chạy main_v5.py với CUDA (khuyến nghị ⭐)
│   │   ├── run_main_sensevoice_with_cuda.sh  # Chạy SenseVoice với CUDA
│   │   ├── run_main_gemini_with_cuda.sh      # Chạy Gemini với CUDA
│   │   └── ...                          # Các scripts khác
│   │
│   ├── docs/                            # Tài liệu
│   │   ├── whisper_docs.md              # Tài liệu về Whisper model
│   │   ├── sensevoice_docs.md           # Tài liệu về SenseVoice model
│   │   └── whisper_vs_sensevoice.md     # So sánh Whisper vs SenseVoice
│   │
│   ├── get_audio.py                     # Utilities để decode audio/video files
│   ├── silero_vad.py                    # VAD (Voice Activity Detection) sử dụng Silero VAD
│   ├── fix_speechbrain.py               # Patch compatibility cho SpeechBrain với huggingface_hub
│   ├── speechbrain_diarization.py       # RealtimeSpeakerDiarization class với SpeechBrain ECAPA-TDNN
│   ├── pyanote_diarization.py           # RealtimeSpeakerDiarization class với Pyannote embeddings
│   ├── fusion_diarization.py            # Fusion diarization (Pyannote + SpeechBrain) với nhiều fusion methods
│   ├── qwen_audio_modal.py              # Qwen Audio model integration
│   │
│   ├── check_cuda.py                    # Script kiểm tra CUDA availability
│   ├── test.py                          # Script test
│   ├── test_call_qwen_audio.py          # Test Qwen Audio
│   │
│   ├── eval/                            # Thư mục evaluation/testing
│   │   ├── eval.py                      # Script đánh giá model
│   │   ├── en/                          # Test cases tiếng Anh
│   │   └── ja/                          # Test cases tiếng Nhật
│   │
│   ├── pretrained_models/               # Thư mục chứa pretrained models
│   └── __pycache__/                     # Python cache (tự động tạo)
│
├── frontend/
│   └── index.html                       # Frontend UI (HTML + JavaScript)
│
└── README.md                            # Tài liệu này
```

### Mô tả các thành phần chính

#### Backend Core Files (Whisper versions)
- **`main.py`**: Server cơ bản sử dụng Whisper model (`faster-whisper`)
  - Realtime: Dual-buffer strategy với partial và full messages
  - File upload: Transcribe với cấu hình tối ưu cho độ chính xác
  - Speaker detection: Không hỗ trợ trong realtime (chỉ file upload)

- **`main_v2.py`**: Server với bootstrap clustering speaker detection
  - Realtime: Dual-buffer strategy + bootstrap clustering
  - Speaker detection: Bootstrap phase thu thập embeddings → K-means clustering → nhận diện speaker
  - Phức tạp hơn nhưng có khả năng tự động phân cụm speakers

- **`main_v3.py`**: Server với SpeechBrain diarization
  - Realtime: Dual-buffer strategy + SpeechBrain ECAPA-TDNN
  - Speaker detection: 2-tier matching (EMA embedding + cluster centroid)
  - Đơn giản, hiệu quả, persistent speaker memory
  - Preload model một lần, tái sử dụng cho tất cả sessions

- **`main_v4.py`**: Server với Pyannote diarization
  - Realtime: Dual-buffer strategy + Pyannote embeddings
  - Speaker detection: 2-tier matching với pyannote.audio
  - Chất lượng embeddings cao, độ chính xác tốt
  - Session management cho multiple conversations

- **`main_v5.py`**: Server với Fusion diarization (KHUYẾN NGHỊ ⭐)
  - Realtime: Dual-buffer strategy + Fusion (Pyannote + SpeechBrain)
  - Speaker detection: Fusion embeddings với nhiều strategies
  - Fusion methods: score_level, concatenate, weighted_average, v.v.
  - Kết hợp ưu điểm của cả hai hệ thống, chính xác cao nhất
  - Session management và dimension alignment tự động

- **`main_sensevoice.py`**: Server sử dụng SenseVoice model (`funasr`)
  - Realtime: Tương tự Whisper với dual-buffer strategy
  - File upload: Hỗ trợ multi-language detection và fallback timestamp synthesis

#### Backend Core Files (API-based versions)

- **`main_gemini.py`**: Server sử dụng Google Gemini 2.5 API
  - Realtime: Whisper (partial) → Gemini API (full transcription)
  - File upload: Gemini API cho toàn bộ file
  - Speaker detection: 3-tier matching (EMA → centroid → verification model)
  - Ưu điểm: Chính xác rất cao, hỗ trợ đa ngôn ngữ tốt nhất
  - Yêu cầu: `GEMINI_API_KEY` environment variable

- **`main_qwenaudio.py`**: Server sử dụng Qwen2-Audio-7B qua Modal/vLLM
  - Realtime: Whisper (partial) → Qwen Audio API (full)
  - File upload: Qwen Audio API
  - Speaker detection: 2-tier matching (EMA + verification)
  - Ưu điểm: Self-hosted, kiểm soát data, chính xác cao
  - Yêu cầu: Modal deployment hoặc local vLLM server

- **`main_qwenomni.py`**: Server sử dụng Qwen3-Omni-30B API
  - Realtime: Whisper (partial) → Qwen Omni API (full)
  - File upload: Qwen Omni API
  - Speaker detection: 2-tier matching
  - Ưu điểm: Multimodal, chính xác cao, hỗ trợ nhiều ngôn ngữ
  - Yêu cầu: Custom API endpoint

- **`main_whispersmall_qwenomni.py`**: Hybrid version (KHUYẾN NGHỊ cho API ⭐)
  - Realtime: Whisper Small (nhanh, độ trễ thấp)
  - Full transcription: Qwen Omni API (chính xác cao)
  - Speaker detection: 2-tier matching
  - Ưu điểm: Cân bằng tốc độ và chính xác, tốt nhất cho production
  - Yêu cầu: Local Whisper + Qwen API endpoint

- **`main_v2_gemini.py`**: Version kết hợp bootstrap speaker detection + Gemini
  - Tương tự main_v2.py nhưng sử dụng Gemini cho transcription
  - Bootstrap clustering cho speaker detection
  - Chính xác rất cao cho cả transcript và speaker

#### Speaker Diarization Module
- **`speechbrain_diarization.py`**: RealtimeSpeakerDiarization với SpeechBrain
  - SpeechBrain ECAPA-TDNN embeddings
  - 2-tier matching: EMA embedding (fast) + cluster centroid (robust)
  - Persistent speaker memory với exponential moving average
  - Max speakers constraint với force-assignment
  - Session management cho multiple conversations
  - Preloaded model support để tối ưu hiệu suất

- **`pyanote_diarization.py`**: RealtimeSpeakerDiarization với Pyannote
  - Pyannote.audio embeddings (high-quality)
  - 2-tier matching: EMA embedding + cluster centroid
  - Session management và persistent tracking
  - Tích hợp với pyannote/speaker-diarization-community-1
  - Requires HuggingFace token

- **`fusion_diarization.py`**: Fusion Speaker Diarization (⭐ TỐT NHẤT)
  - Kết hợp Pyannote + SpeechBrain embeddings
  - **Fusion methods**:
    - `score_level`: Tính similarity riêng rồi kết hợp (khuyến nghị)
    - `concatenate`: [E1 ; E2]
    - `normalized_average`: (norm(E1) + norm(E2)) / 2
    - `weighted_average`: α*E1 + (1-α)*E2
    - `product`: norm(E1) ⊙ norm(E2)
    - `max_pool`: max(E1, E2)
    - `learned_concat`: [w1*E1 ; w2*E2]
  - Dimension alignment tự động (min/max/pad_zero)
  - Session management và NaN handling
  - Chính xác cao nhất, tận dụng cả hai hệ thống

#### Utility Modules
- **`get_audio.py`**: Xử lý decode audio/video files thành numpy array (16kHz mono) sử dụng `av` (PyAV)
- **`silero_vad.py`**: Voice Activity Detection để phát hiện các đoạn có giọng nói, loại bỏ im lặng
- **`fix_speechbrain.py`**: Patch để fix compatibility issue giữa SpeechBrain và huggingface_hub (cần cho speaker detection)
- **`qwen_audio_modal.py`**: Modal deployment script cho Qwen2-Audio-7B
  - Tự động deploy Qwen Audio model lên Modal platform
  - Sử dụng vLLM inference engine (GPU L4/H100)
  - Auto-scaling với scaledown window 15 phút
  - OpenAI-compatible API endpoint
  - Hỗ trợ audio streaming và batch processing

#### CUDA Scripts (trong `scripts/`)

**Whisper-based:**
- **`run_main_with_cuda.sh`**: Script tự động setup CUDA/cuDNN và chạy main.py trên port 8917
- **`run_main_v2_with_cuda.sh`**: Script chạy main_v2.py (bootstrap speaker detection)
- **`run_main_v3_with_cuda.sh`**: Script chạy main_v3.py (SpeechBrain diarization)
- **`run_main_v4_with_cuda.sh`**: Script chạy main_v4.py (Pyannote diarization)
- **`run_main_v5_with_cuda.sh`**: Script chạy main_v5.py (Fusion diarization, khuyến nghị ⭐)
- **`run_main_sensevoice_with_cuda.sh`**: Script chạy SenseVoice server trên port 8918

**API-based:**
- **`run_main_gemini_with_cuda.sh`**: Script chạy main_gemini.py (Gemini API)
- **`run_main_v2_gemini_with_cuda.sh`**: Script chạy main_v2_gemini.py (Bootstrap + Gemini)
- **`run_main_qwenaudio_with_cuda.sh`**: Script chạy main_qwenaudio.py (Qwen Audio)
- **`run_main_qwenomni_with_cuda.sh`**: Script chạy main_qwenomni.py (Qwen Omni)
- **`run_main_whispersmall_qwenomni_with_cuda.sh`**: Script chạy main_whispersmall_qwenomni.py (Hybrid, khuyến nghị ⭐)

**Helper scripts:**
- **`activate_cuda_env.sh`**: Script helper để activate CUDA environment
- **`check_cuda.py`**: Script kiểm tra CUDA có sẵn và hoạt động không

**Tất cả scripts:**
- Tự động activate conda environment `v2t`
- Setup CUDA 12.1 paths và cuDNN libraries
- Cấu hình `LD_LIBRARY_PATH` cho cuDNN và PyTorch
- Sử dụng GPU device 1 (`CUDA_VISIBLE_DEVICES=1`)
- Hỗ trợ `--reload` flag cho development mode

#### Documentation (trong `docs/`)
- **`whisper_docs.md`**: Tài liệu chi tiết về cách sử dụng Whisper model
- **`sensevoice_docs.md`**: Tài liệu chi tiết về cách sử dụng SenseVoice model
- **`whisper_vs_sensevoice.md`**: So sánh ưu/nhược điểm giữa 2 models

#### Evaluation (trong `eval/`)
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
   - **Full Transcript**: 
     - Hiển thị partial text tạm thời từ buffer ~1s (độ trễ thấp)
     - Tự động được thay thế bằng full text chính xác hơn khi phát hiện kết thúc đoạn speech
     - Có speaker ID nếu bật detect speaker
   - **Segments**: Danh sách các đoạn có timestamp (start → end) và speaker ID (nếu có)
     - Segments từ partial messages được thêm vào liên tục
     - Segments từ full messages có thể cập nhật/thay thế segments tương ứng
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

## 🔀 Fusion Methods (main_v5.py)

Fusion diarization kết hợp embeddings từ Pyannote và SpeechBrain để đạt độ chính xác cao hơn. Hỗ trợ nhiều fusion strategies:

### Fusion Methods

**1. Score-level Fusion (Khuyến nghị ⭐)**
```python
fusion_method="score_level"
fusion_alpha=0.4  # Weight: 0.4*Pyannote + 0.6*SpeechBrain
```
- Tính similarity riêng cho mỗi hệ thống, sau đó kết hợp scores
- Formula: `final_score = α*score_pyannote + (1-α)*score_speechbrain`
- Ưu điểm: Linh hoạt, tận dụng thế mạnh của từng hệ thống
- Use case: Khi muốn điều chỉnh trọng số giữa hai hệ thống

**2. Normalized Average**
```python
fusion_method="normalized_average"
```
- Average của normalized embeddings: `E = (norm(E1) + norm(E2)) / 2`
- Ưu điểm: Đơn giản, cân bằng giữa hai hệ thống
- Use case: Khi hai hệ thống có độ chính xác tương đương

**3. Weighted Average**
```python
fusion_method="weighted_average"
fusion_alpha=0.5  # Weight for Pyannote
```
- Weighted average: `E = α*norm(E1) + (1-α)*norm(E2)`
- Ưu điểm: Điều chỉnh được trọng số
- Use case: Khi muốn ưu tiên một hệ thống hơn

**4. Concatenate**
```python
fusion_method="concatenate"
```
- Simple concatenation: `E = [E1 ; E2]`
- Ưu điểm: Giữ nguyên thông tin từ cả hai
- Use case: Khi cần tất cả features từ cả hai hệ thống

**5. Product**
```python
fusion_method="product"
```
- Element-wise product: `E = norm(E1) ⊙ norm(E2)`
- Ưu điểm: Nhấn mạnh features chung
- Use case: Khi muốn lọc noise

**6. Max Pool**
```python
fusion_method="max_pool"
```
- Max pooling: `E = max(norm(E1), norm(E2))`
- Ưu điểm: Chọn features mạnh nhất
- Use case: Khi muốn robust với outliers

**7. Learned Concat**
```python
fusion_method="learned_concat"
fusion_weights=(1.2, 0.8)  # Custom weights
```
- Weighted concatenation: `E = [w1*E1 ; w2*E2]`
- Ưu điểm: Tùy chỉnh weights cho từng embedding
- Use case: Sau khi đã học weights tối ưu

### Dimension Alignment

Pyannote (256-dim) và SpeechBrain (192-dim) có dimensions khác nhau. Fusion diarization tự động xử lý:

```python
dimension_alignment="min"     # Truncate to 192-dim (mặc định, nhanh)
dimension_alignment="max"     # Pad zeros to 256-dim (giữ tất cả info)
dimension_alignment="pad_zero"  # Tương tự max
```

### Configuration Example

```python
diarization_model = RealtimeSpeakerDiarization(
    fusion_method="score_level",      # Fusion strategy
    fusion_alpha=0.4,                  # Weight (0.4 Pyannote, 0.6 SpeechBrain)
    dimension_alignment="max",         # Dimension handling
    similarity_threshold=0.6,          # Matching threshold
    embedding_update_weight=0.3,       # EMA update weight
    min_similarity_gap=0.25,           # Minimum gap for matching
    use_pyannote=True,                 # Enable Pyannote
    use_speechbrain=True,              # Enable SpeechBrain
    pyannote_config={
        "model_name": "pyannote/speaker-diarization-community-1",
        "token": os.getenv("HF_TOKEN"),
    },
    speechbrain_config={
        "model_name": "speechbrain/spkrec-ecapa-tdnn-voxceleb",
    }
)
```

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

**Partial transcription** (trong `run_transcribe_on_buffer()`):
- `beam_size=1`: Tối ưu tốc độ cho streaming
- `best_of=1`: Tối ưu tốc độ
- `condition_on_previous_text=False`: Giảm độ trễ, các chunk độc lập

**Full transcription** (trong `run_transcribe_on_full_buffer()`):
- Whisper: `beam_size=5`, `best_of=5`: Tăng độ chính xác khi transcribe lại toàn bộ đoạn speech
- SenseVoice: `batch_size_s=20`: Tăng batch size để chính xác hơn
- Được gọi tự động khi phát hiện kết thúc đoạn speech (silence hoặc repeated substring)

### Tối ưu hóa cho file upload

Trong `transcribe_file()`:
- Whisper: `beam_size=5`, `best_of=5`: Tăng độ chính xác
- SenseVoice: `batch_size_s=20`: Tăng batch size
- `condition_on_previous_text=True` (Whisper): Sử dụng ngữ cảnh

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
- Dùng script (trong thư mục `backend/scripts/`):
  - `run_main_with_cuda.sh` - main.py
  - `run_main_v2_with_cuda.sh` - main_v2.py
  - `run_main_v3_with_cuda.sh` - main_v3.py (khuyến nghị)
  - `run_main_sensevoice_with_cuda.sh` - SenseVoice
- Hoặc tự set:
  ```bash
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
  ```

### Speaker detection không hoạt động tốt
- **main_v2.py**: Cần đợi bootstrap phase (~1 phút) để thu thập đủ dữ liệu
- **main_v3.py**: SpeechBrain - Hoạt động ngay từ đầu, không cần bootstrap
- **main_v4.py**: Pyannote - Yêu cầu HF_TOKEN, chất lượng cao
- **main_v5.py** (khuyến nghị ⭐): Fusion - Tốt nhất, kết hợp cả hai hệ thống
- Kiểm tra:
  - `max_speakers` có được set đúng không
  - Audio có đủ rõ để trích xuất embedding không
  - Log có hiển thị similarity scores không
  - HF_TOKEN có được set đúng cho main_v4/v5 không
  - Embeddings có bị zero vectors không (check warnings)

### Chọn version nào?
- **Mới bắt đầu hoặc testing**: `main.py` - Đơn giản nhất
- **Cần speaker detection tốt nhất (miễn phí)**: `main_v5.py` ⭐ - Fusion diarization, chính xác cao nhất
- **Cần speaker detection nhanh**: `main_v3.py` - SpeechBrain, nhanh và ổn định
- **Cần speaker detection chất lượng cao**: `main_v4.py` - Pyannote embeddings
- **Cần tự động phân cụm speakers**: `main_v2.py` - Bootstrap clustering
- **Cần chính xác cao nhất (API)**: `main_gemini.py` ⭐ - Google Gemini API
- **Cần self-hosted API**: `main_qwenaudio.py` - Deploy trên Modal
- **Cần hybrid tối ưu**: `main_whispersmall_qwenomni.py` ⭐ - Tốc độ + chính xác

### API không hoạt động
**Gemini API:**
- Kiểm tra `GEMINI_API_KEY` trong `.env` file
- Verify API key tại: https://aistudio.google.com/apikey
- Kiểm tra quota limits (free tier có giới hạn)
- Log error để xem chi tiết lỗi từ API

**Qwen Audio (Modal):**
- Verify Modal deployment: `modal app list`
- Kiểm tra URL trong `API_URL` variable
- Test endpoint: `curl https://your-modal-url/health`
- Kiểm tra Modal logs: `modal app logs qwen-audio-modal`

**Qwen Omni:**
- Kiểm tra API endpoint đang chạy
- Verify API_URL đúng trong code
- Test với curl/Postman trước
- Kiểm tra network/firewall nếu dùng local endpoint

### Modal deployment issues
```bash
# Install Modal CLI
pip install modal

# Login
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET

# Deploy
cd backend
modal deploy qwen_audio_modal.py

# Test
modal run qwen_audio_modal.py

# Check logs
modal app logs qwen-audio-modal

# Check status
modal app list
```

## 📝 Ghi chú

### Về Models và Versions
- Model Whisper/SenseVoice được tải tự động lần đầu chạy
- Speaker diarization models được preload một lần để tối ưu hiệu suất:
  - `main_v3.py`: SpeechBrain ECAPA-TDNN
  - `main_v4.py`: Pyannote.audio embeddings
  - `main_v5.py`: Cả Pyannote và SpeechBrain (fusion)
- File tạm sẽ tự động xóa sau khi xử lý xong
- **Khuyến nghị sử dụng `main_v5.py`** - Version tối ưu nhất với Fusion diarization (Pyannote + SpeechBrain)

### Realtime Transcription
- **Dual-buffer strategy**:
  - **recv_buffer**: Buffer ~1s để transcribe nhanh và gửi partial messages (độ trễ thấp)
  - **full_buffer**: Tích lũy toàn bộ audio của một đoạn speech, được transcribe lại với cấu hình tốt hơn khi phát hiện kết thúc
  - Kết thúc đoạn speech được phát hiện bằng: silence detection hoặc repeated substring detection
- **Partial vs Full Messages**:
  - Partial messages: Được gửi liên tục từ buffer ~1s, có độ trễ thấp
  - Full messages: Được gửi khi phát hiện kết thúc đoạn speech, có độ chính xác cao hơn
  - Frontend tự động thay thế phần partial tương ứng bằng full text để cải thiện chất lượng transcript
- **Repeated Substring Detection**: Hệ thống tự động phát hiện chuỗi lặp lại (≥5 lần) để xác định kết thúc đoạn speech và trigger full transcription

### Speaker Detection
- **main.py**: Không hỗ trợ speaker detection trong realtime (chỉ file upload)
- **main_v2.py**: Bootstrap clustering approach
  - Bootstrap phase: Thu thập embeddings trong ~1 phút đầu
  - K-means clustering: Phân cụm speakers tự động
  - 3-tier matching: EMA embedding → cluster centroid → verification model
  - Phức tạp hơn nhưng có khả năng tự động phân cụm
- **main_v3.py**: SpeechBrain diarization
  - SpeechBrain ECAPA-TDNN embeddings
  - 2-tier matching: EMA embedding (fast) + cluster centroid (robust)
  - Persistent speaker memory và session management
  - Nhanh, ổn định, độ chính xác tốt
- **main_v4.py**: Pyannote diarization
  - Pyannote.audio embeddings (high-quality)
  - 2-tier matching: EMA embedding + cluster centroid
  - Session management và persistent tracking
  - Chất lượng embeddings cao, độ chính xác rất tốt
- **main_v5.py**: Fusion diarization (KHUYẾN NGHỊ ⭐)
  - **Kết hợp cả Pyannote và SpeechBrain**
  - **Fusion methods**: score_level (khuyến nghị), concatenate, weighted_average, product, max_pool, learned_concat
  - **2-tier matching** với fused embeddings
  - **Persistent speaker memory**: Lưu trữ cả fused embeddings và individual embeddings
  - **Dimension alignment**: Tự động xử lý dimension mismatch (Pyannote 256-dim vs SpeechBrain 192-dim)
  - **Max speakers constraint**: Force-assign vào speaker có similarity cao nhất khi đạt limit
  - **Session management**: Multiple conversations độc lập
  - **NaN handling**: Xử lý zero vectors và NaN similarities
  - **Preloaded models**: Load cả hai models một lần, tái sử dụng cho tất cả sessions
  - Speaker ID được gán dạng `SPEAKER_00`, `SPEAKER_01`, ...
  - **Chính xác cao nhất**, tận dụng ưu điểm của cả hai hệ thống
- Với video files, audio sẽ được extract tự động nếu có ffmpeg

### Performance Metrics
- **RTF (Real-Time Factor)**: 
  - RTF = processing_time / audio_duration
  - RTF < 1.0: Xử lý nhanh hơn thời gian thực (tốt)
  - RTF = 1.0: Xử lý bằng thời gian thực
  - RTF > 1.0: Xử lý chậm hơn thời gian thực

### UI/UX Features
- Các tùy chọn cấu hình tự động bị disable khi đang xử lý để tránh thay đổi không mong muốn
- Hệ thống tự động hiển thị/ẩn các phần tử UI dựa trên trạng thái (chỉ hiện transcript khi đang ghi)
- WebSocket hỗ trợ ping/pong để kiểm tra kết nối
- Auto-detect WebSocket URL từ port của backend

### Technical Details
- **Speaker tracking được reset cho mỗi session** (WebSocket connection hoặc file upload)
- **Session cleanup**: Tự động xóa session data để tránh memory leak
- **VAD (Voice Activity Detection)**: Sử dụng Silero VAD để lọc silence
- **Audio format**: PCM16 mono 16kHz cho WebSocket, tự động convert cho file upload

## 🔧 Dependencies

Xem `backend/requirements.txt` để biết danh sách đầy đủ.

**Core Dependencies (Tất cả versions):**
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `numpy`: Xử lý audio
- `soundfile`: Audio I/O
- `av` (PyAV): Decode audio/video files
- `torch`: Deep learning framework
- `speechbrain`: Speaker diarization (ECAPA-TDNN) - main_v3.py, main_v5.py
- `pyannote.audio`: Speaker diarization (embeddings) - main_v4.py, main_v5.py
- `scipy`: Scientific computing (cdist for similarity)

**Whisper-based Versions:**
- `faster-whisper`: Whisper backend (main.py, main_v2.py, main_v3.py)
- `funasr`: SenseVoice backend (main_sensevoice.py)
- `silero-vad`: Voice Activity Detection

**API-based Versions:**
- `google-genai`: Google Gemini API client (main_gemini.py, main_v2_gemini.py)
- `requests`: HTTP client (main_qwenaudio.py, main_qwenomni.py, main_whispersmall_qwenomni.py)
- `python-dotenv`: Environment variables management
- `aiohttp`: Async HTTP client (optional)

**Modal Deployment:**
- `modal`: Modal platform SDK (qwen_audio_modal.py)
- Yêu cầu Modal account và token

**Development:**
- `pytest`: Testing framework
- `black`: Code formatter
- `flake8`: Linter

## 📄 License

Dự án này sử dụng các thư viện mã nguồn mở. Vui lòng xem license của từng thư viện.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📧 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên repository.

