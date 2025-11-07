import asyncio
import json
import logging
import subprocess
import tempfile
import os
import re
from typing import Optional
from datetime import datetime
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# -------- Setup CUDA/cuDNN Library Path ----------
# Tự động thêm cuDNN libraries vào LD_LIBRARY_PATH nếu chưa có
def setup_cudnn_path():
    """Tự động tìm và thêm cuDNN libraries vào LD_LIBRARY_PATH."""
    import sys
    
    # Lấy conda env path từ sys.executable
    if sys.executable and 'envs' in sys.executable:
        env_path = sys.executable.rsplit('/envs/', 1)[0] + '/envs/' + sys.executable.rsplit('/envs/', 1)[1].split('/', 1)[0]
        
        # Tìm cuDNN libraries
        possible_paths = [
            f"{env_path}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cudnn/lib",
            f"{env_path}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/torch/lib",
        ]
        
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = []
        
        for path in possible_paths:
            if os.path.exists(path) and path not in current_ld_path:
                new_paths.append(path)
        
        if new_paths:
            os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths + current_ld_path.split(':')) if current_ld_path else ':'.join(new_paths)

# -------- Setup Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------- Model config ----------
MODEL_NAME = "iic/SenseVoiceSmall"
DEVICE = "cuda"  # "cuda" nếu có GPU, "cpu" nếu không có GPU

if DEVICE == "cuda":
    try:
        setup_cudnn_path()
    except Exception as e:
        pass  # Ignore nếu không setup được

    # Log LD_LIBRARY_PATH để debug
    if os.environ.get('LD_LIBRARY_PATH'):
        logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    else:
        logger.warning("LD_LIBRARY_PATH không được set. Nếu gặp lỗi cuDNN, hãy chạy: ./run_main_sensevoice_with_cuda.sh")

# -------- SenseVoice model ----------
from funasr import AutoModel

model = AutoModel(
    model=MODEL_NAME,
    vad_model="fsmn-vad",                     # bật VAD tích hợp
    vad_kwargs={"max_single_segment_time": 30000},
    device=DEVICE,
)

# -------- App ----------
app = FastAPI(title="Realtime Transcript API (SenseVoice)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Helpers ----
def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    """Chuyển bytes PCM16LE -> float32 [-1, 1] mono."""
    if not pcm16:
        return np.array([], dtype=np.float32)
    audio_i16 = np.frombuffer(pcm16, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    return audio_f32


def get_media_duration_seconds(file_path: str) -> float:
    """Lấy thời lượng media (giây) bằng ffprobe. Trả None nếu không có ffprobe hoặc lỗi."""
    try:
        out = subprocess.check_output([
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ], stderr=subprocess.STDOUT)
        return float(out.decode("utf-8").strip())
    except Exception:
        return None


async def run_transcribe_on_buffer(audio_buffer: bytes, lang_hint: Optional[str] = None, time_offset: float = 0.0) -> dict:
    """
    Streaming transcript (SenseVoice).
    Trả về dict với 'text' và 'segments' (có timestamp) của đoạn ~1s audio.
    """
    audio_f32 = pcm16_to_float32(audio_buffer)
    if audio_f32.size == 0:
        return {"text": "", "segments": []}
    
    # Cần tối thiểu ~0.5s audio để transcribe (8000 samples @ 16kHz)
    MIN_SAMPLES = 8000
    if audio_f32.size < MIN_SAMPLES:
        return {"text": "", "segments": []}

    # Kiểm tra audio có tín hiệu (RMS > ngưỡng nhỏ)
    rms = np.sqrt(np.mean(audio_f32 ** 2))
    if rms < 0.01:  # Ngưỡng im lặng
        return {"text": "", "segments": []}

    try:
        # SenseVoice: model dùng sample_rate 16000, input là numpy.float32
        res = model.generate(
            input=audio_f32,
            cache={},
            language=lang_hint or "auto",
            use_itn=False,         # không chuẩn hóa số thành chữ
            batch_size_s=10,
        )

        # SenseVoice trả list các dict, có thể có timestamp
        texts = []
        segments_list = []
        for r in res:
            if "text" in r and r["text"]:
                text = r["text"].strip()
                # text = 
                if text:
                    texts.append(text)
                    # Nếu có timestamp, dùng nó; nếu không, ước tính từ buffer
                    if "timestamp" in r and len(r["timestamp"]) >= 2:
                        segments_list.append({
                            "start": round(r["timestamp"][0] + time_offset, 2),
                            "end": round(r["timestamp"][1] + time_offset, 2),
                            "text": re.sub(r"<\|[^|>]+\|>", "", text).strip(),
                        })
                    else:
                        # Ước tính timestamp từ buffer size
                        buffer_duration = len(audio_buffer) / 2 / 16000
                        segments_list.append({
                            "start": round(time_offset, 2),
                            "end": round(time_offset + buffer_duration, 2),
                            "text": re.sub(r"<\|[^|>]+\|>", "", text).strip(),
                        })

        return {
            "text": re.sub(r"<\|[^|>]+\|>", "", " ".join(texts)).strip(),
            "segments": segments_list,
        }
    except Exception as e:
        print(f"Transcribe error: {e}")
        return {"text": "", "segments": []}


async def transcribe_file(file_path: str, lang_hint: Optional[str] = None) -> dict:
    """
    Transcribe file audio/video (offline).
    """
    try:
        res = model.generate(
            input=file_path,
            language=lang_hint or "auto",
            use_itn=False,
            batch_size_s=20,
        )

        # SenseVoice trả list kết quả, có timestamp
        texts, segments = [], []
        for r in res:
            if "text" in r and r["text"]:
                texts.append(r["text"])
                if "timestamp" in r:
                    segments.append({
                        "start": round(r["timestamp"][0], 2),
                        "end": round(r["timestamp"][1], 2),
                        "text": re.sub(r"<\|[^|>]+\|>", "", r["text"]).strip()
                    })
        # Nếu model không trả timestamp, synthesize segments dựa trên tổng thời lượng media
        if not segments and texts:
            duration = get_media_duration_seconds(file_path)
            # print(f"Duration: {duration}")
            if duration and duration > 0:
                total_chars = sum(len(re.sub(r"<\|[^|>]+\|>", "", t).strip()) for t in texts if t and t.strip())
                # print(f"Total chars: {total_chars}")
                # Nếu không tính được độ dài, tạo 1 segment duy nhất
                if total_chars <= 0:
                    segments = [{"start": 0.0, "end": round(duration, 2), "text": " ".join(texts).strip()}]
                else:
                    start = 0.0
                    new_segments = []
                    for idx, text in enumerate(texts):
                            for t in re.split(r"<\|[^|>]+\|>", text):
                                tt = (t or "").strip()
                                if not tt:
                                    continue
                                # Phân bổ thời lượng theo tỉ lệ độ dài ký tự
                                seg_dur = duration * (len(tt) / total_chars)
                                # print(f"Seg dur: {seg_dur}")
                                end = start + seg_dur
                                new_segments.append({
                                    "start": round(start, 2),
                                    "end": round(end, 2),
                                    "text": tt,
                                })
                                start = end
                    # Đảm bảo segment cuối cùng khớp đúng duration tổng
                    if new_segments:
                        new_segments[-1]["end"] = round(duration, 2)
                    segments = new_segments
            else:
                # Không lấy được duration -> trả 1 segment không có timestamp cụ thể (0, 0)
                segments = [{"start": 0.0, "end": 0.0, "text": " ".join(texts).strip()}]

        return {"text": re.sub(r"<\|[^|>]+\|>", "", " ".join(texts)).strip(), "segments": segments}
    except Exception as e:
        raise RuntimeError(f"File transcription error: {e}")


# ---- API endpoint for file upload ----
@app.post("/api/transcribe")
async def transcribe_uploaded_file(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    start_time = datetime.now()
    
    if not file.filename:
        logger.warning(f"[API] POST /api/transcribe - No file provided")
        return JSONResponse(status_code=400, content={"error": "No file provided"})

    logger.info(f"[API] POST /api/transcribe - Received request: filename={file.filename}, size={file.size if hasattr(file, 'size') else 'unknown'}, language={language or 'auto'}")

    ext = Path(file.filename).suffix.lower()
    lang_hint = language if language and language.lower() != "auto" else None

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_path = tmp_file.name
        try:
            content = await file.read()
            file_size = len(content)
            tmp_file.write(content)
            tmp_file.flush()
            logger.info(f"[API] File saved to temp: {tmp_path}, size={file_size} bytes")

            logger.info(f"[API] Starting transcription for {file.filename}...")
            result = await transcribe_file(tmp_path, lang_hint)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            segments_count = len(result.get("segments", []))
            text_length = len(result.get("text", ""))
            logger.info(f"[API] POST /api/transcribe - Completed: filename={file.filename}, processing_time={processing_time:.2f}s, segments={segments_count}, text_length={text_length}")
            
            return JSONResponse(content={"success": True, "filename": file.filename, **result})
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[API] POST /api/transcribe - Error: filename={file.filename}, error={str(e)}, processing_time={processing_time:.2f}s", exc_info=True)
            return JSONResponse(status_code=500, content={"error": str(e), "success": False})
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
                logger.debug(f"[API] Temp file deleted: {tmp_path}")
            except Exception as e:
                logger.warning(f"[API] Failed to delete temp file {tmp_path}: {e}")


# ---- WebSocket endpoint ----
@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket):
    client_ip = ws.client.host if ws.client else "unknown"
    session_start = datetime.now()
    logger.info(f"[WS] WebSocket connection request from {client_ip}")
    
    await ws.accept()
    logger.info(f"[WS] WebSocket connection accepted from {client_ip}")
    
    sample_rate = 16000
    lang_hint = None
    fmt = "pcm16"
    total_bytes_received = 0
    total_segments_sent = 0

    CHUNK_TARGET_BYTES = 32000
    recv_buffer = bytearray()
    running = True
    time_offset = 0.0  # Track thời gian tích lũy từ đầu session

    async def flush_and_transcribe():
        nonlocal recv_buffer, time_offset, total_segments_sent
        if len(recv_buffer) == 0:
            return
        try:
            buffer_size = len(recv_buffer)
            transcribe_start = datetime.now()
            # Tính thời gian của buffer này (bytes / 2 / sample_rate = seconds)
            buffer_duration = buffer_size / 2 / sample_rate
            result = await run_transcribe_on_buffer(bytes(recv_buffer), lang_hint, time_offset)
            transcribe_time = (datetime.now() - transcribe_start).total_seconds()
            recv_buffer = bytearray()
            if result["text"]:
                segments_count = len(result.get("segments", []))
                total_segments_sent += segments_count
                await ws.send_text(json.dumps({
                    "type": "partial",
                    "text": result["text"],
                    "segments": result["segments"]
                }))
                logger.debug(f"[WS] Sent partial: segments={segments_count}, text_length={len(result['text'])}, transcribe_time={transcribe_time:.3f}s, buffer_size={buffer_size} bytes")
            # Cập nhật time_offset cho chunk tiếp theo
            time_offset += buffer_duration
        except Exception as e:
            logger.error(f"[WS] Transcription error: {str(e)}", exc_info=True)
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))

    try:
        while running:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    try:
                        payload = json.loads(msg["text"])
                    except Exception:
                        continue
                    event = payload.get("event")
                    if event == "start":
                        fmt = payload.get("format", "pcm16")
                        lang = payload.get("language")
                        if lang and lang.lower() != "auto":
                            lang_hint = lang
                        # Reset time offset khi bắt đầu session mới
                        time_offset = 0.0
                        recv_buffer = bytearray()
                        total_bytes_received = 0
                        total_segments_sent = 0
                        session_start = datetime.now()
                        logger.info(f"[WS] Start event received: sample_rate={sample_rate}, format={fmt}, language={lang_hint or 'auto'}, client={client_ip}")
                        await ws.send_text(json.dumps({"type": "ready"}))
                        logger.info(f"[WS] Ready message sent to {client_ip}")
                    elif event == "stop":
                        logger.info(f"[WS] Stop event received from {client_ip}")
                        await flush_and_transcribe()
                        await ws.send_text(json.dumps({"type": "final", "text": ""}))
                        
                        session_duration = (datetime.now() - session_start).total_seconds()
                        logger.info(f"[WS] Session completed: client={client_ip}, duration={session_duration:.2f}s, total_bytes={total_bytes_received}, total_segments={total_segments_sent}")
                        
                        running = False
                elif "bytes" in msg:
                    chunk = msg["bytes"]
                    if not chunk:
                        continue
                    chunk_size = len(chunk)
                    total_bytes_received += chunk_size
                    recv_buffer.extend(chunk)
                    logger.debug(f"[WS] Received audio chunk: size={chunk_size} bytes, buffer_size={len(recv_buffer)}, total_received={total_bytes_received}")
                    if len(recv_buffer) >= CHUNK_TARGET_BYTES:
                        await flush_and_transcribe()
            elif msg["type"] == "websocket.disconnect":
                logger.info(f"[WS] WebSocket disconnect from {client_ip}")
                break
        await ws.close()
        logger.info(f"[WS] WebSocket connection closed for {client_ip}")
    except WebSocketDisconnect:
        session_duration = (datetime.now() - session_start).total_seconds() if 'session_start' in locals() else 0
        logger.info(f"[WS] WebSocket disconnected: client={client_ip}, duration={session_duration:.2f}s")
    except Exception as e:
        session_duration = (datetime.now() - session_start).total_seconds() if 'session_start' in locals() else 0
        logger.error(f"[WS] WebSocket error: client={client_ip}, error={str(e)}, duration={session_duration:.2f}s", exc_info=True)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
        except Exception:
            pass


@app.get("/")
async def read_root():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(frontend_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8917, reload=True)
