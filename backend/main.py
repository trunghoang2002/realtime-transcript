import asyncio
import json
import tempfile
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# -------- Whisper model ----------
# Model gợi ý: "small" (nhanh) / "medium" (chính xác) / "large-v3" (nặng)
from faster_whisper import WhisperModel

MODEL_NAME = "small"   # đổi thành "medium" hoặc "large-v3" nếu máy khỏe
DEVICE = "cpu"         # "cuda" nếu có GPU, "cpu" nếu không có GPU
COMPUTE_TYPE = "int8"  # "float16" trên GPU, "int8" hoặc "int8_float16" trên CPU

model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

# -------- App ----------
app = FastAPI(title="Realtime Transcript API", version="1.0.0")

# Cho phép gọi từ frontend
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

async def run_transcribe_on_buffer(
    audio_buffer: bytes,
    lang_hint: Optional[str] = None,
) -> str:
    """
    Chạy ASR trên vùng đệm audio (PCM16 mono 16kHz).
    Trả về text gộp (ngắn) của đoạn ~1s để streaming mượt.
    """
    audio_f32 = pcm16_to_float32(audio_buffer)
    if audio_f32.size == 0:
        return ""
    
    # Cần tối thiểu ~0.5s audio để transcribe (8000 samples @ 16kHz)
    MIN_SAMPLES = 8000
    if audio_f32.size < MIN_SAMPLES:
        return ""
    
    # Kiểm tra audio có tín hiệu (RMS > ngưỡng nhỏ)
    rms = np.sqrt(np.mean(audio_f32 ** 2))
    if rms < 0.01:  # Ngưỡng im lặng
        return ""

    try:
        # faster-whisper nhận numpy audio array (1-D float32, sample_rate=16000)
        # Sử dụng vad_filter để bỏ im lặng; beam_size thấp để nhanh hơn.
        segments, _ = model.transcribe(
            audio_f32,
            language=lang_hint,            # None -> auto-detect; "vi" -> tiếng Việt
            vad_filter=True,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,  # giúp các chunk độc lập, đỡ "lệch ngữ cảnh"
            temperature=0.0,
        )

        partial_texts = []
        for seg in segments:
            # seg.text đã loại bỏ khoảng trắng đầu/cuối
            t = seg.text.strip()
            if t:
                partial_texts.append(t)

        return " ".join(partial_texts)
    except (ValueError, RuntimeError) as e:
        # Xử lý lỗi khi segments rỗng (ví dụ: max() arg is an empty sequence)
        # hoặc khi audio quá ngắn/im lặng hoàn toàn
        error_msg = str(e).lower()
        if "empty" in error_msg or "max()" in error_msg or "sequence" in error_msg:
            return ""
        # Log lỗi khác để debug
        print(f"Transcription error: {e}")
        raise

async def transcribe_file(
    file_path: str,
    lang_hint: Optional[str] = None,
) -> dict:
    """
    Transcribe file audio/video.
    Trả về dict với 'text' và 'segments' (có timestamp).
    """
    try:
        # faster-whisper có thể nhận file path trực tiếp
        segments, info = model.transcribe(
            file_path,
            language=lang_hint,
            vad_filter=True,
            beam_size=5,  # Tăng độ chính xác cho file (không cần realtime)
            best_of=5,
            condition_on_previous_text=True,
            temperature=0.0,
        )

        full_text = []
        segments_list = []
        
        for seg in segments:
            text = seg.text.strip()
            if text:
                full_text.append(text)
                segments_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text,
                })

        return {
            "text": " ".join(full_text),
            "full_text": "\n".join(full_text),  # Mỗi segment một dòng
            "segments": segments_list,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
        }
    except Exception as e:
        print(f"File transcription error: {e}")
        raise

# ---- API endpoint for file upload ----
@app.post("/api/transcribe")
async def transcribe_uploaded_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """
    Upload file audio/video và transcribe.
    Hỗ trợ: .mp3, .wav, .m4a, .mp4, .avi, .mov, v.v.
    """
    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No file provided"}
        )

    # Lấy extension
    ext = Path(file.filename).suffix.lower()
    lang_hint = language if language and language.lower() != "auto" else None

    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_path = tmp_file.name
        try:
            # Lưu upload vào file tạm
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()

            # Transcribe
            result = await transcribe_file(tmp_path, lang_hint)
            
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                **result
            })
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "success": False}
            )
        finally:
            # Xóa file tạm
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

# ---- WebSocket endpoint ----
@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket):
    """
    Giao thức đơn giản:
    - Client gửi trước 1 JSON: {"event":"start","sample_rate":16000,"format":"pcm16","language":"vi|auto"}
    - Sau đó gửi các frame nhị phân (bytes PCM16 mono 16kHz).
    - Gửi {"event":"stop"} để kết thúc.
    - Server trả JSON: {"type":"partial","text": "..."} liên tục; và {"type":"final","text":"..."} khi stop.
    """
    await ws.accept()
    sample_rate = 16000
    lang_hint = None
    fmt = "pcm16"

    # buffer: gom ~1s audio rồi nhận dạng
    # 1s PCM16 mono 16kHz = 16000 samples * 2 bytes = 32000 bytes
    CHUNK_TARGET_BYTES = 32000

    recv_buffer = bytearray()
    running = True

    async def flush_and_transcribe():
        """Chạy ASR trên buffer hiện tại và gửi partial về client."""
        nonlocal recv_buffer
        if len(recv_buffer) == 0:
            return
        try:
            text = await run_transcribe_on_buffer(bytes(recv_buffer), lang_hint)
            recv_buffer = bytearray()  # reset buffer sau mỗi lần flush
            if text:
                await ws.send_text(json.dumps({"type": "partial", "text": text}))
        except Exception as e:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))

    try:
        while running:
            msg = await ws.receive()

            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    # Control message
                    try:
                        payload = json.loads(msg["text"])
                    except Exception:
                        # Bỏ qua text không hợp lệ
                        continue

                    event = payload.get("event")
                    if event == "start":
                        sample_rate = int(payload.get("sample_rate", 16000))
                        fmt = payload.get("format", "pcm16")
                        lang = payload.get("language")
                        if lang and lang.lower() != "auto":
                            lang_hint = lang
                        # OK
                        await ws.send_text(json.dumps({"type": "ready"}))

                    elif event == "stop":
                        # Flush phần còn lại để trả final
                        await flush_and_transcribe()
                        await ws.send_text(json.dumps({"type": "final", "text": ""}))
                        running = False

                    elif event == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))

                elif "bytes" in msg:
                    # Audio chunk
                    if fmt != "pcm16":
                        await ws.send_text(json.dumps({"type": "error", "message": "Unsupported audio format"}))
                        continue

                    chunk = msg["bytes"]
                    if not chunk:
                        continue
                    recv_buffer.extend(chunk)

                    # Nếu đủ ~1 giây, transcribe ngay để mượt
                    if len(recv_buffer) >= CHUNK_TARGET_BYTES:
                        await flush_and_transcribe()

            elif msg["type"] == "websocket.disconnect":
                break

        await ws.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
        except Exception:
            pass

# ---- Serve frontend ----
@app.get("/")
async def read_root():
    """Serve the frontend HTML file."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(frontend_path)

# ---- Run (dev) ----
# Chạy: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8917, reload=True)