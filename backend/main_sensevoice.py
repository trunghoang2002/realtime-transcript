import asyncio
import json
import tempfile
from typing import Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# -------- SenseVoice model ----------
from funasr import AutoModel

MODEL_NAME = "iic/SenseVoiceSmall"
DEVICE = "cpu"  # "cuda" nếu có GPU, "cpu" nếu không có GPU

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


async def run_transcribe_on_buffer(audio_buffer: bytes, lang_hint: Optional[str] = None) -> str:
    """
    Streaming transcript (SenseVoice).
    Trả về text ngắn của đoạn ~1s audio.
    """
    audio_f32 = pcm16_to_float32(audio_buffer)
    if audio_f32.size < 8000:
        return ""

    try:
        # SenseVoice: model dùng sample_rate 16000, input là numpy.float32
        res = model.generate(
            input=audio_f32,
            cache={},
            language=lang_hint or "auto",
            use_itn=False,         # không chuẩn hóa số thành chữ
            batch_size_s=10,
        )

        # Trả text (res là list các dict)
        texts = [r["text"] for r in res if "text" in r]
        return " ".join(texts).strip()
    except Exception as e:
        print(f"Transcribe error: {e}")
        return ""


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
                        "text": r["text"]
                    })
        return {"text": " ".join(texts), "segments": segments}
    except Exception as e:
        raise RuntimeError(f"File transcription error: {e}")


# ---- API endpoint for file upload ----
@app.post("/api/transcribe")
async def transcribe_uploaded_file(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file provided"})

    ext = Path(file.filename).suffix.lower()
    lang_hint = language if language and language.lower() != "auto" else None

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_path = tmp_file.name
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            result = await transcribe_file(tmp_path, lang_hint)
            return JSONResponse(content={"success": True, "filename": file.filename, **result})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e), "success": False})
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---- WebSocket endpoint ----
@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    sample_rate = 16000
    lang_hint = None
    fmt = "pcm16"

    CHUNK_TARGET_BYTES = 32000
    recv_buffer = bytearray()
    running = True

    async def flush_and_transcribe():
        nonlocal recv_buffer
        if len(recv_buffer) == 0:
            return
        text = await run_transcribe_on_buffer(bytes(recv_buffer), lang_hint)
        recv_buffer = bytearray()
        if text:
            await ws.send_text(json.dumps({"type": "partial", "text": text}))

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
                        await ws.send_text(json.dumps({"type": "ready"}))
                    elif event == "stop":
                        await flush_and_transcribe()
                        await ws.send_text(json.dumps({"type": "final", "text": ""}))
                        running = False
                elif "bytes" in msg:
                    chunk = msg["bytes"]
                    if not chunk:
                        continue
                    recv_buffer.extend(chunk)
                    if len(recv_buffer) >= CHUNK_TARGET_BYTES:
                        await flush_and_transcribe()
            elif msg["type"] == "websocket.disconnect":
                break
        await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        await ws.close()


@app.get("/")
async def read_root():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(frontend_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8917, reload=True)
