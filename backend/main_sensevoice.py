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
import librosa
from collections import Counter

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
from silero_vad import VadOptions, get_speech_timestamps, collect_chunks

model = AutoModel(
    model=MODEL_NAME,
    vad_model="fsmn-vad", # bật VAD tích hợp
    vad_kwargs={"max_single_segment_time": 30000},
    device=DEVICE,
    # disable_update=True
)

# Patch SpeechBrain compatibility issue với huggingface_hub
import fix_speechbrain  # Phải import TRƯỚC speechbrain
from speechbrain.inference import EncoderClassifier
import torch

# Load speaker embedding model
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
registered_speakers = []  # lưu danh sách các vector trung bình của speaker
speaker_counts = []  # số lần đã thấy speaker này (để tính moving average)
speaker_threshold = 0.6  # cosine similarity ngưỡng nhận cùng người nói
update_alpha = 0.3  # hệ số cho exponential moving average khi update embedding
min_audio_length = 8000  # minimum 0.5 giây (16kHz) - chỉ pad khi quá ngắn

def get_speaker_id(audio_f32: np.ndarray, debug=False, max_speakers=None):
    """Nhận diện người nói bằng cosine similarity với cải thiện stability."""

    # === Reject nếu audio rỗng ===
    if audio_f32 is None or len(audio_f32) == 0:
        return "unknown"

    # Chỉ pad khi audio quá ngắn (< 0.5s), lặp lại audio thay vì padding zeros
    if len(audio_f32) < min_audio_length:
        pad_length = min_audio_length - len(audio_f32)
        pad_start = pad_length // 2
        pad_end = pad_length - pad_start
        
        # Clamp lại để không vượt quá độ dài audio
        pad_start = min(pad_start, len(audio_f32))
        pad_end = min(pad_end, len(audio_f32))

        # Lặp lại audio thay vì padding zeros - tốt hơn cho speaker recognition
        # Lặp lại phần cuối ở đầu, và phần đầu ở cuối (symmetric repetition)
        if len(audio_f32) > 0:
            # Lặp lại phần cuối audio ở đầu
            repeat_start = audio_f32[-pad_start:] if pad_start > 0 else np.array([], dtype=audio_f32.dtype)
            # Lặp lại phần đầu audio ở cuối
            repeat_end = audio_f32[:pad_end] if pad_end > 0 else np.array([], dtype=audio_f32.dtype)
            
            audio_f32 = np.concatenate([
                repeat_start,
                audio_f32,
                repeat_end
            ])
    
    # === Trích embedding ===
    # SpeechBrain yêu cầu tensor shape (batch, time)
    try:
        tensor = torch.tensor(audio_f32).unsqueeze(0)
        with torch.no_grad():
            emb = spk_model.encode_batch(tensor).detach().cpu().numpy().mean(axis=1)[0]
    except Exception:
        return "unknown"
    
    # Normalize embedding để ổn định hơn
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

    # Nếu chưa có người nói nào, tạo mới
    if not registered_speakers:
        registered_speakers.append(emb_norm)
        speaker_counts.append(1)
        return "spk_01"

    # So khớp cosine với danh sách speaker đã biết (đã normalize nên chỉ cần dot product)
    sims = [np.dot(emb_norm, s) for s in registered_speakers]
    max_sim = max(sims)
    idx = np.argmax(sims)
    
    if debug:
        print(f"  Similarities: {[f'{s:.3f}' for s in sims]}, max={max_sim:.3f}")

    # === Nếu mới chỉ có 1 speaker duy nhất và speaker_count = 1 thì giảm ngưỡng ===
    if len(registered_speakers) == 1 and speaker_counts[0] == 1:
        speaker_threshold = 0.3
    else:
        speaker_threshold = 0.5

    # === Nếu giống speaker cũ ===
    if max_sim > speaker_threshold:
        # Update speaker embedding với exponential moving average để ổn định hơn
        registered_speakers[idx] = (1 - update_alpha) * registered_speakers[idx] + update_alpha * emb_norm
        speaker_counts[idx] += 1
        return f"spk_{idx+1:02d}"
    
    # === Nếu vượt quá số speaker cho phép → gán speaker gần nhất
    if max_speakers is not None and len(registered_speakers) >= max_speakers:
        # Update speaker embedding với exponential moving average để ổn định hơn
        registered_speakers[idx] = (1 - update_alpha) * registered_speakers[idx] + update_alpha * emb_norm
        speaker_counts[idx] += 1
        return f"spk_{idx+1:02d}"
    
    # === Ngược lại → tạo speaker mới ===
    registered_speakers.append(emb_norm)
    speaker_counts.append(1)
    return f"spk_{len(registered_speakers):02d}"

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

VALID_LANGS = {"ja", "yue", "en", "zh", "ko"}

def extract_language_codes(text):
    tokens = re.findall(r"<\|([a-zA-Z]+)\|>", text)
    return [t for t in tokens if t in VALID_LANGS]

def get_dominant_language(lang_list):
    if not lang_list:
        return None, 0.0

    counter = Counter(lang_list)
    lang, count = counter.most_common(1)[0]
    probability = count / len(lang_list)

    return lang, round(probability, 2)

async def transcribe_file(
    file_path: str,
    lang_hint: Optional[str] = None,
    detect_speaker: bool = False,
    max_speakers: Optional[int] = 2,
) -> dict:
    """
    Transcribe file audio/video (offline).
    """
    # Reset registered_speakers và speaker_counts
    global registered_speakers
    registered_speakers = []
    global speaker_counts
    speaker_counts = []
    normalized_max_speakers: Optional[int] = None
    if detect_speaker:
        try:
            normalized_max_speakers = max(1, int(max_speakers if max_speakers is not None else 2))
        except (TypeError, ValueError):
            normalized_max_speakers = 2

    try:
        res = model.generate(
            input=file_path,
            language=lang_hint or "auto",
            use_itn=False,
            batch_size_s=20,
            # merge_vad=True,  # merge các khoảng âm thanh dựa trên VAD
            # merge_length_s=15,
        )

        # SenseVoice trả list kết quả, có timestamp
        texts, segments, languages = [], [], []
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
                    segments = [{"start": 0.0, "end": round(duration, 2), "text": "||".join(texts).strip()}]
                else:
                    start = 0.0
                    new_segments = []
                    for idx, text in enumerate(texts):
                        languages.extend(extract_language_codes(text))
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
                segments = [{"start": 0.0, "end": 0.0, "text": "||".join(texts).strip()}]

        waveform = None
        if detect_speaker:
            waveform, _ = librosa.load(file_path, sr=16000, mono=True)

        for seg in segments:
            if detect_speaker and waveform is not None:
                start_idx = max(0, int(round(seg["start"] * 16000)))
                end_idx = max(0, int(round(seg["end"] * 16000)))
                start_idx = min(start_idx, len(waveform))
                end_idx = max(end_idx, start_idx + 1600)  # ~0.1s fallback để lấy đủ mẫu
                end_idx = min(len(waveform), end_idx)
                if end_idx <= start_idx:
                    end_idx = min(len(waveform), start_idx + 1600)
                chunk = collect_chunks(
                    waveform,
                    [{"start": start_idx, "end": min(len(waveform), end_idx)}],
                )
                seg["speaker_id"] = get_speaker_id(
                    chunk,
                    debug=True,
                    max_speakers=normalized_max_speakers,
                )
            else:
                seg.pop("speaker_id", None)
        
        print(f"languages: {languages}")
        if lang_hint and lang_hint.lower() != "auto":
            dominant_lang = lang_hint
            language_probability = 1.0
        else:
            dominant_lang, language_probability = get_dominant_language(languages)

        def format_segment_line(segment: dict) -> str:
            text = segment.get("text", "")
            speaker_id = segment.get("speaker_id")
            return f"{speaker_id}: {text}" if speaker_id else text

        formatted_lines = [format_segment_line(seg) for seg in segments]

        return {
            "text": "\n".join(formatted_lines),
            "full_text": "\n".join(formatted_lines),
            "segments": segments,
            "language": dominant_lang,
            "language_probability": round(language_probability, 2),
        }
    except Exception as e:
        raise RuntimeError(f"File transcription error: {e}")


# ---- API endpoint for file upload ----
@app.post("/api/transcribe")
async def transcribe_uploaded_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    detect_speaker: Optional[str] = Form("false"),
    max_speakers: Optional[str] = Form(None),
):
    start_time = datetime.now()
    
    if not file.filename:
        logger.warning(f"[API] POST /api/transcribe - No file provided")
        return JSONResponse(status_code=400, content={"error": "No file provided"})

    logger.info(
        "[API] POST /api/transcribe - Received request: filename=%s, size=%s, language=%s, detect_speaker=%s, max_speakers=%s",
        file.filename,
        getattr(file, "size", "unknown"),
        language or "auto",
        detect_speaker,
        max_speakers if max_speakers is not None else "n/a",
    )

    ext = Path(file.filename).suffix.lower()
    lang_hint = language if language and language.lower() != "auto" else None
    detect_speaker_enabled = str(detect_speaker).lower() in {"true", "1", "yes", "on"}
    max_speakers_value: Optional[int] = None
    if detect_speaker_enabled:
        try:
            max_speakers_value = int(max_speakers) if max_speakers is not None else 2
        except (TypeError, ValueError):
            max_speakers_value = 2
        if max_speakers_value < 1:
            max_speakers_value = 1

    # if lang_hint != "auto" and lang_hint not in VALID_LANGS:
    #     return JSONResponse(status_code=400, content={"error": "Invalid language code. Valid languages are: auto, " + ", ".join(VALID_LANGS) + "."})

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_path = tmp_file.name
        try:
            content = await file.read()
            file_size = len(content)
            tmp_file.write(content)
            tmp_file.flush()
            logger.info(f"[API] File saved to temp: {tmp_path}, size={file_size} bytes")

            logger.info(f"[API] Starting transcription for {file.filename}...")
            result = await transcribe_file(
                tmp_path,
                lang_hint,
                detect_speaker=detect_speaker_enabled,
                max_speakers=max_speakers_value,
            )
            print("result.text: ", result.get("text", ""))
            
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

async def run_transcribe_on_buffer(
    audio_buffer: bytes,
    lang_hint: Optional[str] = None,
    time_offset: float = 0.0,
    detect_speaker: bool = False,
    max_speakers: Optional[int] = 2,
) -> dict:
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
        return {"text": "silence", "segments": []}

    normalized_max_speakers: Optional[int] = None
    if detect_speaker:
        try:
            normalized_max_speakers = max(1, int(max_speakers if max_speakers is not None else 2))
        except (TypeError, ValueError):
            normalized_max_speakers = 2

    try:
        # VAD filter
        # Kiểm tra nếu audio có khoảng im lặng > 750ms trong 1s thì trả về "silence"
        vad_options = VadOptions(min_silence_duration_ms=750)
        speech_chunks = get_speech_timestamps(audio_f32, vad_options)
        if len(speech_chunks) == 0:
            return {"text": "silence", "segments": []}

        # audio_f32 = collect_chunks(audio_f32, speech_chunks)

        speaker_id = None
        if detect_speaker:
            speech_audio = collect_chunks(audio_f32, speech_chunks)
            speaker_id = get_speaker_id(
                speech_audio,
                debug=True,
                max_speakers=normalized_max_speakers,
            )
            print("speaker_id: ", speaker_id)

        # SenseVoice: model dùng sample_rate 16000, input là numpy.float32
        res = model.generate(
            input=audio_f32,
            cache={},
            language=lang_hint or "auto",
            use_itn=False,         # không chuẩn hóa số thành chữ
            batch_size_s=10,
        )

        # SenseVoice trả list các dict, có thể có timestamp
        partial_texts = []
        segments_list = []
        languages = []
        for r in res:
            if "text" in r and r["text"]:
                text = r["text"].strip()
                if text:
                    languages.extend(extract_language_codes(text))
                    partial_texts.append(text)
                    # Nếu có timestamp, dùng nó; nếu không, ước tính từ buffer
                    if "timestamp" in r and len(r["timestamp"]) >= 2:
                        segment_payload = {
                            "start": round(r["timestamp"][0] + time_offset, 2),
                            "end": round(r["timestamp"][1] + time_offset, 2),
                            "text": re.sub(r"<\|[^|>]+\|>", "", text).strip(),
                        }
                    else:
                        # Ước tính timestamp từ buffer size
                        buffer_duration = len(audio_buffer) / 2 / 16000
                        segment_payload = {
                            "start": round(time_offset, 2),
                            "end": round(time_offset + buffer_duration, 2),
                            "text": re.sub(r"<\|[^|>]+\|>", "", text).strip(),
                        }
                    if speaker_id:
                        segment_payload["speaker_id"] = speaker_id
                    segments_list.append(segment_payload)

        print(f"languages: {languages}")
        if lang_hint and lang_hint.lower() != "auto":
            dominant_lang = lang_hint
            language_probability = 1.0
        else:
            dominant_lang, language_probability = get_dominant_language(languages)
        
        print("partial_texts: ", "||".join(partial_texts))
        return {
            "text": re.sub(r"<\|[^|>]+\|>", "", "||".join(partial_texts)).strip(),
            "speaker_id": speaker_id,
            "language": dominant_lang,
            "language_probability": round(language_probability, 2),
            "segments": segments_list,
        }
    except Exception as e:
        print(f"Transcribe error: {e}")
        return {"text": "", "segments": []}

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
    detect_speaker_enabled = False
    max_speakers_limit: Optional[int] = None
    total_bytes_received = 0
    total_segments_sent = 0

    CHUNK_TARGET_BYTES = 32000
    recv_buffer = bytearray()
    running = True
    time_offset = 0.0  # Track thời gian tích lũy từ đầu session
    end_speech = False

    async def flush_and_transcribe():
        nonlocal recv_buffer, time_offset, total_segments_sent, end_speech
        if len(recv_buffer) == 0:
            return
        try:
            buffer_size = len(recv_buffer)
            transcribe_start = datetime.now()
            # Tính thời gian của buffer này (bytes / 2 / sample_rate = seconds)
            buffer_duration = buffer_size / 2 / sample_rate
            result = await run_transcribe_on_buffer(
                bytes(recv_buffer),
                lang_hint,
                time_offset,
                detect_speaker=detect_speaker_enabled,
                max_speakers=max_speakers_limit if detect_speaker_enabled else None,
            )
            transcribe_time = (datetime.now() - transcribe_start).total_seconds()
            recv_buffer = bytearray()
            if result["text"]:
                if result["text"] == "silence":
                    end_speech = True
                    print("end_speech: ", end_speech)
                else:
                    segments_count = len(result.get("segments", []))
                    total_segments_sent += segments_count
                    speaker_label = result.get("speaker_id")
                    base_text = result.get("text", "")
                    if end_speech:
                        if speaker_label:
                            result_text = f"/newline{speaker_label}: {base_text}"
                        else:
                            result_text = f"/newline{base_text}"
                    else:
                        result_text = base_text
                    if result["language"] != "ja":
                        result_text = result_text.replace("||", " ")
                    else:
                        result_text = result_text.replace("||", "")
                    print("result_text: ", result_text)
                    await ws.send_text(json.dumps({
                        "type": "partial",
                        "speaker_id": speaker_label,
                        "language": result["language"],
                        "language_probability": result["language_probability"],
                        "text": result_text,
                        "segments": result["segments"]
                    }))

                    logger.debug(f"[WS] Sent partial: segments={segments_count}, text_length={len(result['text'])}, transcribe_time={transcribe_time:.3f}s, buffer_size={buffer_size} bytes")
                    end_speech = False
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
                        else:
                            lang_hint = None
                        detect_speaker_enabled = bool(payload.get("detect_speaker"))
                        if detect_speaker_enabled:
                            max_payload = payload.get("max_speakers")
                            try:
                                max_speakers_limit = max(1, int(max_payload)) if max_payload is not None else 2
                            except (TypeError, ValueError):
                                max_speakers_limit = 2
                        else:
                            max_speakers_limit = None
                        # Reset time offset khi bắt đầu session mới
                        time_offset = 0.0
                        recv_buffer = bytearray()
                        total_bytes_received = 0
                        total_segments_sent = 0
                        session_start = datetime.now()
                        logger.info(
                            "[WS] Start event received: sample_rate=%s, format=%s, language=%s, detect_speaker=%s, max_speakers=%s, client=%s",
                            sample_rate,
                            fmt,
                            lang_hint or "auto",
                            detect_speaker_enabled,
                            max_speakers_limit if max_speakers_limit is not None else "n/a",
                            client_ip,
                        )
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
