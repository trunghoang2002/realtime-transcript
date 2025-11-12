import asyncio
import json
import logging
import os
import tempfile
import uuid
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
MODEL_NAME = "small"   # đổi thành "medium" hoặc "large-v3" nếu máy khỏe
DEVICE = "cuda"         # "cuda" nếu có GPU, "cpu" nếu không có GPU
COMPUTE_TYPE = "float16"  # "float16" trên GPU, "int8" hoặc "int8_float16" trên CPU

if DEVICE == "cuda":
    # Chạy setup trước khi import faster_whisper
    try:
        setup_cudnn_path()
    except Exception as e:
        pass  # Ignore nếu không setup được

    # Log LD_LIBRARY_PATH để debug
    if os.environ.get('LD_LIBRARY_PATH'):
        logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    else:
        logger.warning("LD_LIBRARY_PATH không được set. Nếu gặp lỗi cuDNN, hãy chạy: ./run_main_with_cuda.sh")

# -------- Whisper model ----------
# Model gợi ý: "small" (nhanh) / "medium" (chính xác) / "large-v3" (nặng)
from faster_whisper import WhisperModel
from silero_vad import VadOptions, get_speech_timestamps, collect_chunks
from get_audio import decode_audio

model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

# Patch SpeechBrain compatibility issue với huggingface_hub
import fix_speechbrain  # Phải import TRƯỚC speechbrain
from speechbrain.inference import EncoderClassifier
import torch

# Load speaker embedding model
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# Quản lý state của speaker cho mỗi session (tránh xung đột giữa các user)
# Key: session_id (str), Value: dict với 'registered_speakers' và 'speaker_counts'
speaker_sessions: dict[str, dict] = {}

speaker_threshold = 0.5  # cosine similarity ngưỡng nhận cùng người nói
update_alpha = 0.3  # hệ số cho exponential moving average khi update embedding
min_audio_length = 8000  # minimum 0.5 giây (16kHz) - chỉ pad khi quá ngắn

def get_speaker_id(audio_f32: np.ndarray, session_id: str = "default", debug=False, max_speakers=None):
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

    # Lấy hoặc tạo state cho session này
    if session_id not in speaker_sessions:
        speaker_sessions[session_id] = {
            "registered_speakers": [],
            "speaker_counts": []
        }
    
    registered_speakers = speaker_sessions[session_id]["registered_speakers"]
    speaker_counts = speaker_sessions[session_id]["speaker_counts"]

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
    current_threshold = 0.3 if (len(registered_speakers) == 1 and speaker_counts[0] == 1) else speaker_threshold

    # === Nếu giống speaker cũ ===
    if max_sim > current_threshold:
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

async def transcribe_file(
    file_path: str,
    lang_hint: Optional[str] = None,
    detect_speaker: bool = False,
    max_speakers: Optional[int] = 2,
) -> dict:
    """
    Transcribe file audio/video.
    Trả về dict với 'text' và 'segments' (có timestamp).
    """
    try:
        # faster-whisper có thể nhận file path trực tiếp
        audio = decode_audio(file_path)
        duration = audio.shape[0] / 16000
        print("duration: ", duration)

        segments_list = []
        info = None

        # VAD filter
        vad_options = VadOptions(min_silence_duration_ms=800)
        speech_chunks = get_speech_timestamps(audio, vad_options)
        # print("speech_chunks: ", speech_chunks)

        # Tạo session_id duy nhất cho file này
        file_session_id = f"file_{uuid.uuid4().hex[:8]}"
        
        # Reset state cho session này
        if file_session_id in speaker_sessions:
            speaker_sessions[file_session_id] = {
                "registered_speakers": [],
                "speaker_counts": []
            }

        normalized_max_speakers: Optional[int] = None
        if detect_speaker:
            try:
                normalized_max_speakers = max(1, int(max_speakers if max_speakers is not None else 2))
            except (TypeError, ValueError):
                normalized_max_speakers = 2

        formatted_lines = []

        for idx, chunk in enumerate(speech_chunks):
            print(f"chunk {idx}: start={chunk['start']} - {chunk['start'] / 16000}s, end={chunk['end']} - {chunk['end'] / 16000}s")
            
            audio_idx = collect_chunks(audio, [chunk])

            speaker_id = None
            if detect_speaker:
                speaker_id = get_speaker_id(
                    audio_idx,
                    session_id=file_session_id,
                    debug=True,
                    max_speakers=normalized_max_speakers,
                )
                print("speaker_id: ", speaker_id)
                print("--------------------------------")

            segments, info = model.transcribe(
                audio_idx,
                language=lang_hint,
                vad_filter=False,
                beam_size=5,  # Tăng độ chính xác cho file (không cần realtime)
                best_of=5,
                condition_on_previous_text=True,
                temperature=0.0,
                word_timestamps=True,
            )

            chunk_texts = []
            for seg in segments:
                # print("seg: ", seg)
                text = seg.text.strip()
                if text:
                    chunk_texts.append(text)
                    segment_entry = {
                        "start": round(chunk['start'] / 16000, 2),
                        "end": round(chunk['end'] / 16000, 2),
                        "text": text,
                    }
                    if speaker_id:
                        segment_entry["speaker_id"] = speaker_id
                    segments_list.append(segment_entry)

            if chunk_texts:
                chunk_line = " ".join(chunk_texts)
                if speaker_id:
                    formatted_lines.append(f"{speaker_id}: {chunk_line}")
                else:
                    formatted_lines.append(chunk_line)

        result = {
            "text": "\n".join(formatted_lines),
            "full_text": "\n".join(formatted_lines),  # Mỗi segment một dòng
            "segments": segments_list,
            "language": info.language if info else "auto",
            "language_probability": round(info.language_probability if info else 0, 3),
        }
        
        # Cleanup: xóa session sau khi hoàn thành để tránh memory leak
        if file_session_id in speaker_sessions:
            del speaker_sessions[file_session_id]
        
        return result
    except Exception as e:
        print(f"File transcription error: {e}")
        # Cleanup session ngay cả khi có lỗi
        if 'file_session_id' in locals() and file_session_id in speaker_sessions:
            del speaker_sessions[file_session_id]
        raise

# ---- API endpoint for file upload ----
@app.post("/api/transcribe")
async def transcribe_uploaded_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    detect_speaker: Optional[str] = Form("false"),
    max_speakers: Optional[str] = Form(None),
):
    """
    Upload file audio/video và transcribe.
    Hỗ trợ: .mp3, .wav, .m4a, .mp4, .avi, .mov, v.v.
    """
    start_time = datetime.now()
    
    if not file.filename:
        logger.warning(f"[API] POST /api/transcribe - No file provided")
        return JSONResponse(
            status_code=400,
            content={"error": "No file provided"}
        )

    logger.info(
        "[API] POST /api/transcribe - Received request: filename=%s, size=%s, language=%s, detect_speaker=%s, max_speakers=%s",
        file.filename,
        getattr(file, "size", "unknown"),
        language or "auto",
        detect_speaker,
        max_speakers if max_speakers is not None else "n/a",
    )

    # Lấy extension
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

    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_path = tmp_file.name
        try:
            # Lưu upload vào file tạm
            content = await file.read()
            file_size = len(content)
            tmp_file.write(content)
            tmp_file.flush()
            logger.info(f"[API] File saved to temp: {tmp_path}, size={file_size} bytes")

            # Transcribe
            logger.info(f"[API] Starting transcription for {file.filename}...")
            result = await transcribe_file(
                tmp_path,
                lang_hint,
                detect_speaker=detect_speaker_enabled,
                max_speakers=max_speakers_value,
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            segments_count = len(result.get("segments", []))
            text_length = len(result.get("text", ""))
            # Tính RTF
            audio_duration = decode_audio(tmp_path).shape[0] / 16000
            rtf = round(processing_time / audio_duration, 3) if audio_duration and audio_duration > 0 else None

            logger.info(
                "[API] POST /api/transcribe - Completed: filename=%s, processing_time=%.2fs, audio_duration=%.2fs, rtf=%s, segments=%s, text_length=%s",
                file.filename,
                processing_time,
                audio_duration,
                f"{rtf:.3f}" if isinstance(rtf, float) else "n/a",
                segments_count,
                text_length,
            )
            
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "rtf": rtf,
                **result
            })
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[API] POST /api/transcribe - Error: filename={file.filename}, error={str(e)}, processing_time={processing_time:.2f}s", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "success": False}
            )
        finally:
            # Xóa file tạm
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
    session_id: str = "default",
) -> dict:
    """
    Chạy ASR trên vùng đệm audio (PCM16 mono 16kHz).
    Trả về dict với 'text' và 'segments' (có timestamp) của đoạn ~1s để streaming mượt.
    """
    # buffer_duration = len(audio_buffer) / 2 / 16000
    # print("buffer_duration: ", buffer_duration)
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
        # for idx, chunk in enumerate(speech_chunks):
        #     print(f"chunk {idx}: start={chunk['start']} - {chunk['start'] / 16000}s, end={chunk['end']} - {chunk['end'] / 16000}s")
        
        # audio_f32 = collect_chunks(audio_f32, speech_chunks)

        speaker_id = None
        if detect_speaker:
            speaker_id = get_speaker_id(
                audio_f32,
                session_id=session_id,
                debug=True,
                max_speakers=normalized_max_speakers,
            )
            print("speaker_id: ", speaker_id)

        # faster-whisper nhận numpy audio array (1-D float32, sample_rate=16000)
        # Sử dụng vad_filter để bỏ im lặng; beam_size thấp để nhanh hơn.
        segments, info = model.transcribe(
            audio_f32,
            language=lang_hint,            # None -> auto-detect; "vi" -> tiếng Việt
            vad_filter=False,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,  # giúp các chunk độc lập, đỡ "lệch ngữ cảnh"
            repetition_penalty=1.2, # phạt các từ lặp lại
            no_repeat_ngram_size=3, # không lặp lại các cụm 3 từ liên tiếp
            temperature=0.0,
            word_timestamps=True,
        )
        # print("info: ", _)
        # print("time_offset: ", time_offset)

        partial_texts = []
        segments_list = []
        words_list = []
        for seg in segments:
            # print("seg: ", seg)
            # seg.text đã loại bỏ khoảng trắng đầu/cuối
            t = seg.text.strip()
            if t:
                partial_texts.append(t)
                segment_entry = {
                    "start": round(seg.start + time_offset, 2),
                    "end": round(seg.end + time_offset, 2),
                    "text": t,
                }
                if speaker_id:
                    segment_entry["speaker_id"] = speaker_id
                segments_list.append(segment_entry)

                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        w_start = round(w.start + time_offset, 2) if w.start is not None else None
                        w_end = round(w.end + time_offset, 2) if w.end is not None else None
                        words_list.append({
                            "start": w_start,
                            "end": w_end,
                            "word": w.word,
                            "prob": getattr(w, "probability", None),
                        })

        print("partial_texts: ", "||".join(partial_texts))
        return {
            "speaker_id": speaker_id,
            "language": info.language if info else "auto",
            "language_probability": round(info.language_probability if info else 0, 3),
            "text": "||".join(partial_texts),
            "segments": segments_list,
            "words": words_list,
        }
    except (ValueError, RuntimeError) as e:
        # Xử lý lỗi khi segments rỗng (ví dụ: max() arg is an empty sequence)
        # hoặc khi audio quá ngắn/im lặng hoàn toàn
        error_msg = str(e).lower()
        if "empty" in error_msg or "max()" in error_msg or "sequence" in error_msg:
            return {"text": "", "segments": []}
        # Log lỗi khác để debug
        print(f"Transcription error: {e}")
        raise

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
    client_ip = ws.client.host if ws.client else "unknown"
    session_start = datetime.now()
    logger.info(f"[WS] WebSocket connection request from {client_ip}")
    
    await ws.accept()
    logger.info(f"[WS] WebSocket connection accepted from {client_ip}")
    
    # Tạo session_id duy nhất cho WebSocket connection này
    ws_session_id = f"ws_{uuid.uuid4().hex[:8]}"
    
    sample_rate = 16000
    lang_hint = None
    fmt = "pcm16"
    detect_speaker_enabled = False
    max_speakers_limit: Optional[int] = None
    total_bytes_received = 0
    total_segments_sent = 0

    # buffer: gom ~1s audio rồi nhận dạng
    # 1s PCM16 mono 16kHz = 16000 samples * 2 bytes = 32000 bytes
    CHUNK_TARGET_BYTES = 32000
    MAX_WINDOW_BYTES = 32000 * 2   # 2 giây PCM16

    full_buffer = bytearray()
    recv_buffer = bytearray()
    running = True
    time_offset = 0.0  # Track thời gian tích lũy từ đầu session
    end_speech = False

    emitted_until = 0.0         # mốc thời gian đã phát đến đâu (global timeline)
    carry_over_words = []       # words treo ở mép phải chờ xác nhận ở cửa sổ sau
    last_emitted_word = None    # từ cuối cùng đã emit
    last_emitted_end = 0.0
    COMMIT_DELAY = 0.1         # 100ms
    EPS = 1e-3

    def _overlap(s, e, t0, t1):
        # true nếu [s, e] giao [t0, t1)
        if s is None or e is None:
            return False
        return not (e <= t0 or s >= t1)

    def _smart_join(words, lang_code):
        # Ghép từ thành câu: 
        # - tiếng Anh/Âu: join bằng space rồi chỉnh dấu câu
        # - CJK (ja/zh/ko): join không space (đơn giản)
        text = ""
        if lang_code and str(lang_code).lower().startswith(("ja", "zh", "ko")):
            text = "".join(w["word"] for w in words)
            text = re.sub(r"\s+", "", text)
            # Loại khoảng trắng thừa trước dấu câu tiếng Nhật/Trung nếu có
            return text.strip()
        else:
            raw = " ".join(w["word"] for w in words)
            # Fix space trước dấu câu phổ biến
            import re
            raw = re.sub(r"\s+([,.;:!?])", r"\1", raw)
            raw = re.sub(r"\(\s+", "(", raw)
            raw = re.sub(r"\s+\)", ")", raw)
            raw = re.sub(r"\s+", " ", raw)
            return raw.strip()

    def char_overlap_ratio(a: str, b: str) -> float:
        a = a.strip().lower()
        b = b.strip().lower()
        if not a or not b:
            return 0.0

        overlap_a_in_b = sum(1 for c in a if c in b)
        overlap_a_in_b_ratio = overlap_a_in_b / len(a)  # % ký tự của A xuất hiện trong B

        overlap_b_in_a = sum(1 for c in b if c in a)
        overlap_b_in_a_ratio = overlap_b_in_a / len(b)  # % ký tự của B xuất hiện trong A

        return max(overlap_a_in_b_ratio, overlap_b_in_a_ratio)

    def _dedupe_boundary(words, emitted_until, last_emitted_word, last_emitted_end, MIN_GAP=0.1, EPS=1e-3):
        """Loại bỏ từ đã phát (theo thời gian) + trùng mép (theo nội dung)."""
        out = []
        for w in words:
            s, e, t = w["start"], w["end"], w["word"]
            if e is None or s is None:
                continue
            # 1) Bỏ từ có end <= emitted_until (+EPS)
            if e <= emitted_until + EPS:
                continue
            # 2) Nếu quá gần từ cuối đã emit → bỏ (trùng ~ chắc chắn)
            if last_emitted_word is not None:
                if (s - last_emitted_end) < MIN_GAP and char_overlap_ratio(t, last_emitted_word) > 0.8:
                    continue
            out.append(w)
        return out

    def _dedupe_commit_now(words, overlap_eps=0.01):
        """
        Loại bỏ trùng lặp ở commit_now:
        - Giữ lại từ có prob cao hơn khi start/end overlap.
        - Loại bỏ ký tự nhiễu như '-' nếu bị overlap và prob thấp.
        """

        if not words:
            return []

        # Sort theo start time
        words = sorted(words, key=lambda w: (w["start"] or 0.0, w["end"] or 0.0))

        out = []

        for w in words:
            if not out:
                if w and w not in ["-", ".", "...", "—", "_"]:
                    out.append(w)
                    continue

            last = out[-1]

            s, e = w["start"], w["end"]
            ls, le = last["start"], last["end"]

            # ---- (1) Kiểm tra overlap thời gian ----
            overlap = not (e <= ls + overlap_eps or s >= le - overlap_eps)

            # ---- (2) Normalize từ để so sánh chữ ----
            w_norm = w["word"].strip().lower()
            last_norm = last["word"].strip().lower()

            # ----- Case A: cùng chữ và overlap → giữ từ prob cao -----
            if w_norm == last_norm and overlap:
                if w.get("prob", 0) > last.get("prob", 0):
                    out[-1] = w  # replace
                continue

            # ----- Case B: từ nhiễu ('-', '.', ...) → bỏ nếu overlap -----
            if w_norm in ["-", ".", "...", "—", "_"] and overlap:
                continue

            # ----- Case C: các chữ khác nhau nhưng overlapped mạnh -----
            # Giữ từ có prob cao hơn
            if overlap and w.get("prob", 0) < last.get("prob", 0):
                continue

            out.append(w)

        return out

    async def flush_and_transcribe():
        """Chạy ASR trên buffer hiện tại và gửi partial về client."""
        nonlocal full_buffer, recv_buffer, time_offset, total_segments_sent, end_speech, emitted_until, carry_over_words, last_emitted_word, last_emitted_end
        if len(recv_buffer) == 0:
            return
        try:
            buffer_size = len(recv_buffer)
            buffer_duration = buffer_size / 2 / sample_rate
            # print("buffer_duration: ", buffer_duration)
            print("time_offset: ", time_offset)
            # 1 giây mới vừa nhận
            new_chunk = bytes(recv_buffer)
            recv_buffer = bytearray()
            # Append vào buffer lớn
            full_buffer.extend(new_chunk)
            # Nếu full_buffer > 2s thì chỉ giữ lại cuối cùng 2 giây
            if len(full_buffer) > MAX_WINDOW_BYTES:
                full_buffer = full_buffer[-MAX_WINDOW_BYTES:]
            # Nếu chưa đủ 2 giây, thì chưa transcribe (đợi thêm)
            if len(full_buffer) < MAX_WINDOW_BYTES:
                time_offset += buffer_duration
                return
            # ---- Sliding Window ----
            window_bytes = bytes(full_buffer)  # đúng 2 giây audio
            # Thời gian thật của cửa sổ 2s
            window_duration = len(window_bytes) / 2 / sample_rate
            # print("window_duration: ", window_duration)
            # Run ASR
            transcribe_start = datetime.now()
            
            result = await run_transcribe_on_buffer(
                window_bytes,
                lang_hint,
                time_offset - 1.0,   # timestamp cửa sổ bắt đầu = 1 giây trước
                detect_speaker=detect_speaker_enabled,
                max_speakers=max_speakers_limit if detect_speaker_enabled else None,
                session_id=ws_session_id,
            )
            transcribe_time = (datetime.now() - transcribe_start).total_seconds()
            recv_buffer = bytearray()  # reset buffer sau mỗi lần flush
            
            # Text tương ứng với phần 1 giây MỚI (cuối cửa sổ)
            if result["text"]:
                if result["text"] == "silence":
                    end_speech = True
                    print("end_speech: ", end_speech)
                else:
                    segments_count = len(result.get("segments", []))
                    total_segments_sent += segments_count
                    speaker_label = result.get("speaker_id")
                    print("result['segments']: ", result["segments"])
                    print("---")
                    print("result['words']: ", result["words"])
                    print("---")

                    t0 = time_offset
                    t1 = time_offset + 1.0
                    lang_code = result.get("language")

                    # Gộp carry_over từ cửa sổ trước (chúng đang treo do sát mép)
                    words = (carry_over_words or []) + (result.get("words", []) or [])
                    # Lọc các từ thuộc 1 giây mới (cho overlap để không cắt từ ở biên)
                    if last_emitted_word:
                        new_words = [w for w in words if _overlap(w["start"], w["end"], t0 - 0.1, t1 + 0.1)]
                    else:
                        new_words = words
                    print("new_words: ", new_words)
                    print("---")

                    # Chia 2 nhóm: commit ngay vs treo lại (sát mép phải)
                    commit_now = []
                    next_carry = []  # carry cho vòng sau
                    for w in new_words:
                        if w["end"] is not None and w["end"] > (t1 - COMMIT_DELAY):
                            next_carry.append(w)    # treo lại → đợi cửa sổ sau xác nhận
                        else:
                            commit_now.append(w)    # an toàn để emit
                    
                    print("commit_now: ", commit_now)
                    print("---")
                    print("next_carry: ", next_carry)
                    print("---")

                    # Dedupe theo emitted_until + trùng chữ ở mép
                    commit_now = _dedupe_boundary(commit_now, emitted_until, last_emitted_word, last_emitted_end)

                    print("commit_now after dedupe: ", commit_now)
                    print("---")

                    commit_now = _dedupe_commit_now(commit_now)
                    print("commit_now after dedupe_commit_now: ", commit_now)
                    print("---")

                    # Emit
                    if commit_now:
                        base_text = _smart_join(commit_now, lang_code)
                        print("base_text: ", base_text)

                        if base_text:
                            if end_speech:
                                if speaker_label:
                                    result_text = f"/newline{speaker_label}: {base_text}"
                                else:
                                    result_text = f"/newline{base_text}"
                            else:
                                result_text = base_text
                            
                            if result.get("language") != "ja":
                                result_text = result_text.replace("||", " ")
                            else:
                                result_text = result_text.replace("||", "")
                            print("result_text: ", result_text)
                            print("---")
                            
                            await ws.send_text(json.dumps({
                                "type": "partial",
                                "speaker_id": speaker_label,
                                "language": result.get("language"),
                                "language_probability": result.get("language_probability"),
                                "text": result_text,
                                "words": commit_now,
                                "segments": result.get("segments", []),
                                "window": {"t0": t0, "t1": t1}
                            }))
                            logger.debug(f"[WS] Sent partial: segments={segments_count}, text_length={len(result['text'])}, transcribe_time={transcribe_time:.3f}s, buffer_size={buffer_size} bytes")
                            # Cập nhật state dedupe
                            # mốc đã phát: max end trong commit_now
                            emitted_until = max(emitted_until, max(w["end"] for w in commit_now if w["end"] is not None))
                            # nhớ từ cuối
                            last = commit_now[-1]
                            last_emitted_word = last["word"]
                            last_emitted_end = last["end"] or last_emitted_end
                            # Cập nhật carry_over cho vòng sau (chỉ giữ phần "chưa commit")
                            carry_over_words = next_carry
                            print("emitted_until: ", emitted_until)
                            print("carry_over_words: ", carry_over_words)
                            print("last_emitted_word: ", last_emitted_word)
                            print("last_emitted_end: ", last_emitted_end)
                            end_speech = False

            # Cập nhật time_offset cho chunk tiếp theo
            time_offset += buffer_duration
            print("--------------------------------")
            print("--------------------------------")
            print("--------------------------------")
        except Exception as e:
            logger.error(f"[WS] Transcription error: {str(e)}", exc_info=True)
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
                        # Reset speaker tracking for new session
                        if ws_session_id in speaker_sessions:
                            speaker_sessions[ws_session_id] = {
                                "registered_speakers": [],
                                "speaker_counts": []
                            }
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
                        # OK
                        await ws.send_text(json.dumps({"type": "ready"}))
                        logger.info(f"[WS] Ready message sent to {client_ip}")

                    elif event == "stop":
                        logger.info(f"[WS] Stop event received from {client_ip}")
                        # Flush phần còn lại để trả final
                        await flush_and_transcribe()
                        await ws.send_text(json.dumps({"type": "final", "text": ""}))
                        
                        session_duration = (datetime.now() - session_start).total_seconds()
                        logger.info(f"[WS] Session completed: client={client_ip}, duration={session_duration:.2f}s, total_bytes={total_bytes_received}, total_segments={total_segments_sent}")
                        
                        running = False

                    elif event == "ping":
                        logger.debug(f"[WS] Ping received from {client_ip}")
                        await ws.send_text(json.dumps({"type": "pong"}))

                elif "bytes" in msg:
                    # Audio chunk
                    if fmt != "pcm16":
                        logger.warning(f"[WS] Unsupported audio format: {fmt} from {client_ip}")
                        await ws.send_text(json.dumps({"type": "error", "message": "Unsupported audio format"}))
                        continue

                    chunk = msg["bytes"]
                    if not chunk:
                        continue
                    chunk_size = len(chunk)
                    total_bytes_received += chunk_size
                    recv_buffer.extend(chunk)
                    logger.debug(f"[WS] Received audio chunk: size={chunk_size} bytes, buffer_size={len(recv_buffer)}, total_received={total_bytes_received}")

                    # Nếu đủ ~1 giây, transcribe ngay để mượt
                    if len(recv_buffer) >= CHUNK_TARGET_BYTES:
                        await flush_and_transcribe()

            elif msg["type"] == "websocket.disconnect":
                logger.info(f"[WS] WebSocket disconnect from {client_ip}")
                break

        await ws.close()
        logger.info(f"[WS] WebSocket connection closed for {client_ip}")
        
        # Cleanup: xóa session để tránh memory leak
        if ws_session_id in speaker_sessions:
            del speaker_sessions[ws_session_id]
            logger.debug(f"[WS] Cleaned up session: {ws_session_id}")

    except WebSocketDisconnect:
        session_duration = (datetime.now() - session_start).total_seconds() if 'session_start' in locals() else 0
        logger.info(f"[WS] WebSocket disconnected: client={client_ip}, duration={session_duration:.2f}s")
        # Cleanup session
        if 'ws_session_id' in locals() and ws_session_id in speaker_sessions:
            del speaker_sessions[ws_session_id]
            logger.debug(f"[WS] Cleaned up session: {ws_session_id}")
    except Exception as e:
        session_duration = (datetime.now() - session_start).total_seconds() if 'session_start' in locals() else 0
        logger.error(f"[WS] WebSocket error: client={client_ip}, error={str(e)}, duration={session_duration:.2f}s", exc_info=True)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
        except Exception:
            pass
        finally:
            # Cleanup session
            if 'ws_session_id' in locals() and ws_session_id in speaker_sessions:
                del speaker_sessions[ws_session_id]
                logger.debug(f"[WS] Cleaned up session: {ws_session_id}")

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