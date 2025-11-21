import asyncio
import json
import logging
import os
import tempfile
from typing import Optional
from datetime import datetime
import uuid

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
MODEL_NAME = "small"   # "small" , "medium" , "large-v3"
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

# -------- Speaker Diarization model ----------
# Preload speaker embedding model một lần để tránh load lại mỗi session
from fusion_diarization import RealtimeSpeakerDiarization
import torch

diarization_model = RealtimeSpeakerDiarization(
    fusion_method="score_level",
    fusion_alpha=0.4,
    pyannote_config={
        "model_name": "pyannote/speaker-diarization-community-1",
        "token": os.getenv("HF_TOKEN"),
    },
    speechbrain_config={
        "model_name": "speechbrain/spkrec-ecapa-tdnn-voxceleb",
    },
    similarity_threshold=0.6,
    embedding_update_weight=0.3,
    min_similarity_gap=0.25
)

def pad_audio_for_embedding(audio_f32: np.ndarray, min_audio_length = 8000) -> np.ndarray:
    if audio_f32 is None:
        return np.array([], dtype=np.float32)
    if len(audio_f32) >= min_audio_length:
        return audio_f32

    pad_length = min_audio_length - len(audio_f32)
    pad_start = pad_length // 2
    pad_end = pad_length - pad_start

    pad_start = min(pad_start, len(audio_f32))
    pad_end = min(pad_end, len(audio_f32))

    if len(audio_f32) == 0:
        return np.zeros(min_audio_length, dtype=np.float32)

    repeat_start = audio_f32[-pad_start:] if pad_start > 0 else np.array([], dtype=audio_f32.dtype)
    repeat_end = audio_f32[:pad_end] if pad_end > 0 else np.array([], dtype=audio_f32.dtype)

    return np.concatenate([repeat_start, audio_f32, repeat_end])

def get_speaker_id(audio_f32: np.ndarray, session_id: str = "default", max_speakers=None):
    # Lấy hoặc tạo state cho session này
    diarization_model.set_session(session_id)
    return diarization_model(pad_audio_for_embedding(audio_f32), max_speakers=max_speakers, use_memory=True)['speaker_labels'][0]

def has_repeated_substring(s: str, min_repeat: int = 5, min_len: int = 1) -> bool:
    """
    Phát hiện xem trong chuỗi s có tồn tại substring (độ dài >= min_len)
    bị lặp lại liên tiếp ít nhất min_repeat lần hay không.
    """
    n = len(s)
    for length in range(min_len, n // min_repeat + 1):  # độ dài substring
        for i in range(0, n - length * min_repeat + 1):
            sub = s[i:i+length]
            count = 1
            # kiểm tra liên tiếp
            while i + count * length + length <= n and s[i + count * length:i + (count + 1) * length] == sub:
                count += 1
            if count >= min_repeat:
                logger.warning(f"Repeated substring: {sub} in {s}")
                return True
    return False

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
        vad_options = VadOptions(min_silence_duration_ms=1000)
        speech_chunks = get_speech_timestamps(audio, vad_options)
        # print("speech_chunks: ", speech_chunks)

        # Tạo session_id duy nhất cho file này
        file_session_id = f"file_{uuid.uuid4().hex[:8]}"

        # Reset state cho session này
        diarization_model.reset_session(file_session_id)

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
        
        # Cleanup: xóa session sau khi hoàn thành để tránh memory leak
        diarization_model.delete_session(file_session_id)

        return {
            "text": "\n".join(formatted_lines),
            "full_text": "\n".join(formatted_lines),  # Mỗi segment một dòng
            "segments": segments_list,
            "language": info.language if info else "auto",
            "language_probability": round(info.language_probability if info else 0, 3),
        }
    except Exception as e:
        print(f"File transcription error: {e}")

        # Cleanup session ngay cả khi có lỗi
        if 'file_session_id' in locals():
            diarization_model.delete_session(file_session_id)
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

async def run_transcribe_on_full_buffer(
    audio_buffer: bytes,
    lang_hint: Optional[str] = None,
    time_offset: float = 0.0,
    detect_speaker: bool = False,
    max_speakers: Optional[int] = 2,
    session_id: str = "default",
) -> dict:
    """
    Chạy ASR trên full buffer với cấu hình tốt hơn để có độ chính xác cao.
    Sử dụng beam_size=5, best_of=5 để có kết quả chính xác hơn.
    """
    audio_f32 = pcm16_to_float32(audio_buffer)
    if audio_f32.size == 0:
        return {"text": "", "segments": []}
    
    # Cần tối thiểu ~0.5s audio để transcribe (8000 samples @ 16kHz)
    MIN_SAMPLES = 8000
    if audio_f32.size < MIN_SAMPLES:
        return {"text": "", "segments": []}
    
    normalized_max_speakers: Optional[int] = None
    if detect_speaker:
        try:
            normalized_max_speakers = max(1, int(max_speakers if max_speakers is not None else 2))
        except (TypeError, ValueError):
            normalized_max_speakers = 2

    try:
        # VAD filter
        vad_options = VadOptions(min_silence_duration_ms=900)
        speech_chunks = get_speech_timestamps(audio_f32, vad_options)
        if len(speech_chunks) == 0:
            return {"text": "", "segments": []}
        
        audio_f32 = collect_chunks(audio_f32, speech_chunks)

        speaker_id = None
        if detect_speaker:
            speaker_id = get_speaker_id(
                audio_f32,
                session_id=session_id,
                max_speakers=normalized_max_speakers
            )
            print("speaker_id (full): ", speaker_id)

        # Transcribe với cấu hình tốt hơn để có độ chính xác cao
        segments, info = model.transcribe(
            audio_f32,
            language=lang_hint,
            vad_filter=False,
            beam_size=5,  # Tăng độ chính xác
            best_of=5,
            condition_on_previous_text=True,
            temperature=0.0,
            word_timestamps=True,
        )

        full_texts = []
        segments_list = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                full_texts.append(t)
                segment_entry = {
                    "start": round(seg.start + time_offset, 2),
                    "end": round(seg.end + time_offset, 2),
                    "text": t,
                }
                if speaker_id:
                    segment_entry["speaker_id"] = speaker_id
                segments_list.append(segment_entry)

        return {
            "speaker_id": speaker_id,
            "language": info.language if info else "auto",
            "language_probability": round(info.language_probability if info else 0, 3),
            "text": " ".join(full_texts),
            "segments": segments_list,
        }
    except (ValueError, RuntimeError) as e:
        error_msg = str(e).lower()
        if "empty" in error_msg or "max()" in error_msg or "sequence" in error_msg:
            return {"text": "", "segments": []}
        print(f"Full transcription error: {e}")
        raise

async def run_transcribe_on_buffer(
    audio_buffer: bytes,
    lang_hint: Optional[str] = None,
    time_offset: float = 0.0,
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

    try:
        # VAD filter
        # Kiểm tra nếu audio có khoảng im lặng > 750ms trong 1s thì trả về "silence"
        vad_options = VadOptions(min_silence_duration_ms=750)
        speech_chunks = get_speech_timestamps(audio_f32, vad_options)
        if len(speech_chunks) == 0:
            return {"text": "silence", "segments": []}
        # for idx, chunk in enumerate(speech_chunks):
        #     print(f"chunk {idx}: start={chunk['start']} - {chunk['start'] / 16000}s, end={chunk['end']} - {chunk['end'] / 16000}s")
        # audio_f32 = collect_chunks(audio_f32, speech_chunks) # 1s nên không cần dùng vad lọc silence

        # faster-whisper nhận numpy audio array (1-D float32, sample_rate=16000)
        # Sử dụng vad_filter để bỏ im lặng; beam_size thấp để nhanh hơn.
        segments, info = model.transcribe(
            audio_f32,
            language=lang_hint,            # None -> auto-detect; "vi" -> tiếng Việt
            vad_filter=False,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,  # giúp các chunk độc lập, đỡ "lệch ngữ cảnh"
            temperature=0.0,
            word_timestamps=True,
        )
        # print("info: ", _)
        # print("time_offset: ", time_offset)

        partial_texts = []
        segments_list = []
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
                segments_list.append(segment_entry)

        print("partial_texts: ", "||".join(partial_texts))
        return {
            "language": info.language if info else "auto",
            "language_probability": round(info.language_probability if info else 0, 3),
            "text": "||".join(partial_texts),
            "segments": segments_list,
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

    recv_buffer = bytearray()
    full_buffer = bytearray()  # Lưu trữ toàn bộ audio của một đoạn speech
    full_buffer_start_time = 0.0  # Thời gian bắt đầu của full_buffer
    running = True
    time_offset = 0.0  # Track thời gian tích lũy từ đầu session
    end_speech = False

    async def flush_and_transcribe():
        """Chạy ASR trên buffer hiện tại và gửi partial về client."""
        nonlocal recv_buffer, time_offset, total_segments_sent, end_speech, full_buffer, full_buffer_start_time
        if len(recv_buffer) == 0:
            return
        try:
            buffer_size = len(recv_buffer)
            transcribe_start = datetime.now()
            # Tính thời gian của buffer này (bytes / 2 / sample_rate = seconds)
            buffer_duration = buffer_size / 2 / sample_rate
            
            # Lưu buffer hiện tại vào full_buffer trước khi transcribe
            current_buffer_bytes = bytes(recv_buffer)
            
            # Nếu full_buffer rỗng, đây là bắt đầu của một đoạn speech mới
            if len(full_buffer) == 0:
                full_buffer_start_time = time_offset
            
            result = await run_transcribe_on_buffer(
                current_buffer_bytes,
                lang_hint,
                time_offset,
            )

            transcribe_time = (datetime.now() - transcribe_start).total_seconds()
            recv_buffer = bytearray()  # reset buffer sau mỗi lần flush
            
            if result["text"]:
                if result["text"] == "silence" or has_repeated_substring(result["text"]):
                    end_speech = True
                    print("end_speech: ", end_speech)
                    
                    # Khi phát hiện end_speech, transcribe lại trên full_buffer với cấu hình tốt hơn
                    if len(full_buffer) > 0:
                        try:
                            full_transcribe_start = datetime.now()
                            full_result = await run_transcribe_on_full_buffer(
                                bytes(full_buffer),
                                lang_hint,
                                full_buffer_start_time,
                                detect_speaker=detect_speaker_enabled,
                                max_speakers=max_speakers_limit if detect_speaker_enabled else None,
                                session_id=ws_session_id,
                            )
                            full_transcribe_time = (datetime.now() - full_transcribe_start).total_seconds()
                            
                            if full_result["text"] and not has_repeated_substring(full_result["text"]):
                                speaker_label = full_result.get("speaker_id")
                                full_text = full_result.get("text", "")
                                
                                # Format text với speaker label nếu có
                                if speaker_label:
                                    formatted_full_text = f"{speaker_label}: {full_text}"
                                else:
                                    formatted_full_text = full_text
                                
                                print("full_result_text: ", formatted_full_text)
                                print("--------------------------------")
                                
                                await ws.send_text(json.dumps({
                                    "type": "full",
                                    "speaker_id": speaker_label,
                                    "language": full_result.get("language"),
                                    "language_probability": full_result.get("language_probability"),
                                    "text": formatted_full_text,
                                    "segments": full_result.get("segments", []),
                                }))
                                
                                full_segments_count = len(full_result.get("segments", []))
                                logger.info(f"[WS] Sent full: segments={full_segments_count}, text_length={len(full_text)}, transcribe_time={full_transcribe_time:.3f}s, full_buffer_size={len(full_buffer)} bytes")
                            
                            # Reset full_buffer sau khi đã transcribe và gửi
                            full_buffer = bytearray()
                            full_buffer_start_time = 0.0
                        except Exception as e:
                            logger.error(f"[WS] Full transcription error: {str(e)}", exc_info=True)
                            # Vẫn reset full_buffer để tránh tích lũy quá nhiều
                            full_buffer = bytearray()
                            full_buffer_start_time = 0.0
                    # else:
                    #     end_speech = False
                    #     print("end_speech: ", end_speech)
                else:
                    # Lưu buffer vào full_buffer (chỉ khi không phải silence)
                    full_buffer.extend(current_buffer_bytes)
                    
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
                    if result.get("language") != "ja":
                        result_text = result_text.replace("||", " ")
                    else:
                        result_text = result_text.replace("||", "")
                    print("result_text: ", result_text)
                    print("--------------------------------")
                    await ws.send_text(json.dumps({
                        "type": "partial",
                        "speaker_id": speaker_label,
                        "language": result.get("language"),
                        "language_probability": result.get("language_probability"),
                        "text": result_text,
                        "segments": result.get("segments", []),
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
                        diarization_model.reset_session(ws_session_id)
                        
                        # Reset time offset khi bắt đầu session mới
                        time_offset = 0.0
                        recv_buffer = bytearray()
                        full_buffer = bytearray()
                        full_buffer_start_time = 0.0
                        end_speech = False
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
                        
                        # Nếu còn full_buffer, transcribe và gửi full transcript
                        if len(full_buffer) > 0:
                            try:
                                full_result = await run_transcribe_on_full_buffer(
                                    bytes(full_buffer),
                                    lang_hint,
                                    full_buffer_start_time,
                                    detect_speaker=detect_speaker_enabled,
                                    max_speakers=max_speakers_limit if detect_speaker_enabled else None,
                                )
                                
                                if full_result["text"]:
                                    speaker_label = full_result.get("speaker_id")
                                    full_text = full_result.get("text", "")
                                    
                                    if speaker_label:
                                        formatted_full_text = f"{speaker_label}: {full_text}"
                                    else:
                                        formatted_full_text = full_text
                                    
                                    await ws.send_text(json.dumps({
                                        "type": "full",
                                        "speaker_id": speaker_label,
                                        "language": full_result.get("language"),
                                        "language_probability": full_result.get("language_probability"),
                                        "text": formatted_full_text,
                                        "segments": full_result.get("segments", []),
                                    }))
                                    logger.info(f"[WS] Sent final full transcript: text_length={len(full_text)}")
                            except Exception as e:
                                logger.error(f"[WS] Final full transcription error: {str(e)}", exc_info=True)
                        
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
        diarization_model.delete_session(ws_session_id)
        logger.debug(f"[WS] Cleaned up session: {ws_session_id}")

    except WebSocketDisconnect:
        session_duration = (datetime.now() - session_start).total_seconds() if 'session_start' in locals() else 0
        logger.info(f"[WS] WebSocket disconnected: client={client_ip}, duration={session_duration:.2f}s")
        # Cleanup session
        if 'ws_session_id' in locals():
            diarization_model.delete_session(ws_session_id)
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
            if 'ws_session_id' in locals():
                diarization_model.delete_session(ws_session_id)
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