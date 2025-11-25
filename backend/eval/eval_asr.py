import sys
sys.path.append("..")
import csv
from jiwer import wer, cer
import jiwer.transforms as tr
import regex as re
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from google import genai
from google.genai import types
import io
import soundfile as sf
import os
import dotenv
dotenv.load_dotenv()
from sudachipy import dictionary, tokenizer
from faster_whisper import WhisperModel
from funasr import AutoModel
from silero_vad import VadOptions, get_speech_timestamps, collect_chunks
from get_audio import decode_audio
import time
import librosa
import base64
from openai import OpenAI

class JPTokenizer:
    def __init__(self):
        self.tok = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.A

    def _tokenize_str(self, text: str):
        return [m.surface() for m in self.tok.tokenize(text, self.mode)]

    def __call__(self, text):
        # Case 1: string
        if isinstance(text, str):
            return [self._tokenize_str(text)]  # list-of-list như yêu cầu WER

        # Case 2: list-of-strings (jiwer đôi khi truyền list)
        if isinstance(text, list):
            result = []
            for t in text:
                if isinstance(t, str):
                    result.append(self._tokenize_str(t))
                elif isinstance(t, list):
                    # nested list (rare case)
                    result.append([self._tokenize_str(x) for x in t])
            return result

        raise TypeError(f"Unsupported type for JPTokenizer: {type(text)}")

wer_ja = tr.Compose([
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
    JPTokenizer(),
])

def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0.0

def eval_score(ground_truth, prediction):
    # normalize text
    ground_truth = ground_truth.lower()
    prediction = prediction.lower()

    pattern = r"[\p{P}～~＋＝＄|]+"
    ground_truth = re.sub(pattern, "", ground_truth)
    prediction = re.sub(pattern, "", prediction)

    prediction = prediction.replace("<[^>]*>", "")

    ground_truth = re.sub(r"[A-Za-z\s]+", "", ground_truth).strip()
    prediction = re.sub(r"[A-Za-z\s]+", "", prediction).strip()

    wer_score = wer(ground_truth, prediction, reference_transform=wer_ja, hypothesis_transform=wer_ja)
    cer_score = cer(ground_truth, prediction)
    return wer_score, cer_score

# ============================================
#               BASE CLASS
# ============================================
class BaseASR(ABC):
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def _transcribe_no_vad(self, audio):
        """Transcribe thuần không VAD."""
        pass

    @abstractmethod
    def _transcribe_with_vad(self, audio):
        """Transcribe với VAD."""
        pass

    def transcribe(self, file_path: str, vad_filter: bool = False) -> str:
        audio = decode_audio(file_path)
        return (
            self._transcribe_with_vad(audio)
            if vad_filter else
            self._transcribe_no_vad(audio)
        )

# ============================================
#         WHISPER JAPANESE IMPLEMENTATION
# ============================================
class WhisperJA(BaseASR):
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
    ):
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature

    # -------------------------------
    #   Transcribe WITHOUT VAD
    # -------------------------------
    def _transcribe_no_vad(self, audio):
        segments, info = self.model.transcribe(
            audio,
            language="ja",
            vad_filter=False,
            beam_size=self.beam_size,
            best_of=self.best_of,
            condition_on_previous_text=True,
            temperature=self.temperature,
            without_timestamps=True,
        )

        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return "".join(texts)

    # -------------------------------
    #   Transcribe WITH VAD
    # -------------------------------
    def _transcribe_with_vad(self, audio):
        vad_options = VadOptions(min_silence_duration_ms=800)
        speech_chunks = get_speech_timestamps(audio, vad_options)

        results = []

        for chunk in speech_chunks:
            audio_idx = collect_chunks(audio, [chunk])

            segments, info = self.model.transcribe(
                audio_idx,
                language="ja",
                vad_filter=False,
                beam_size=self.beam_size,
                best_of=self.best_of,
                condition_on_previous_text=True,
                temperature=self.temperature,
                word_timestamps=True,
            )

            texts = [seg.text.strip() for seg in segments if seg.text.strip()]
            if texts:
                results.append("".join(texts))

        return "".join(results)

    # -------------------------------
    #   PUBLIC API
    # -------------------------------
    def transcribe(self, file_path: str, vad_filter: bool = False) -> str:
        audio = decode_audio(file_path)

        if vad_filter:
            return self._transcribe_with_vad(audio)
        else:
            return self._transcribe_no_vad(audio)

# ============================================
#        SENSEVOICE JAPANESE IMPLEMENTATION
# ============================================
class SenseVoiceJA(BaseASR):
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        max_single_segment_time: int = 30000,
        batch_size_s: int = 20,
        use_itn: bool = False,
    ):
        self.model_name = model_name
        self.use_itn = use_itn
        self.batch_size_s = batch_size_s

        # Load model
        self.model = AutoModel(
            model=model_name,
            device=device,
            vad_kwargs={"max_single_segment_time": max_single_segment_time},
        )

    # -----------------------------------
    #   Helper: Clean output text
    # -----------------------------------
    def _clean_text(self, text: str):
        text = re.sub(r"<\|[^|>]+\|>", "", text).strip()
        text = text.replace(" ", "")
        return text

    # -----------------------------------
    #      Transcribe WITHOUT VAD
    # -----------------------------------
    def _transcribe_no_vad(self, audio):
        res = self.model.generate(
            input=audio,
            language="ja",
            use_itn=self.use_itn,
            batch_size_s=self.batch_size_s,
        )
        raw = res[0]["text"]
        return self._clean_text(raw)

    # -----------------------------------
    #        Transcribe WITH VAD
    # -----------------------------------
    def _transcribe_with_vad(self, audio):
        vad_options = VadOptions(min_silence_duration_ms=800)
        speech_chunks = get_speech_timestamps(audio, vad_options)

        all_text = []

        for chunk in speech_chunks:
            audio_idx = collect_chunks(audio, [chunk])

            res = self.model.generate(
                input=audio_idx,
                language="ja",
                use_itn=self.use_itn,
                batch_size_s=self.batch_size_s,
            )

            raw = res[0]["text"]
            cleaned = self._clean_text(raw)

            if cleaned:
                all_text.append(cleaned)

        return "".join(all_text)

    # -----------------------------------
    #           PUBLIC API
    # -----------------------------------
    def transcribe(self, file_path: str, vad_filter: bool = False) -> str:
        audio = decode_audio(file_path)

        if vad_filter:
            return self._transcribe_with_vad(audio)
        else:
            return self._transcribe_no_vad(audio)

# ======================================================
#               GEMINI ASR CLASS
# ======================================================
class GeminiASR(BaseASR):
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.5-flash" 
    ):

        # Initialize Gemini client
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    # ------------------------------------------------------
    # Helper: Convert audio ndarray → WAV bytes
    # ------------------------------------------------------
    def _to_wav_bytes(self, audio: np.ndarray, sr: int = 16000) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=sr, format="WAV")
        buffer.seek(0)
        return buffer.read()

    # ------------------------------------------------------
    # Helper: Call Gemini
    # ------------------------------------------------------
    def _call_gemini(self, audio_bytes: bytes, max_retries: int = 3) -> str:
        last_exception = None
        response = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        (
                            "Generate a transcript of the speech. "
                            "Return ONLY the transcript. "
                            "If no speech, return 'no speech'. "
                            "Language: Japanese."
                        ),
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type="audio/wav",
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=2048,
                        # safety_settings=[
                        #     types.SafetySetting(
                        #         category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        #         threshold=types.HarmBlockThreshold.OFF,
                        #     ),
                        #     types.SafetySetting(
                        #         category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        #         threshold=types.HarmBlockThreshold.OFF,
                        #     ),
                        #     types.SafetySetting(
                        #         category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        #         threshold=types.HarmBlockThreshold.OFF,
                        #     ),
                        #     types.SafetySetting(
                        #         category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        #         threshold=types.HarmBlockThreshold.OFF,
                        #     ),
                        #     types.SafetySetting(
                        #         category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        #         threshold=types.HarmBlockThreshold.OFF,
                        #     ),
                        #     types.SafetySetting(
                        #         category=types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                        #         threshold=types.HarmBlockThreshold.OFF,
                        #     ),
                        # ]
                    ),
                )
                return response.text.strip()
            
            except Exception as e:
                print("Error: ", e)
                print("response: ", response)
                last_exception = e
                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"All {max_retries} attempts failed.")
        
        # If all retries failed, raise the last exception
        # raise last_exception
        return ""
    
    # ------------------------------------------------------
    #     Transcribe WITHOUT VAD
    # ------------------------------------------------------
    def _transcribe_no_vad(self, audio):
        wav_bytes = self._to_wav_bytes(audio)
        text = self._call_gemini(wav_bytes)
        if text and text.lower() != "no speech":
            return text
        else:
            return ""

    # ------------------------------------------------------
    #       Transcribe WITH VAD
    # ------------------------------------------------------
    def _transcribe_with_vad(self, audio):
        vad_options = VadOptions(min_silence_duration_ms=800)
        speech_chunks = get_speech_timestamps(audio, vad_options)

        all_text = []

        for chunk in speech_chunks:
            audio_idx = collect_chunks(audio, [chunk])
            wav_bytes = self._to_wav_bytes(audio_idx)
            text = self._call_gemini(wav_bytes)
            if text and text.lower() != "no speech":
                all_text.append(text)

        return "".join(all_text)

    # ------------------------------------------------------
    #           PUBLIC API
    # ------------------------------------------------------
    def transcribe(self, file_path: str, vad_filter: bool = False) -> str:
        audio = decode_audio(file_path)
        if vad_filter:
            return self._transcribe_with_vad(audio)
        else:
            return self._transcribe_no_vad(audio)

# ======================================================
#               VLLM ASR CLASS
# ======================================================
class VllmASR(BaseASR):
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        prompt: str = "Transcribe the audio into text."
    ):
        """
        Initialize VLLM ASR with OpenAI-compatible API endpoint.
        
        Args:
            base_url: Base URL of the vLLM server (e.g., "https://...modal.run/v1")
            api_key: API key (default: "EMPTY" for endpoints that don't require auth)
            model_name: Model name to use
            prompt: Transcription prompt
        """
        self.model_name = model_name
        self.prompt = prompt
        
        # Initialize OpenAI client with custom base URL
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    # ------------------------------------------------------
    # Helper: Convert audio ndarray → base64 data URL
    # ------------------------------------------------------
    def _audio_to_data_url(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Convert audio numpy array to base64 data URL.
        """
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=sr, format="WAV")
        buffer.seek(0)
        wav_bytes = buffer.read()
        
        # Encode to base64
        encoded_string = base64.b64encode(wav_bytes).decode("utf-8")
        mime_type = "audio/wav"
        
        return f"data:{mime_type};base64,{encoded_string}"

    # ------------------------------------------------------
    # Helper: Call vLLM API
    # ------------------------------------------------------
    def _call_vllm(self, audio_data_url: str, max_retries: int = 3) -> str:
        """
        Call vLLM API with audio data URL.
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {
                                        "url": audio_data_url
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": self.prompt
                                }
                            ]
                        }
                    ]
                )
                
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"All {max_retries} attempts failed: {last_exception}")
        
        return ""

    # ------------------------------------------------------
    #     Transcribe WITHOUT VAD
    # ------------------------------------------------------
    def _transcribe_no_vad(self, audio):
        audio_data_url = self._audio_to_data_url(audio)
        text = self._call_vllm(audio_data_url)
        if text and text.lower() != "no speech":
            return text
        else:
            return ""

    # ------------------------------------------------------
    #       Transcribe WITH VAD
    # ------------------------------------------------------
    def _transcribe_with_vad(self, audio):
        vad_options = VadOptions(min_silence_duration_ms=800)
        speech_chunks = get_speech_timestamps(audio, vad_options)

        all_text = []

        for chunk in speech_chunks:
            audio_idx = collect_chunks(audio, [chunk])
            audio_data_url = self._audio_to_data_url(audio_idx)
            text = self._call_vllm(audio_data_url)
            if text and text.lower() != "no speech":
                all_text.append(text)

        return "".join(all_text)

    # ------------------------------------------------------
    #           PUBLIC API
    # ------------------------------------------------------
    def transcribe(self, file_path: str, vad_filter: bool = False) -> str:
        audio = decode_audio(file_path)
        if vad_filter:
            return self._transcribe_with_vad(audio)
        else:
            return self._transcribe_no_vad(audio)

transcriber = WhisperJA(
    model_name="large-v1", # tiny, base, small, medium, large, large-v1, large-v2, large-v3, turbo
    device="cuda",
    compute_type="float16"
)

# transcriber = SenseVoiceJA(
#     model_name="iic/SenseVoiceSmall",
#     device="cuda",
#     max_single_segment_time=30000,
#     batch_size_s=20,
#     use_itn=False,
# )

# transcriber = GeminiASR(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     model_name="gemini-2.5-pro", # gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro
# )

# MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
# BASE_URL = "https://vjclaspi1--qwen-audio-modal-serve-dev.modal.run/v1"
# # MODEL_NAME = "qwen3-omni-30B"
# # BASE_URL = "https://game-powerful-kit.ngrok-free.app/v1"
# transcriber = VllmASR(
#     base_url=BASE_URL,
#     api_key="EMPTY",
#     model_name=MODEL_NAME,
#     prompt=(
#             "Generate a transcript of the speech. "
#             "Return ONLY the transcript. "
#             "If no speech, return 'no speech'. "
#             "Language: Japanese."
#         )
# )

# ============================================
#   CHECKPOINT MECHANISM
# ============================================
results_file = "eval_results_checkpoint.csv"
completed_files = set()

# Load previous results if exists
if os.path.exists(results_file):
    print(f"Found checkpoint file: {results_file}")
    with open(results_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 5:
                completed_files.add(row[0])  # file_path
    print(f"Already completed: {len(completed_files)} test cases")
else:
    # Create new results file with header
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "ground_truth", "prediction", "wer_score", "cer_score", "rtf", "audio_duration", "processing_time"])
    print(f"Created new checkpoint file: {results_file}")

wer_scores = []
cer_scores = []
rtf_scores = []

# Read dataset & evaluate
with open("dataset_400_testcases.csv", "r") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    rows = list(reader)
    
    print(f"Total test cases: {len(rows)}")
    print(f"Remaining: {len(rows) - len(completed_files)}")
    
    for row in tqdm(rows):
        file_path = row[3]
        ground_truth = row[4]
        
        # Skip if already processed
        if file_path in completed_files:
            continue
        
        try:
            # Get audio duration
            audio_duration = get_audio_duration(file_path)
            
            # Measure processing time
            start_time = time.time()
            prediction = transcriber.transcribe(file_path)
            processing_time = time.time() - start_time
            
            # Calculate RTF
            rtf = processing_time / audio_duration if audio_duration > 0 else 0.0
            
            if prediction == "":
                print(f"No prediction for {file_path}")
                continue
            wer_score, cer_score = eval_score(ground_truth, prediction)
            
            # Save result immediately after each test case
            with open(results_file, "a", encoding="utf-8", newline="") as f_out:
                writer = csv.writer(f_out)
                writer.writerow([file_path, ground_truth, prediction, wer_score, cer_score, rtf, audio_duration, processing_time])
            
            wer_scores.append(wer_score)
            cer_scores.append(cer_score)
            rtf_scores.append(rtf)
            
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            print("Progress saved. You can resume by running the script again.")
            raise

# Calculate final scores from checkpoint file
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

results_file = "eval_results_checkpoint.csv"
all_wer_scores = []
all_cer_scores = []
all_rtf_scores = []

with open(results_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 6 and row[3] != "" and row[4] != "" and row[5] != "":
            all_wer_scores.append(float(row[3]))
            all_cer_scores.append(float(row[4]))
            all_rtf_scores.append(float(row[5]))
        # # Re-evaluate all test cases
        # if len(row) >= 3 and row[1] != "" and row[2] != "":
        #     wer_score, cer_score = eval_score(row[1], row[2])
        #     all_wer_scores.append(wer_score)
        #     all_cer_scores.append(cer_score)
        # if len(row) >= 6 and row[5] != "":
        #     all_rtf_scores.append(float(row[5]))


print(f"Total evaluated: {len(all_wer_scores)} test cases")
print(f"WER: {np.mean(all_wer_scores):.4f}")
print(f"CER: {np.mean(all_cer_scores):.4f}")
if all_rtf_scores:
    print(f"RTF: {np.mean(all_rtf_scores):.4f} (mean)")
    print(f"RTF: {np.median(all_rtf_scores):.4f} (median)")
    print(f"RTF: {np.min(all_rtf_scores):.4f} (min) | {np.max(all_rtf_scores):.4f} (max)")