import base64
import os
from openai import OpenAI

def file_to_audio_data_url(file_path: str):
    """
    Convert a local audio file (e.g., .wav, .mp3, .ogg) to a base64 data URL.
    """
    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

    _, extension = os.path.splitext(file_path)
    extension = extension[1:].lower()  # bỏ dấu chấm
    mime_type = f"audio/{'mpeg' if extension == 'mp3' else extension}"

    return f"data:{mime_type};base64,{encoded_string}"

audio_source_url = file_to_audio_data_url("eval/TestJ.mp3")

MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
BASE_URL = "https://vjclaspi1--qwen-audio-modal-serve-dev.modal.run/v1"
# MODEL_NAME = "qwen3-omni-30B"
# BASE_URL = "https://game-powerful-kit.ngrok-free.app/v1"
# Initialize OpenAI client with custom base URL
client = OpenAI(
    base_url=BASE_URL,
    api_key="EMPTY"  # Some endpoints may not require a real key
)

# Create chat completion using OpenAI compatible API

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": audio_source_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "Transcribe the audio into text."
                    }
                ]
            }
        ]
    )
    
    print("Response:")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"Error: {e}")

