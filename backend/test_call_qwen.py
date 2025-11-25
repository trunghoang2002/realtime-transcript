import base64
import os
import requests
import json

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

audio_source_url =file_to_audio_data_url("eval/TestJ.mp3")

url = "https://trunghoang2002--qwen-audio-modal-serve-dev.modal.run/v1/chat/completions"

# payload = {
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "audio_url", "audio_url": {"url": audio_source_url}},
#                 # {
#                 #     "type": "audio",
#                 #     "audio": {
#                 #         "data": audio_base64,
#                 #         "format": "wav"
#                 #     }
#                 # },
#                 {"type": "text", "text": "Give me the transcription of the audio. Only provide the transcription without any additional information or any other text."}
#             ]
#         }
#     ],
#     "max_tokens": 50,
#     "stream": False
# }
url = "https://game-powerful-kit.ngrok-free.app/v1/chat/completions"
payload = {
    "model": "qwen3-omni-30B",
    "messages": [
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
}

headers = {"Content-Type": "application/json"}

# For streaming response
# with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as resp:
#     if resp.status_code != 200:
#         print("Error:", resp.status_code, resp.text)
#     else:
#         full_text = ""
#         for line in resp.iter_lines(decode_unicode=True):
#             if not line or not line.startswith("data: "):
#                 continue
#             data_str = line[len("data: "):]
#             if data_str.strip() == "[DONE]":
#                 break

#             try:
#                 data = json.loads(data_str)
#                 delta = data["choices"][0].get("delta", {}).get("content", "")
#                 full_text += delta
#                 print(delta, end="", flush=True)  # stream real-time
#             except Exception as e:
#                 print("\n⚠️ Parse error:", e)

# For non-streaming response
response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status code:", response.status_code)
print("Response:")
# print(response.json())
print(response.json()["choices"][0]["message"]["content"])
# print(response.json()['transcription'])
