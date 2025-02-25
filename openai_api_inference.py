from pathlib import Path

import openai

client = openai.Client(base_url="http://localhost:8080", api_key="sk-template")

speech_file_path = Path(__file__).parent / "./results/speech.wav"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="冷氣團南下 北部轉涼白天氣溫降8度",
)

with open(speech_file_path, "wb") as audio_file:
    audio_file.write(response.content)
