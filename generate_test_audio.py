"""
generate_test_audio.py — Generate test audio files using Sarvam AI TTS + standard wave for silence
"""
import os
import wave
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY is not set in environment or .env")

AUDIO_DIR = "eval/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def generate_sarvam_audio(text: str, target_lang: str, output_path: str):
    """Generate audio using Sarvam AI Text-to-Speech API."""
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Try bulbul:v2, fallback to bulbul:v1 if needed
    for model_name in ["bulbul:v2"]:
        payload = {
            "inputs": [text],
            "target_language_code": target_lang,
            "speaker": "anushka",
            "model": model_name
        }
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        if res.status_code == 200:
            data = res.json()
            audio_base64 = data["audios"][0]
            audio_bytes = base64.b64decode(audio_base64)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"Generated {output_path} ({len(audio_bytes)} bytes) using Sarvam {model_name} for '{text}'")
            return
        else:
            print(f"Sarvam TTS with {model_name} returned {res.status_code}: {res.text}")

    raise RuntimeError(f"Failed to generate Sarvam audio for: {text}")

def generate_silent_wav(output_path: str, duration_sec: float = 2.0, sample_rate: int = 16000):
    """Generate a 2-second silent WAV file using standard library wave module."""
    num_samples = int(duration_sec * sample_rate)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        # Write zeros for silence
        wf.writeframes(b"\x00\x00" * num_samples)
    print(f"Generated silent WAV {output_path} ({duration_sec}s)")

if __name__ == "__main__":
    # 1. Clear English audio
    generate_sarvam_audio(
        text="What was the immediate impact of the success of the Manhattan Project?",
        target_lang="en-IN",
        output_path=os.path.join(AUDIO_DIR, "manhattan_en.wav")
    )

    # 2. Clear Hindi audio
    generate_sarvam_audio(
        text="मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",
        target_lang="hi-IN",
        output_path=os.path.join(AUDIO_DIR, "manhattan_hi.wav")
    )

    # 3. Silent audio clip
    generate_silent_wav(
        output_path=os.path.join(AUDIO_DIR, "silent.wav"),
        duration_sec=2.0
    )
