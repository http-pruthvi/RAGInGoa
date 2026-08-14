"""
test_invalid_api_keys.py — Test invalid Groq and Sarvam API keys with FastAPI TestClient.
Tests the complete HTTP layer, status codes, and error formatting.
"""

import os
import sys
import io
import json
from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()

REAL_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
REAL_SARVAM_KEY = os.environ.get("SARVAM_API_KEY", "")

from app import app

AUDIO_PATH = "eval/audio/manhattan_hi.wav"

def run_tests():
    print("=" * 70)
    print("INVALID API KEYS RESILIENCE TESTS (GROQ & SARVAM)")
    print("=" * 70)

    with TestClient(app) as client:
        # -------------------------------------------------------------
        # 1. TEST A: INVALID GROQ_API_KEY
        # -------------------------------------------------------------
        print("\n--- 1. TESTING INVALID GROQ_API_KEY ---")
        os.environ["GROQ_API_KEY"] = "invalid_test_key_12345"
        os.environ["SARVAM_API_KEY"] = REAL_SARVAM_KEY

        r = client.post(
            "/query/text",
            json={"query": "What was the immediate impact of the success of the manhattan project?", "language": "en-IN"}
        )
        print(f"HTTP Status Code: {r.status_code}")
        print(f"Response JSON:\n{json.dumps(r.json(), indent=2)}")

        r_health = client.get("/health")
        print(f"/health Status After Error: {r_health.status_code} -> {r_health.json()}")

        assert r.status_code in (200, 500)
        body = r.json()
        assert body.get("status") == "internal-error"
        assert "groq" in body.get("error_message", "").lower() or "401" in body.get("error_message", "") or "invalid" in body.get("error_message", "").lower()
        assert r_health.status_code == 200
        print("✅ Test A PASSED: Server returned structured error response for invalid Groq key and stayed alive.")

        # -------------------------------------------------------------
        # 2. TEST B: INVALID SARVAM_API_KEY
        # -------------------------------------------------------------
        print("\n--- 2. TESTING INVALID SARVAM_API_KEY ---")
        os.environ["GROQ_API_KEY"] = REAL_GROQ_KEY
        os.environ["SARVAM_API_KEY"] = "invalid_test_key_12345"

        with open(AUDIO_PATH, "rb") as f:
            audio_bytes = f.read()

        r_voice = client.post(
            "/query/voice",
            files={"audio": ("manhattan_hi.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"language": "hi-IN"}
        )
        print(f"HTTP Status Code: {r_voice.status_code}")
        print(f"Response JSON:\n{json.dumps(r_voice.json(), indent=2)}")

        r_health2 = client.get("/health")
        print(f"/health Status After Error: {r_health2.status_code} -> {r_health2.json()}")

        assert r_voice.status_code in (200, 502, 500)
        body_v = r_voice.json()
        assert body_v.get("status") == "stt-error"
        assert "sarvam" in body_v.get("error_message", "").lower() or "401" in body_v.get("error_message", "") or "403" in body_v.get("error_message", "") or "unauthorized" in body_v.get("error_message", "").lower() or "client error" in body_v.get("error_message", "").lower()
        assert r_health2.status_code == 200
        print("✅ Test B PASSED: Server returned structured error response for invalid Sarvam key and stayed alive.")

        # -------------------------------------------------------------
        # 3. TEST C: RECOVERY WITH VALID KEYS
        # -------------------------------------------------------------
        print("\n--- 3. TESTING FULL RECOVERY WITH VALID KEYS ---")
        os.environ["GROQ_API_KEY"] = REAL_GROQ_KEY
        os.environ["SARVAM_API_KEY"] = REAL_SARVAM_KEY

        # Test Text
        r_text_rec = client.post(
            "/query/text",
            json={"query": "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?", "language": "hi-IN"}
        )
        print(f"Text Recovery Status Code: {r_text_rec.status_code} | Pipeline Status: '{r_text_rec.json().get('status')}'")
        print(f"Answer: {r_text_rec.json().get('answer')[:120]}...")
        assert r_text_rec.status_code == 200
        assert r_text_rec.json().get("status") == "success"

        # Test Voice
        r_voice_rec = client.post(
            "/query/voice",
            files={"audio": ("manhattan_hi.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"language": "hi-IN"}
        )
        print(f"Voice Recovery Status Code: {r_voice_rec.status_code} | Pipeline Status: '{r_voice_rec.json().get('status')}'")
        print(f"Answer: {r_voice_rec.json().get('answer')[:120]}...")
        assert r_voice_rec.status_code == 200
        assert r_voice_rec.json().get("status") == "success"

        print("✅ Test C PASSED: Server fully recovers and successfully serves text and voice queries with valid keys.")

    print("\n" + "=" * 70)
    print("ALL INVALID-KEY TESTS AND RECOVERY TESTS PASSED CLEANLY! ✅")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
