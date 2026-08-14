"""
stt.py — Speech-to-Text via Sarvam AI API

Wraps Sarvam's saarika:v2 STT model for Indic language transcription.
Chosen because it has the best coverage for Indian languages (Hindi, Tamil,
Bengali, etc.) compared to alternatives like Whisper which have weaker
Indic accuracy. Includes retry logic with exponential backoff since the
Sarvam API is a network call and transient failures are expected.
"""

import os
import time
import requests
from dataclasses import dataclass


class STTError(Exception):
    """Raised when speech-to-text transcription fails after all retries.

    Separate exception class so the pipeline orchestrator can distinguish
    STT failures from other errors and set the appropriate status code.
    """
    pass


@dataclass
class STTResult:
    """Structured result from speech-to-text transcription."""
    transcript: str
    language_code: str


# -- Configuration --
_SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
_SARVAM_MODEL = os.environ.get("SARVAM_MODEL", "saarika:v2.5")
_MAX_RETRIES = 2
# Backoff schedule: 0.5s after first failure, 1.0s after second
_BACKOFF_SECONDS = [0.5, 1.0]

# Module-level session to reuse TCP connections for STT
_session = requests.Session()


def _get_api_key() -> str:
    """Fetch SARVAM_API_KEY from environment, fail fast if missing."""
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise STTError(
            "SARVAM_API_KEY environment variable is not set. "
            "Get your key at https://www.sarvam.ai/"
        )
    return key


def transcribe(audio_bytes: bytes, language_code: str = "hi-IN") -> STTResult:
    """Transcribe audio bytes to text using Sarvam's STT API.

    Args:
        audio_bytes: Raw audio file bytes (WAV, MP3, etc.)
        language_code: BCP-47 language code (e.g. "hi-IN" for Hindi,
                       "ta-IN" for Tamil). Sarvam supports most Indic languages.

    Returns:
        STTResult with transcript text and language code.

    Raises:
        STTError: If transcription fails after all retries (2 retries = 3 total attempts).
    """
    api_key = _get_api_key()

    headers = {
        "api-subscription-key": api_key,
    }

    # Sarvam expects multipart form data with the audio file and model params
    files = {
        "file": ("audio.wav", audio_bytes, "audio/wav"),
    }
    data = {
        "model": _SARVAM_MODEL,
        "language_code": language_code,
    }

    last_error = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = _session.post(
                _SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=30,  # 30s timeout — STT can be slow for long audio
            )

            if response.status_code == 200:
                result = response.json()
                transcript = result.get("transcript", "")
                return STTResult(
                    transcript=transcript,
                    language_code=language_code,
                )

            # Non-retryable client errors (4xx except 429 rate limit)
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise STTError(
                    f"Sarvam STT API returned client error {response.status_code}: "
                    f"{response.text}"
                )

            # Server errors (5xx) and rate limits (429) are retryable
            last_error = (
                f"Sarvam STT API returned {response.status_code}: {response.text}"
            )

        except requests.exceptions.RequestException as e:
            # Network errors (timeout, connection refused, etc.) are retryable
            last_error = f"Network error calling Sarvam STT API: {e}"

        # Apply backoff before retry (but not after the last attempt)
        if attempt < _MAX_RETRIES:
            backoff = _BACKOFF_SECONDS[attempt]
            time.sleep(backoff)

    # All retries exhausted
    raise STTError(
        f"Sarvam STT transcription failed after {_MAX_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )
