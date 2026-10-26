"""
PetPulse voice processing.

Both directions run on this computer, inside the Python server:
- Speech to text: faster-whisper ("base" model, CPU, INT8). The model is
  downloaded once from Hugging Face, then loaded from the local cache.
- Text to speech: the macOS built-in `say` command.

Recordings stay in memory and are never saved. Spoken answers are written to a
temporary folder that is deleted right after the audio is read back.
"""

import functools
import io
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

WHISPER_MODEL = "base"
LANGUAGE = "en"
SAY_TIMEOUT_SECONDS = 60


class VoiceError(Exception):
    """A voice problem the sitter should see as a friendly message."""


@functools.lru_cache(maxsize=1)
def get_whisper_model():
    """Load the Whisper model once and reuse it for every recording."""
    from faster_whisper import WhisperModel  # imported here so the app starts fast

    try:
        # Use the copy already on this computer, without checking online.
        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        # First run only: download the model (about 140 MB) from Hugging Face.
        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe_audio(audio_bytes):
    """Turn a WAV recording into text. Raises VoiceError if nothing usable was heard."""
    if not audio_bytes or len(audio_bytes) < 1000:  # a WAV header alone is ~44 bytes
        raise VoiceError("The recording is empty. Please record your question again.")

    try:
        model = get_whisper_model()
        segments, _info = model.transcribe(
            io.BytesIO(audio_bytes),
            language=LANGUAGE,
            vad_filter=True,  # skip silence, which also stops Whisper inventing words
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception:
        traceback.print_exc()  # details go to the terminal only
        raise VoiceError("Sorry, I couldn't understand that recording. Please try again or type your question.")

    if not text:
        raise VoiceError("No speech was detected. Please speak closer to the microphone, or type your question.")
    return text


def speech_text(text):
    """Remove Markdown symbols so `say` doesn't read them out loud."""
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def synthesize_speech(text):
    """Read text aloud with macOS `say` and return WAV audio bytes."""
    text = speech_text(text or "")
    if not text:
        raise VoiceError("There is no answer to read aloud.")
    if shutil.which("say") is None:
        raise VoiceError("Spoken answers need the macOS `say` command, which isn't available here.")

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "answer.wav"
        try:
            # The text goes in through stdin, not as an argument, so it can never be read as an option.
            subprocess.run(
                ["say", "-o", str(output), "--file-format=WAVE", "--data-format=LEI16@22050"],
                input=text,
                text=True,
                capture_output=True,
                timeout=SAY_TIMEOUT_SECONDS,
                check=True,
            )
            audio = output.read_bytes()
        except (OSError, subprocess.SubprocessError):
            traceback.print_exc()
            raise VoiceError("The spoken answer couldn't be created, but you can read it in the chat.")

    if not audio:
        raise VoiceError("The spoken answer couldn't be created, but you can read it in the chat.")
    return audio
