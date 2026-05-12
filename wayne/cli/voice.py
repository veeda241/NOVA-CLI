from __future__ import annotations

import sys
from pathlib import Path

import pyaudio
import webrtcvad
from rich.console import Console

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.language_engine import language_engine  # noqa: E402

console = Console()

RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(RATE * FRAME_MS / 1000)
MAX_SECONDS = 30


def record_until_silence() -> bytes:
    vad = webrtcvad.Vad(2)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE)
    frames: list[bytes] = []
    silence_count = 0
    speaking = False
    max_frames = int(MAX_SECONDS * 1000 / FRAME_MS)
    console.print("[blink yellow]Listening with local Whisper...[/blink yellow]")
    try:
        while len(frames) < max_frames:
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            frames.append(data)
            try:
                is_speech = vad.is_speech(data, RATE)
            except Exception:
                is_speech = True
            if is_speech:
                speaking = True
                silence_count = 0
            elif speaking:
                silence_count += 1
            if speaking and silence_count >= 26:
                break
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    return b"".join(frames)


def transcribe_voice(language: str | None = None) -> dict:
    audio_bytes = record_until_silence()
    return language_engine.transcribe(audio_bytes, language=language)


def listen() -> str:
    result = transcribe_voice()
    text = result.get("text", "")
    language = result.get("language", "en")
    confidence = float(result.get("confidence", 0.0))
    console.print(f"[dim]Heard ({language}, {confidence:.0%} confidence):[/dim] [cyan]{text}[/cyan]")
    if confidence < 0.35:
        console.print("[amber]Low confidence. Please repeat clearly.[/amber]")
        return ""
    return text
