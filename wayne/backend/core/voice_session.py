from __future__ import annotations

import asyncio
import tempfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad
import whisper


_WHISPER_MODEL = None


def _model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL


class VoiceSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.is_speaking = False
        self.is_listening = True
        self.current_stream: asyncio.Task | None = None
        self.audio_buffer = b""
        self.vad = webrtcvad.Vad(2)
        self.sample_rate = 16000
        self.frame_ms = 30
        self.silence_ms = 0
        self.has_speech = False
        self._lock = asyncio.Lock()

    def process_audio_chunk(self, chunk: bytes) -> bool:
        self.audio_buffer += chunk
        frame_size = int(self.sample_rate * self.frame_ms / 1000) * 2
        if len(chunk) < frame_size:
            return False

        speech_detected = False
        for offset in range(0, len(chunk) - frame_size + 1, frame_size):
            frame = chunk[offset : offset + frame_size]
            try:
                if self.vad.is_speech(frame, self.sample_rate):
                    speech_detected = True
                    break
            except Exception:
                speech_detected = True
                break

        if speech_detected:
            self.has_speech = True
            self.silence_ms = 0
            return False

        if self.has_speech:
            self.silence_ms += self.frame_ms
        return self.has_speech and self.silence_ms >= 800

    async def transcribe(self) -> str:
        async with self._lock:
            audio = self.audio_buffer
            self.audio_buffer = b""
            self.silence_ms = 0
            self.has_speech = False

        if not audio:
            return ""

        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: bytes) -> str:
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = handle.name

            if self._looks_like_pcm(audio):
                samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                sf.write(temp_path, samples, self.sample_rate)
            else:
                Path(temp_path).write_bytes(audio)

            result = _model().transcribe(temp_path, fp16=False)
            return str(result.get("text", "")).strip()
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def _looks_like_pcm(self, audio: bytes) -> bool:
        if audio.startswith(b"RIFF") or audio.startswith(b"\x1aE\xdf\xa3") or audio.startswith(b"OggS"):
            return False
        return len(audio) % 2 == 0

    def cancel_stream(self) -> None:
        if self.current_stream is not None and not self.current_stream.done():
            self.current_stream.cancel()
        self.current_stream = None
        self.is_speaking = False
        self.is_listening = True

    def export_wav(self, path: str) -> None:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.audio_buffer)
