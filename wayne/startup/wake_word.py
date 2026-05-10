from __future__ import annotations

import os
import struct
import time
from collections.abc import Callable

PICOVOICE_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "")
CUSTOM_WAKE_WORD = os.getenv("WAYNE_WAKE_WORD_PATH", "")
WAKE_WORDS = ["computer"]


class WakeWordDetector:
    def __init__(self, on_wake_callback: Callable[[], None]) -> None:
        self.on_wake = on_wake_callback
        self.porcupine = None
        self.audio = None
        self.stream = None
        self.use_fallback = False
        self.running = True

    def initialize(self) -> None:
        if not PICOVOICE_KEY:
            self.use_fallback = True
            return

        import pvporcupine
        import pyaudio

        create_args: dict[str, object] = {"access_key": PICOVOICE_KEY}
        if CUSTOM_WAKE_WORD and os.path.exists(CUSTOM_WAKE_WORD):
            create_args["keyword_paths"] = [CUSTOM_WAKE_WORD]
        else:
            create_args["keywords"] = WAKE_WORDS

        self.porcupine = pvporcupine.create(**create_args)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
        )

    def listen_loop(self) -> None:
        if self.use_fallback or self.porcupine is None or self.stream is None:
            raise RuntimeError("Porcupine wake word detector is not initialized.")
        print("[W.A.Y.N.E] Wake word detection active. Say 'WAYNE' to initialize.")
        while self.running:
            pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            frame = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            result = self.porcupine.process(frame)
            if result >= 0:
                self.on_wake()
                time.sleep(1.0)

    def cleanup(self) -> None:
        self.running = False
        if self.stream:
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        if self.porcupine:
            self.porcupine.delete()


class FallbackWakeWordDetector:
    def __init__(self, on_wake_callback: Callable[[], None]) -> None:
        import speech_recognition as sr

        self.on_wake = on_wake_callback
        self.sr = sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.running = True

    def listen_loop(self) -> None:
        print("[W.A.Y.N.E] Passive listening active. Say 'WAYNE' to initialize.")
        with self.sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.running:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    try:
                        text = self.recognizer.recognize_sphinx(audio).lower()
                    except Exception:
                        text = self.recognizer.recognize_google(audio).lower()
                    if "wayne" in text or "hey wayne" in text:
                        self.on_wake()
                        time.sleep(1.0)
                except self.sr.WaitTimeoutError:
                    continue
                except self.sr.UnknownValueError:
                    continue
                except Exception:
                    time.sleep(1.0)

    def cleanup(self) -> None:
        self.running = False


class ManualWakeWordDetector:
    def __init__(self, on_wake_callback: Callable[[], None], reason: str) -> None:
        self.on_wake = on_wake_callback
        self.reason = reason
        self.running = True

    def listen_loop(self) -> None:
        print(f"[W.A.Y.N.E] Wake word audio unavailable: {self.reason}")
        print("[W.A.Y.N.E] Manual wake mode active. Type WAYNE and press Enter to initialize.")
        while self.running:
            try:
                text = input().strip().lower()
            except EOFError:
                time.sleep(1.0)
                continue
            if text in {"wayne", "hey wayne"}:
                self.on_wake()

    def cleanup(self) -> None:
        self.running = False
