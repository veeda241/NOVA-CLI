from __future__ import annotations

from rich.console import Console

console = Console()


def listen() -> str:
    try:
        import speech_recognition as sr
    except ImportError:
        console.print("[red]SpeechRecognition is not installed. Install CLI requirements first.[/red]")
        return ""

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        console.print("[blink yellow]Listening...[/blink yellow]")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=8, phrase_time_limit=20)

    try:
        return recognizer.recognize_google(audio)
    except Exception:
        try:
            return recognizer.recognize_vosk(audio)
        except Exception as exc:
            console.print(f"[red]Voice transcription failed: {exc}[/red]")
            return ""
