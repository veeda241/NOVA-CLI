from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

WINDOWS_PYTHON311 = Path(r"C:\Program Files\Python311\python.exe")
if (
    os.name == "nt"
    and sys.version_info >= (3, 13)
    and WINDOWS_PYTHON311.exists()
    and Path(sys.executable).resolve() != WINDOWS_PYTHON311.resolve()
    and os.getenv("WAYNE_NO_REEXEC") != "1"
):
    os.environ["WAYNE_NO_REEXEC"] = "1"
    raise SystemExit(subprocess.call([str(WINDOWS_PYTHON311), *sys.argv]))

import httpx
from dotenv import load_dotenv

WAYNE_DIR = Path(__file__).resolve().parents[1]
if str(WAYNE_DIR) not in sys.path:
    sys.path.insert(0, str(WAYNE_DIR))

from startup.boot_sequence import BootSequence
from startup.wake_word import FallbackWakeWordDetector, ManualWakeWordDetector, WakeWordDetector

load_dotenv(WAYNE_DIR / "backend" / ".env", override=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
if "your-local-ip" in BACKEND_URL:
    BACKEND_URL = "http://localhost:8000"


def _service_python() -> str:
    configured = os.getenv("WAYNE_PYTHON")
    if configured:
        return configured
    if sys.version_info >= (3, 13) and WINDOWS_PYTHON311.exists():
        return str(WINDOWS_PYTHON311)
    return sys.executable


class WAYNEDaemon:
    def __init__(self) -> None:
        self.backend_process: subprocess.Popen[bytes] | None = None
        self.laptop_agent_process: subprocess.Popen[bytes] | None = None
        self.ollama_process: subprocess.Popen[bytes] | None = None
        self.is_initialized = False
        self.boot_sequence = BootSequence()
        self.wake_detector: WakeWordDetector | FallbackWakeWordDetector | ManualWakeWordDetector | None = None

    def start(self) -> None:
        print("[W.A.Y.N.E DAEMON] Starting background services...")
        self._start_ollama()
        self._start_backend()
        self._start_laptop_agent()
        self._start_wake_word()
        print("[W.A.Y.N.E DAEMON] All services running. Listening for wake word.")
        self._keep_alive()

    def _start_ollama(self) -> None:
        try:
            if self._ollama_ready():
                print("[W.A.Y.N.E DAEMON] Ollama service already running.")
                return
            self.ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            print("[W.A.Y.N.E DAEMON] Ollama service started.")
        except Exception as exc:
            print(f"[W.A.Y.N.E DAEMON] Ollama error: {exc}")

    def _ollama_ready(self) -> bool:
        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _start_backend(self) -> None:
        if self._backend_ready():
            print("[W.A.Y.N.E DAEMON] FastAPI backend already running.")
            return
        backend_path = WAYNE_DIR / "backend"
        log_dir = WAYNE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        stdout = open(log_dir / "backend_stdout.log", "a", encoding="utf-8")
        stderr = open(log_dir / "backend_stderr.log", "a", encoding="utf-8")
        python_executable = _service_python()
        self.backend_process = subprocess.Popen(
            [python_executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=backend_path,
            stdout=stdout,
            stderr=stderr,
        )
        time.sleep(2)
        if self.backend_process.poll() is None:
            print("[W.A.Y.N.E DAEMON] FastAPI backend started.")
        else:
            print("[W.A.Y.N.E DAEMON] FastAPI backend failed. See logs/backend_stderr.log.")

    def _backend_ready(self) -> bool:
        try:
            response = httpx.get(f"{BACKEND_URL}/system/status", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _start_laptop_agent(self) -> None:
        agent_path = WAYNE_DIR / "backend" / "device" / "laptop_agent.py"
        self.laptop_agent_process = subprocess.Popen([_service_python(), str(agent_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[W.A.Y.N.E DAEMON] Laptop agent started.")

    def _start_wake_word(self) -> None:
        try:
            detector = WakeWordDetector(on_wake_callback=self.on_wake_detected)
            detector.initialize()
            if detector.use_fallback:
                raise RuntimeError("Porcupine unavailable; using fallback.")
            self.wake_detector = detector
        except Exception as porcupine_error:
            try:
                self.wake_detector = FallbackWakeWordDetector(on_wake_callback=self.on_wake_detected)
            except Exception as fallback_error:
                reason = f"{porcupine_error}; fallback unavailable: {fallback_error}"
                self.wake_detector = ManualWakeWordDetector(on_wake_callback=self.on_wake_detected, reason=reason)
        thread = threading.Thread(target=self.wake_detector.listen_loop, daemon=True)
        thread.start()
        print("[W.A.Y.N.E DAEMON] Wake word detection active.")

    def on_wake_detected(self) -> None:
        if self.is_initialized:
            return
        self.is_initialized = True
        print("\n[W.A.Y.N.E] Wake word detected. Initializing...")
        self.boot_sequence.run_cli()
        self._notify_web("wake")
        self._open_cli()
        threading.Timer(30.0, self._reset_initialized).start()

    def _notify_web(self, event_type: str) -> None:
        path = "/wayne/wake" if event_type == "wake" else "/wayne/sleep"
        try:
            httpx.post(f"{BACKEND_URL}{path}", json={"source": "wake_word"}, timeout=3.0)
        except Exception:
            pass

    def _open_cli(self) -> None:
        wayne_cli = WAYNE_DIR / "cli" / "wayne.py"
        if sys.platform == "darwin":
            subprocess.Popen(["osascript", "-e", f'tell app "Terminal" to do script "python3 {wayne_cli}"'])
        elif sys.platform == "win32":
            subprocess.Popen(["cmd", "/c", "start", "W.A.Y.N.E", "cmd", "/k", f'python "{wayne_cli}"'])
        else:
            terminal = os.getenv("TERMINAL", "x-terminal-emulator")
            subprocess.Popen([terminal, "-e", f'python3 "{wayne_cli}"'])

    def _reset_initialized(self) -> None:
        self.is_initialized = False

    def _keep_alive(self) -> None:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[W.A.Y.N.E DAEMON] Shutting down.")
            self._cleanup()

    def _cleanup(self) -> None:
        if self.wake_detector:
            self.wake_detector.cleanup()
        if self.backend_process:
            self.backend_process.terminate()
        if self.laptop_agent_process:
            self.laptop_agent_process.terminate()


if __name__ == "__main__":
    WAYNEDaemon().start()
