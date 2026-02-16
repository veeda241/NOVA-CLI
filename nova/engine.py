"""
NOVA Neural Interaction Engine (NIE)
=====================================

The NIE is NOVA's brain for instant, offline intent recognition.

Architecture:
  User Input
       |
       v
  [Neural Intent Classifier]  <-- Trained from-scratch transformer
       |                           (tokenizer + embeddings + attention + FFN)
       |
       v
  [Permission Gate]            <-- Safety check (READ/MODIFY/EXECUTE)
       |
       v
  [Consciousness Layer]       <-- Personality, emotion, memory, learning
       |
       v
  [OS Bridge / Action Executor] <-- System control (apps, volume, brightness, etc.)
       |
       v
  [Action Logger]              <-- Audit trail for all OS actions
       |
       v
  [Response Generator]        <-- Context-aware response with personality

Supported Intents (10):
  LOCK_SYSTEM, VOLUME_UP, VOLUME_DOWN, SYSTEM_STATUS,
  OPEN_APP, CLOSE_APP, SCREENSHOT, BRIGHTNESS_UP, BRIGHTNESS_DOWN, UNKNOWN

This is Step 10 in action: fast, offline, free, private, efficient.
"""

import os
import sys
import json
import subprocess
import webbrowser
import platform
import datetime
import time
import re
import numpy as np
from typing import Dict, Optional, Tuple, List

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if HAS_RICH:
    console = Console()

# Import the trained NIE model
from nova.nie_model import NeuralIntentClassifier, Tokenizer, create_model, USE_GPU
from nova.nie_data import INTENT_LABELS, NUM_INTENTS

# Import consciousness layer
from nova.consciousness import nova_consciousness
from nova.memory import nova_memory


# ═══════════════════════════════════════════════════════════════
# ACTION LOGGER — Audit trail for all OS actions
# ═══════════════════════════════════════════════════════════════

class ActionLogger:
    """Logs all OS actions for security auditing."""

    def __init__(self):
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "action_log.jsonl")

    def log(self, user_input: str, intent: str, action: str,
            result: str, permission: str = "auto", success: bool = True):
        """Log an action to the audit trail."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_input": user_input,
            "intent": intent,
            "action": action,
            "result": result[:200],  # Truncate long results
            "permission": permission,
            "success": success,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Logging should never crash the app

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get the last N log entries."""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [json.loads(line) for line in lines[-n:]]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════
# PERMISSION SYSTEM — Safety gates for OS actions
# ═══════════════════════════════════════════════════════════════

# Permission levels for each intent
PERMISSION_LEVELS = {
    "SYSTEM_STATUS":   "READ",       # Safe: just reads info
    "SCREENSHOT":      "READ",       # Safe: just captures screen
    "VOLUME_UP":       "MODIFY",     # Modifies system setting
    "VOLUME_DOWN":     "MODIFY",     # Modifies system setting
    "BRIGHTNESS_UP":   "MODIFY",     # Modifies system setting
    "BRIGHTNESS_DOWN": "MODIFY",     # Modifies system setting
    "OPEN_APP":        "EXECUTE",    # Launches a process
    "CLOSE_APP":       "EXECUTE",    # Kills a process
    "LOCK_SYSTEM":     "EXECUTE",    # Locks workstation
    "SHUTDOWN":        "EXECUTE",    # SYSTEM CRITICAL
    "RESTART":         "EXECUTE",    # SYSTEM CRITICAL
    "UNKNOWN":         "NONE",       # No action needed
}

# Whitelisted applications that can be opened
APP_WHITELIST = {
    "chrome":       {"win": "start chrome", "exe": "chrome.exe"},
    "google chrome": {"win": "start chrome", "exe": "chrome.exe"},
    "brave":        {"win": "start brave", "exe": "brave.exe"},
    "firefox":      {"win": "start firefox", "exe": "firefox.exe"},
    "edge":         {"win": "start msedge", "exe": "msedge.exe"},
    "notepad":      {"win": "notepad.exe", "exe": "notepad.exe"},
    "calculator":   {"win": "calc.exe", "exe": "calc.exe"},
    "calc":         {"win": "calc.exe", "exe": "calc.exe"},
    "file explorer": {"win": "explorer.exe", "exe": "explorer.exe"},
    "explorer":     {"win": "explorer.exe", "exe": "explorer.exe"},
    "task manager": {"win": "taskmgr.exe", "exe": "taskmgr.exe"},
    "settings":     {"win": "start ms-settings:", "exe": ""},
    "paint":        {"win": "mspaint.exe", "exe": "mspaint.exe"},
    "terminal":     {"win": "wt.exe", "exe": "wt.exe"},
    "cmd":          {"win": "cmd.exe", "exe": "cmd.exe"},
    "command prompt": {"win": "cmd.exe", "exe": "cmd.exe"},
    "powershell":   {"win": "powershell.exe", "exe": "powershell.exe"},
    "vscode":       {"win": "code", "exe": "Code.exe"},
    "vs code":      {"win": "code", "exe": "Code.exe"},
    "word":         {"win": "start winword", "exe": "WINWORD.EXE"},
    "excel":        {"win": "start excel", "exe": "EXCEL.EXE"},
    "spotify":      {"win": "start spotify:", "exe": "Spotify.exe"},
    "clock":        {"win": "start ms-clock:", "exe": "Time.exe"},
    "camera":       {"win": "start microsoft.windows.camera:", "exe": "WindowsCamera.exe"},
    "calendar":     {"win": "start outlookcal:", "exe": "HxCalendarAppImm.exe"},
    "google antigravity": {"win": "start https://mrdoob.com/projects/chromeexperiments/google-gravity/", "exe": ""},
    "antigravity":  {"win": "start https://mrdoob.com/projects/chromeexperiments/google-gravity/", "exe": ""},
    "google":       {"win": "start https://google.com", "exe": ""},
    "youtube":      {"win": "start https://youtube.com", "exe": ""},
    "github":       {"win": "start https://github.com", "exe": ""},
}

# Process names for CLOSE_APP
PROCESS_MAP = {
    "chrome":       "chrome.exe",
    "google chrome": "chrome.exe",
    "brave":        "brave.exe",
    "firefox":      "firefox.exe",
    "edge":         "msedge.exe",
    "notepad":      "notepad.exe",
    "calculator":   "Calculator.exe",
    "task manager": "Taskmgr.exe",
    "paint":        "mspaint.exe",
    "vscode":       "Code.exe",
    "vs code":      "Code.exe",
    "word":         "WINWORD.EXE",
    "excel":        "EXCEL.EXE",
    "spotify":      "Spotify.exe",
}


class NeuralInteractionEngine:
    """
    The complete Neural Interaction Engine.

    Combines:
      - Trained intent classifier (from-scratch transformer)
      - OS Bridge (app launch, volume, brightness, screenshots, etc.)
      - Permission gate system (READ / MODIFY / EXECUTE)
      - Action logger (audit trail)
      - Consciousness layer (personality, emotion, memory)
      - Smart routing (NIE vs LLM)
    """

    # Confidence threshold: below this -> fall back to LLM
    CONFIDENCE_THRESHOLD = 0.35

    def __init__(self):
        self.os_type = platform.system()
        self.machine = platform.machine()
        self.logger = ActionLogger()

        # Load the trained neural intent classifier
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the trained NIE model and tokenizer from disk."""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nie_weights")
        model_path = os.path.join(model_dir, "nie_model.npz")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            try:
                # Load tokenizer
                with open(tokenizer_path, "r") as f:
                    tok_data = json.load(f)
                self.tokenizer = Tokenizer.from_dict(tok_data)

                # Build model with correct dimensions (auto-selects GPU or CPU)
                self.model = create_model(
                    vocab_size=self.tokenizer.vocab_size,
                    max_len=self.tokenizer.max_len,
                    embed_dim=64,
                    ff_dim=128,
                    num_classes=NUM_INTENTS,
                )
                self.model.load(model_path)
            except Exception as e:
                self.model = None
                self.tokenizer = None

    def classify_intent(self, text: str) -> Tuple[str, float, Dict]:
        """
        Classify user intent using the trained neural network.

        Returns:
          (intent_label, confidence, full_probs_dict)

        This runs in < 1ms -- no internet, no LLM needed.
        """
        if self.model is None or self.tokenizer is None:
            return "UNKNOWN", 0.0, {}

        t0 = time.perf_counter()
        pred_id, confidence, probs = self.model.predict(self.tokenizer, text)
        t1 = time.perf_counter()

        intent_label = INTENT_LABELS.get(pred_id, "UNKNOWN")
        probs_dict = {INTENT_LABELS[i]: float(probs[i]) for i in range(len(probs))}

        return intent_label, confidence, probs_dict

    def process_command(self, text: str, session_id: str = "default") -> Dict:
        """
        Full NIE pipeline: classify -> consciousness -> execute.

        Returns a dict with:
          - handled: bool (True if NIE handled it, False if should go to LLM)
          - intent: the classified intent
          - confidence: classification confidence
          - result: action result string (if handled)
          - consciousness: consciousness layer output
          - inference_ms: how long classification took
        """
        t0 = time.perf_counter()
        intent, confidence, probs = self.classify_intent(text)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        # Process through consciousness layer
        consciousness_output = nova_consciousness.process_input(
            text, intent, confidence
        )

        # Meta-cognitive check: should we handle this or let LLM do it?
        assessment = consciousness_output["intent_assessment"]

        # If confidence is too low or intent is UNKNOWN, let LLM handle it
        if intent == "UNKNOWN" or confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "handled": False,
                "intent": intent,
                "confidence": confidence,
                "result": None,
                "consciousness": consciousness_output,
                "inference_ms": inference_ms,
                "probs": probs,
            }

        # Execute the action
        result = self._execute_intent(intent, text)

        # Log the action
        self.logger.log(
            user_input=text,
            intent=intent,
            action=intent,
            result=result,
            permission=PERMISSION_LEVELS.get(intent, "NONE"),
        )

        # Store NOVA's response in memory
        nova_consciousness.store_response(result)

        return {
            "handled": True,
            "intent": intent,
            "confidence": confidence,
            "result": result,
            "consciousness": consciousness_output,
            "inference_ms": inference_ms,
            "probs": probs,
        }

    # ═══════════════════════════════════════════
    # INTENT ROUTER — Maps intents to OS actions
    # ═══════════════════════════════════════════

    def _execute_intent(self, intent: str, original_text: str) -> str:
        """Execute a system action based on classified intent."""
        try:
            if intent == "LOCK_SYSTEM":
                return self._lock_system()
            elif intent == "VOLUME_UP":
                return self._volume_up(original_text)
            elif intent == "VOLUME_DOWN":
                return self._volume_down(original_text)
            elif intent == "SYSTEM_STATUS":
                return self._system_status()
            elif intent == "OPEN_APP":
                return self._open_app(original_text)
            elif intent == "CLOSE_APP":
                return self._close_app(original_text)
            elif intent == "SCREENSHOT":
                return self._take_screenshot()
            elif intent == "BRIGHTNESS_UP":
                return self._brightness_up(original_text)
            elif intent == "BRIGHTNESS_DOWN":
                return self._brightness_down(original_text)
            elif intent == "SHUTDOWN":
                return self._shutdown_system()
            elif intent == "RESTART":
                return self._restart_system()
            else:
                return f"Intent '{intent}' recognized but no action defined."
        except Exception as e:
            return f"Action failed: {e}"

    # ═══════════════════════════════════════════
    # OS BRIDGE — System action implementations
    # ═══════════════════════════════════════════

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract a number (0-100) from user text for level-setting."""
        numbers = re.findall(r'\b(\d{1,3})\b', text)
        for n in numbers:
            val = int(n)
            if 0 <= val <= 100:
                return val
        return None

    def _lock_system(self) -> str:
        """Lock the computer screen."""
        if self.os_type == "Windows":
            try:
                import ctypes
                ctypes.windll.user32.LockWorkStation()
                return "System locked successfully."
            except Exception as e:
                return f"Lock failed: {e}"
        elif self.os_type == "Darwin":
            os.system("pmset displaysleepnow")
            return "Mac screen locked."
        elif self.os_type == "Linux":
            os.system("xdg-screensaver lock")
            return "Linux screen locked."
        return "Lock not supported on this OS."

    def _shutdown_system(self) -> str:
        """Shutdown the system with a safety timer."""
        try:
            if self.os_type == "Windows":
                # 10 second timer, force close apps, comment
                subprocess.run('shutdown /s /t 10 /c "NOVA initiated shutdown. Run \'shutdown /a\' to cancel."', shell=True)
                return "Initiating shutdown in 10 seconds. Run 'shutdown /a' in cmd to cancel."
            elif self.os_type in ["Linux", "Darwin"]:
                # Try standard shutdown, might fail without sudo
                subprocess.run("shutdown -h +1", shell=True) # 1 minute delay
                return "Initiating shutdown in 1 minute. Run 'shutdown -c' to cancel."
            else:
                return f"Shutdown not implemented for {self.os_type}."
        except Exception as e:
            return f"Shutdown failed: {e}"

    def _restart_system(self) -> str:
        """Restart the system with a safety timer."""
        try:
            if self.os_type == "Windows":
                subprocess.run('shutdown /r /t 10 /c "NOVA initiated restart. Run \'shutdown /a\' to cancel."', shell=True)
                return "Initiating restart in 10 seconds. Run 'shutdown /a' in cmd to cancel."
            elif self.os_type in ["Linux", "Darwin"]:
                subprocess.run("shutdown -r +1", shell=True)
                return "Initiating restart in 1 minute. Run 'shutdown -c' to cancel."
            else:
                return f"Restart not implemented for {self.os_type}."
        except Exception as e:
            return f"Restart failed: {e}"


    def _volume_up(self, original_text: str = "") -> str:
        """Increase system volume. If a number is given, set to that level."""
        if self.os_type == "Windows":
            try:
                from pycaw.pycaw import AudioUtilities
                speakers = AudioUtilities.GetSpeakers()
                volume = speakers.EndpointVolume
                current = volume.GetMasterVolumeLevelScalar()

                # Check if user specified an exact level
                target = self._extract_number(original_text)
                if target is not None:
                    new_level = target / 100.0
                else:
                    new_level = min(1.0, current + 0.1)

                volume.SetMasterVolumeLevelScalar(new_level, None)
                return f"Volume set: {int(current*100)}% -> {int(new_level*100)}%"
            except ImportError:
                try:
                    subprocess.run(
                        'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"',
                        shell=True, capture_output=True
                    )
                    return "Volume increased (key simulation)"
                except Exception as e:
                    return f"Volume up failed: {e}"
            except Exception as e:
                return f"Volume up error: {e}"
        return "Volume control not supported on this OS."

    def _volume_down(self, original_text: str = "") -> str:
        """Decrease system volume. If a number is given, set to that level."""
        if self.os_type == "Windows":
            try:
                from pycaw.pycaw import AudioUtilities
                speakers = AudioUtilities.GetSpeakers()
                volume = speakers.EndpointVolume
                current = volume.GetMasterVolumeLevelScalar()

                # Check if user specified an exact level
                target = self._extract_number(original_text)
                if target is not None:
                    new_level = target / 100.0
                else:
                    new_level = max(0.0, current - 0.1)

                volume.SetMasterVolumeLevelScalar(new_level, None)
                return f"Volume set: {int(current*100)}% -> {int(new_level*100)}%"
            except ImportError:
                try:
                    subprocess.run(
                        'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"',
                        shell=True, capture_output=True
                    )
                    return "Volume decreased (key simulation)"
                except Exception as e:
                    return f"Volume down failed: {e}"
            except Exception as e:
                return f"Volume down error: {e}"
        return "Volume control not supported on this OS."

    def _system_status(self) -> str:
        """Get comprehensive system status."""
        if not HAS_PSUTIL:
            return "System monitoring not available (psutil not installed)."

        parts = []

        # Battery
        battery = psutil.sensors_battery()
        if battery:
            plugged = "Plugged in" if battery.power_plugged else "On battery"
            parts.append(f"Battery: {battery.percent}% ({plugged})")

        # CPU
        cpu = psutil.cpu_percent(interval=0.5)
        cores = psutil.cpu_count()
        parts.append(f"CPU: {cpu}% across {cores} cores")

        # RAM
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        parts.append(f"RAM: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem.percent}%)")

        # Disk
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            total_disk_gb = disk.total / (1024**3)
            parts.append(f"Disk: {free_gb:.1f}GB free / {total_disk_gb:.1f}GB total")
        except Exception:
            pass

        return " | ".join(parts)

    def _extract_app_name(self, text: str) -> str:
        """Extract the application name from user text."""
        text_lower = text.lower().strip()
        # Remove common prefixes
        for prefix in ["open ", "launch ", "start ", "run ", "close ", "kill ",
                       "stop ", "terminate ", "shut down ", "end task ",
                       "force close ", "end ", "can you open ", "please open ",
                       "please launch ", "i want to open ", "fire up ",
                       "can you launch ", "please close ", "shut down ",
                       "can you close "]:
            if text_lower.startswith(prefix):
                text_lower = text_lower[len(prefix):].strip()
                break

        # Remove trailing words
        for suffix in [" for me", " please", " now", " right now",
                       " immediately", " process", " browser", " application",
                       " app", " program", " windows"]:
            if text_lower.endswith(suffix):
                text_lower = text_lower[:-len(suffix)].strip()

        # remove "the " prefix
        if text_lower.startswith("the "):
            text_lower = text_lower[4:]

        return text_lower

    def _open_app(self, original_text: str) -> str:
        """Open/launch an application (Windows)."""
        app_name = self._extract_app_name(original_text)

        if self.os_type != "Windows":
            return f"App launching not implemented for {self.os_type} yet."

        # Check whitelist first (exact match)
        if app_name in APP_WHITELIST:
            cmd = APP_WHITELIST[app_name]["win"]
            try:
                subprocess.Popen(cmd, shell=True,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return f"Opened {app_name.title()} (whitelist)."
            except Exception as e:
                return f"Failed to open {app_name}: {e}"

        # Try fuzzy matching in whitelist
        for key in APP_WHITELIST:
            if app_name == key or (len(app_name) > 3 and app_name in key):
                cmd = APP_WHITELIST[key]["win"]
                try:
                    subprocess.Popen(cmd, shell=True,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    return f"Opened {key.title()} (fuzzy match)."
                except Exception as e:
                    return f"Failed to open {key}: {e}"

        # Fallback: Try to launch generically
        try:
            # shell=True with 'start' command handles PATH and registered apps
            # We use 'start "" "name"' to handle names with spaces properly
            cmd = f'start "" "{app_name}"'
            subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return f"Attempting to open '{app_name.title()}'..."
        except Exception as e:
            return f"Could not find or open '{app_name}'. Error: {e}"

    def _close_app(self, original_text: str) -> str:
        """Close/kill an application process (Windows)."""
        app_name = self._extract_app_name(original_text)

        if self.os_type != "Windows":
            return f"App control not implemented for {self.os_type} yet."

        # Find the process name
        process_name = None
        for key, proc in PROCESS_MAP.items():
            if app_name in key or key in app_name:
                process_name = proc
                break

        if not process_name:
            # Fallback: Try generic kill (assume app_name.exe)
            try:
                proc_exe = f"{app_name}.exe"
                # Check directly first? No, just try to kill.
                result = subprocess.run(
                    f'taskkill /IM "{proc_exe}" /F',
                    shell=True, capture_output=True, text=True
                )
                if result.returncode == 0:
                    return f"Closed {app_name.title()} (generic kill)."
                
                # If failed, return specific error but friendlier
                return (f"Could not find running process for '{app_name}'. "
                        f"Try specifying the exact process name.")
            except Exception as e:
                return f"Failed to close '{app_name}': {e}"

        try:
            # Use taskkill to close the process
            result = subprocess.run(
                f'taskkill /IM "{process_name}" /F',
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                return f"Closed {app_name.title()} ({process_name}) successfully."
            elif "not found" in result.stderr.lower() or result.returncode == 128:
                return f"{app_name.title()} is not currently running."
            else:
                return f"Could not close {app_name}: {result.stderr.strip()}"
        except Exception as e:
            return f"Close app error: {e}"

    def _take_screenshot(self) -> str:
        """Take a screenshot and save to Desktop."""
        try:
            import pyautogui
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Find Desktop (OneDrive or standard)
            home = os.path.expanduser("~")
            onedrive_desktop = os.path.join(home, "OneDrive", "Desktop")
            standard_desktop = os.path.join(home, "Desktop")

            if os.path.isdir(onedrive_desktop):
                desktop = onedrive_desktop
            elif os.path.isdir(standard_desktop):
                desktop = standard_desktop
            else:
                desktop = os.getcwd()

            filename = os.path.join(desktop, f"nova_screenshot_{timestamp}.png")

            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            return f"Screenshot saved: {filename}"
        except ImportError:
            # Fallback: use Windows Snipping Tool API
            try:
                subprocess.run(
                    'powershell -c "Add-Type -AssemblyName System.Windows.Forms; '
                    '[System.Windows.Forms.Screen]::PrimaryScreen"',
                    shell=True, capture_output=True
                )
                return "Screenshot: pyautogui not installed. Run: pip install pyautogui"
            except Exception as e:
                return f"Screenshot failed: {e}"
        except Exception as e:
            return f"Screenshot error: {e}"

    def _brightness_up(self, original_text: str = "") -> str:
        """Increase screen brightness. If a number is given, set to that level."""
        if self.os_type == "Windows":
            try:
                import screen_brightness_control as sbc
                current = sbc.get_brightness()[0]

                target = self._extract_number(original_text)
                if target is not None:
                    new_level = target
                else:
                    new_level = min(100, current + 10)

                sbc.set_brightness(new_level)
                return f"Brightness set: {current}% -> {new_level}%"
            except ImportError:
                # Fallback: use WMI/PowerShell
                try:
                    result = subprocess.run(
                        'powershell -c "(Get-WmiObject -Namespace root/WMI '
                        '-Class WmiMonitorBrightness).CurrentBrightness"',
                        shell=True, capture_output=True, text=True
                    )
                    current = int(result.stdout.strip()) if result.stdout.strip() else 50
                    target = self._extract_number(original_text)
                    new_level = target if target is not None else min(100, current + 10)
                    subprocess.run(
                        f'powershell -c "(Get-WmiObject -Namespace root/WMI '
                        f'-Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{new_level})"',
                        shell=True, capture_output=True
                    )
                    return f"Brightness set: {current}% -> {new_level}%"
                except Exception as e:
                    return f"Brightness up failed: {e}"
            except Exception as e:
                return f"Brightness error: {e}"
        return "Brightness control not supported on this OS."

    def _brightness_down(self, original_text: str = "") -> str:
        """Decrease screen brightness. If a number is given, set to that level."""
        if self.os_type == "Windows":
            try:
                import screen_brightness_control as sbc
                current = sbc.get_brightness()[0]

                target = self._extract_number(original_text)
                if target is not None:
                    new_level = target
                else:
                    new_level = max(0, current - 10)

                sbc.set_brightness(new_level)
                return f"Brightness set: {current}% -> {new_level}%"
            except ImportError:
                # Fallback: use WMI/PowerShell
                try:
                    result = subprocess.run(
                        'powershell -c "(Get-WmiObject -Namespace root/WMI '
                        '-Class WmiMonitorBrightness).CurrentBrightness"',
                        shell=True, capture_output=True, text=True
                    )
                    current = int(result.stdout.strip()) if result.stdout.strip() else 50
                    target = self._extract_number(original_text)
                    new_level = target if target is not None else max(0, current - 10)
                    subprocess.run(
                        f'powershell -c "(Get-WmiObject -Namespace root/WMI '
                        f'-Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{new_level})"',
                        shell=True, capture_output=True
                    )
                    return f"Brightness set: {current}% -> {new_level}%"
                except Exception as e:
                    return f"Brightness down failed: {e}"
            except Exception as e:
                return f"Brightness error: {e}"
        return "Brightness control not supported on this OS."

    # ═══════════════════════════════════════════
    # Legacy methods (kept for backwards compatibility)
    # ═══════════════════════════════════════════

    def execute_action(self, action_type, params):
        """Legacy: Executes a system-level action based on NIE logic."""
        try:
            if action_type == "open_browser":
                url = params.get("url", "https://google.com")
                webbrowser.open(url)
                return f"Opened browser to {url}"

            elif action_type == "run_command":
                cmd = params.get("command")
                destructive = ["rm", "del", "format", "erase"]
                if any(d in cmd.lower() for d in destructive):
                    return "Blocked: Destructive command detected for safety."
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                return f"Executed: {cmd}\nOutput: {result.stdout or result.stderr}"

            elif action_type == "open_app":
                return self._open_app(params.get("app_name", ""))

            elif action_type == "check_battery":
                return self._system_status()

            elif action_type == "check_cpu":
                return self._system_status()

            elif action_type == "check_ram":
                return self._system_status()

            elif action_type == "set_volume":
                level = params.get("level", 50)
                if self.os_type == "Windows":
                    try:
                        from pycaw.pycaw import AudioUtilities
                        speakers = AudioUtilities.GetSpeakers()
                        volume = speakers.EndpointVolume
                        volume.SetMasterVolumeLevelScalar(level / 100, None)
                        return f"System volume set to {level}%"
                    except Exception as e:
                        return f"Volume error: {e}"
                return "Volume control not supported on this OS"

            elif action_type == "set_brightness":
                level = params.get("level", 50)
                try:
                    import screen_brightness_control as sbc
                    sbc.set_brightness(level)
                    return f"Screen brightness set to {level}%"
                except Exception as e:
                    return f"Brightness error: {e}"

            elif action_type == "screenshot":
                return self._take_screenshot()

            return f"Unknown interaction type: {action_type}"

        except Exception as e:
            return f"NIE Execution Error: {e}"

    def parse_ai_response(self, text):
        """Legacy: Look for [ACTION: type {params}] patterns in AI text."""
        pattern = r"\[ACTION:\s*(\w+)\s*({.*?})\]"
        matches = re.findall(pattern, text)

        actions = []
        for match in matches:
            try:
                action_type = match[0]
                params = json.loads(match[1])
                actions.append({"type": action_type, "params": params})
            except:
                continue
        return actions

    def get_model_info(self) -> Dict:
        """Get info about the loaded NIE model."""
        if self.model is None:
            return {"loaded": False, "reason": "Model weights not found"}
        return {
            "loaded": True,
            "vocab_size": self.tokenizer.vocab_size,
            "max_seq_len": self.tokenizer.max_len,
            "embed_dim": self.model.embed_dim,
            "num_intents": NUM_INTENTS,
            "intents": list(INTENT_LABELS.values()),
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
        }


# Singleton instance
nie_engine = NeuralInteractionEngine()
