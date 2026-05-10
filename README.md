# W.A.Y.N.E — Wireless Artificial Yielding Network Engine

> Your personal AI system. Local. Private. Always on.

## What Is W.A.Y.N.E?

W.A.Y.N.E (Wireless Artificial Yielding Network Engine) is a fully local personal AI assistant. It runs Gemma through Ollama on your laptop, controls your devices, manages your schedule, reads your files, and listens to your voice — with zero cloud dependency and zero API costs.

Interfaces:

- CLI terminal app (`wayne.py`)
- Web dashboard (`localhost:3000`)
- Native iOS app (`W.A.Y.N.E` on iPhone)

Powered by: Gemma via Ollama | Whisper | FastAPI | Supabase | SwiftUI

W.A.Y.N.E is a local-first assistant with four interfaces:

- Python CLI for terminal workflows
- Next.js web app for browser control
- SwiftUI iOS app for iPhone control over local WiFi
- Real-time device control, tracking, and full duplex live voice

All interfaces use the same FastAPI backend, Supabase Postgres or local SQLite database, local Whisper transcription, and a local Ollama model running through Ollama. No external AI API key is needed for responses.

## Prerequisites

- Python 3.11+
- Node.js 20.9+
- Xcode 15+
- iOS 16+ device or simulator
- Ollama
- Google Calendar API credentials, optional
- Apple Developer account for APNs push notifications, optional
- Picovoice access key, optional, for low-power Porcupine wake word detection

## Ollama Setup

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On Windows, download the installer from `https://ollama.com`.

Pull the local model:

```bash
ollama pull gemma3:4b
```

Verify:

```bash
ollama run gemma3:4b "Hello WAYNE"
```

Ollama exposes its local API at `http://localhost:11434`.

## Setup

1. Enter the assistant directory:

   ```bash
   cd wayne
   ```

2. Copy the environment template:

   ```bash
   cp backend/.env.example backend/.env
   ```

3. Set `BASE_FILE_DIRECTORY` and `BACKEND_URL` in `backend/.env`.

4. Set `DATABASE_URL` in `backend/.env`. Use SQLite for local-only testing, or Supabase Postgres for cloud sync.

5. Optional: set `PICOVOICE_ACCESS_KEY` in `backend/.env` for Porcupine wake detection. Without it, W.A.Y.N.E falls back to SpeechRecognition.

6. Start Ollama, backend, laptop agent, and web:

   ```bash
   chmod +x start.sh
   ./start.sh
   ```

7. Open `ios/WAYNE/WAYNE.xcodeproj` in Xcode.

8. Set `backendURL` and `backendWebSocketURL` in `ios/WAYNE/Config.swift` to your laptop IP.

9. Build and run the iOS app on your iPhone.

## Supabase Database Setup

W.A.Y.N.E stores its learning, device, file, task, and history data in Supabase Postgres when `DATABASE_URL` is configured. Local SQLite remains available for offline development.

1. Open your Supabase project dashboard.
2. Go to `Project Settings` -> `Database` -> `Connection string`.
3. Choose the pooled connection string, usually port `6543`.
4. Copy the URI and replace `[YOUR-PASSWORD]` with your real database password.
5. Put it in `wayne/backend/.env`:

```env
DATABASE_URL=postgresql://postgres.PROJECT_REF:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres?sslmode=require
```

If your password has special characters like `@`, `#`, `:` or `/`, URL-encode it first. For example, `@` becomes `%40`.

To verify the connection after starting the backend, open:

```text
http://localhost:8000/database/status
```

You should see:

```json
{"status":"connected","backend":"supabase-postgres"}
```

## Reinforcement Learning System

W.A.Y.N.E is built around a reinforcement learning control loop. Every chat, voice turn, task, file access, device command, and feedback event is treated as an agent step: W.A.Y.N.E observes the current state, selects a response policy, acts through Gemma and tools, stores the experience, scores the outcome, and updates its preference model.

RL modules live in `wayne/backend/core/rl/`:

- `agent.py` coordinates observe, act, learn, memory, habits, and self-improvement.
- `state_builder.py` builds the current user state from time, tasks, files, contacts, habits, devices, and history.
- `reward_engine.py` combines explicit ratings, implicit follow-up signals, timing, emotion, and voice feedback.
- `policy.py` builds the adaptive system prompt and retrieves high-reward examples.
- `memory_bank.py` stores experience replay data.
- `preference_model.py`, `behavior_tracker.py`, `habit_learner.py`, `emotion_detector.py`, and `intent_classifier.py` continuously personalize W.A.Y.N.E.

Learning endpoints:

- `POST /feedback` records explicit 1-5 star feedback.
- `POST /feedback/implicit` scores the previous response from the next user turn.
- `POST /feedback/voice` turns phrases like "good answer" or "wrong" into reward signals.
- `GET /learning/stats` returns the learning score, topics, preferences, emotions, heatmap, files, contacts, habits, and task completion.
- `GET /learning/preferences` returns the learned user profile.
- `PATCH /learning/preferences` manually overrides a preference.
- `GET /learning/history` returns recent scored interactions.
- `GET /learning/habits` returns detected routines.
- `GET /learning/contacts` returns known people.
- `GET /learning/improvements` returns self-improvement decisions.
- `GET /learning/heatmap` returns day/hour interaction counts.
- `DELETE /learning/reset` clears learned data and resets default preferences.

The web app includes a full `Learning` tab with learning score, personality model, emotion history, activity heatmap, topics, habits, contacts, improvements, export, and reset controls. Star ratings below each W.A.Y.N.E response feed directly into the reward engine. The CLI also prompts for 1-5 feedback after each response.

## Model Selection

The web sidebar includes a local model switcher. It calls:

- `GET /models`
- `POST /models/select`

You can switch between installed Ollama models like `qwen2.5:1.5b`, `llama3`, `mistral`, or `phi3` without restarting.

## Performance Notes

- `gemma3:4b`: Gemma-first local model target for W.A.Y.N.E
- `qwen2.5:1.5b`: fast fallback for low-memory CPU-only runs

Ollama auto-detects supported NVIDIA or AMD GPUs. Check active model/GPU status with:

```bash
ollama ps
```

## Google Calendar Setup

1. Create OAuth 2.0 credentials at `console.cloud.google.com`.
2. Enable the Google Calendar API.
3. Add `http://localhost:3000/auth/callback` as a redirect URI.
4. Fill `GOOGLE_CALENDAR_CLIENT_ID` and `GOOGLE_CALENDAR_CLIENT_SECRET` in `backend/.env`.
5. Visit `http://localhost:3000/auth/google` to authorize.

## Apple Push Setup

1. Create an APNs key in the Apple Developer portal.
2. Download the `.p8` file.
3. Place it at `backend/certs/apns.p8`.
4. Fill `APNS_KEY_ID`, `APNS_TEAM_ID`, and `APNS_BUNDLE_ID` in `backend/.env`.

## Laptop Agent

The laptop agent runs automatically with `start.sh`.

Manual run:

```bash
cd wayne/backend
python device/laptop_agent.py
```

Keep the terminal open, or install it as a system service for always-on tracking and control.

## Always-On Wake Word And Autostart

W.A.Y.N.E can run as a background daemon when the laptop boots. The daemon starts Ollama, the FastAPI backend, the laptop agent, and passive wake word detection.

Manual daemon run:

```bash
cd wayne
python startup/wayne_daemon.py
```

Wake behavior:

- Say `WAYNE` or `Hey WAYNE` to initialize.
- CLI shows the W.A.Y.N.E boot sequence and opens the terminal interface.
- Web clients connected to the backend show the same boot animation and start live voice mode.
- Say `Sleep WAYNE`, `Goodbye WAYNE`, `Standby WAYNE`, or `Goodnight WAYNE` to return to passive mode.

Wake word engine:

- Preferred: Porcupine via `pvporcupine` and `pyaudio`, using built-in `computer` or a custom `wayne_en.ppn`.
- Fallback: `SpeechRecognition`, with offline Sphinx attempted first and Google Web Speech used only if available.
- Set `PICOVOICE_ACCESS_KEY` in `backend/.env` to enable Porcupine.
- Set `WAYNE_WAKE_WORD_PATH=/absolute/path/to/wayne_en.ppn` to use a custom Picovoice wake word file.

macOS launchd install:

```bash
cp startup/autostart/wayne.plist ~/Library/LaunchAgents/com.wayne.daemon.plist
launchctl load ~/Library/LaunchAgents/com.wayne.daemon.plist
```

Edit `/path/to/wayne` inside the plist before loading it.

Linux systemd install:

```bash
sudo cp startup/autostart/wayne.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable wayne
sudo systemctl start wayne
```

Edit `User=YOUR_USERNAME`, `WorkingDirectory`, `ExecStart`, and `EnvironmentFile` before installing.

Windows autostart:

1. Edit `startup/autostart/wayne_startup.bat` and replace `C:\path\to\wayne`.
2. Press `Win+R`, run `shell:startup`, and place a shortcut to the batch file there.
3. For a quieter setup, create a Task Scheduler task at log on that runs `python C:\path\to\wayne\startup\wayne_daemon.py`.

## iOS App Usage

- Chat tab: talk to W.A.Y.N.E by text or voice
- Live Voice: tap the microphone in Chat, speak continuously, interrupt while W.A.Y.N.E is talking, and say "stop" to end
- Devices tab: view laptop and phone status, send power commands
- Tasks tab: manage your to-do list
- Calendar tab: view today's events
- Tracking tab: live battery, CPU, RAM, IP, and location status

The iOS app never talks to Ollama directly. It connects to the FastAPI backend on your laptop over local WiFi.

## CLI

Run CLI only:

```bash
cd wayne
chmod +x start_cli.sh
./start_cli.sh
```

Commands:

```text
/tasks
/files [path]
/schedule
/status
/devices
/shutdown laptop
/restart laptop
/sleep laptop
/lock laptop
/notify "message"
/voice
/clear
/history
/help
/exit
```

## API

REST:

- `POST /chat`
- `GET /models`
- `POST /models/select`
- `GET /tasks`
- `POST /tasks`
- `PATCH /tasks/{id}`
- `DELETE /tasks/{id}`
- `GET /files`
- `POST /files/open`
- `GET /events/today`
- `GET /system/status`
- `POST /device/command`
- `GET /device/status`
- `POST /device/register`
- `POST /device/push`
- `POST /wayne/wake`
- `POST /wayne/sleep`
- `GET /wayne/status`
- `GET /auth/google`
- `GET /auth/callback`

WebSocket:

- `WS /ws/track/{device_id}`
- `WS /ws/commands/{device_id}`
- `WS /ws/chat/{session_id}`
- `WS /ws/voice/{session_id}`
- `WS /ws/wayne/events`

## Live Voice Mode

The backend voice socket accepts microphone audio chunks, detects speech end with VAD, transcribes with local Whisper base, streams local Ollama response tokens, and handles immediate interruption messages.

Web uses browser-native MediaRecorder, Web Speech API, SpeechSynthesis, Web Audio interruption detection, and the animated W.A.Y.N.E orb.

iOS uses AVFoundation, Speech, AVSpeechSynthesizer, URLSessionWebSocketTask, and a native SwiftUI orb interface.

## Offline Mode

The AI is local by default. Offline mode now means the Ollama service is not running. If that happens, responses are prefixed with `[OFFLINE MODE]` and W.A.Y.N.E reports:

```text
W.A.Y.N.E AI core offline. Please start the local engine.
```

Local tools for files, tasks, date/time, and system status remain available.
