# NOVA-CLI

A basic Command-Line Interface (CLI) named "Nova".

## Quick Start
...
## AI Features

Nova supports two AI modes:
1. **Local (Ollama)**: Runs offline, private, and fast.
2. **Cloud (Hugging Face)**: Access powerful models like Gemma, Mistral, and Phi-3 without local hardware limits.

### 1. Local Setup (Ollama)
1. Install [Ollama](https://ollama.com).
2. Run `ollama pull tinyllama`.
3. Start Nova: `python -m nova` (Default mode).

### 2. Cloud Setup (Hugging Face)
 To use cloud models, you need a free API Token.

1. **Get your Token**:
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Log in or Sign up.
   - Click **"Create new token"**.
   - Name it "Nova" and select **"Read"** permission.
   - Copy the token (starts with `hf_...`).

2. **Set the Environment Variable**:
   
   **PowerShell (Windows)**:
   ```powershell
   $env:HF_API_TOKEN = "your_token_here"
   ```
   
   **CMD (Windows)**:
   ```cmd
   set HF_API_TOKEN=your_token_here
   ```
   
   **Mac/Linux**:
   ```bash
   export HF_API_TOKEN=your_token_here
   ```

3. **Use it in Nova**:
   ```bash
   python -m nova
   # Inside Nova:
   /model                        # See all available models
   /model 1                      # Switch to Google Gemma 2B
   /model 5                      # Switch to Microsoft Phi-3
   /model 7                      # Switch back to Local Ollama
   /model some/other-model       # Use any HF model ID
   ```
## 🧠 Neural Interaction Engine (NIE)
Nova now includes the **Neural Interaction Engine**, allowing it to react to your requests and control your Windows laptop:

- **System Control**: `Set volume to 50%`, `Set brightness to 20%`.
- **Browsing**: `Open YouTube`, `Search Google for Nova CLI`.
- **Productivity**: `Take a screenshot`, `Open Notepad`, `Run dir`.
- **System Awareness**: `What is my battery level?`.

---

## Installation
...