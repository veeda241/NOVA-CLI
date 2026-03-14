"""
NOVA Gemini Chat Manager
=========================
Uses Google's Gemini API via the generativelanguage REST endpoint.
No extra SDK needed — just requests + your GEMINI_API_KEY.
"""

import os
import requests
from rich.console import Console

console = Console()


class GeminiChatManager:
    """Chat manager for Google Gemini models via REST API."""

    def __init__(self, model_id="gemini-2.5-flash"):
        self.model_id = model_id
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def set_model(self, model_id):
        self.model_id = model_id

    def is_available(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.api_key) and len(self.api_key) > 10

    def stream_response(self, user_input: str) -> str:
        """Send a message to Gemini and return the response."""
        if not self.is_available():
            return "Error: GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey"

        system_prompt = (
            "You are NOVA, a Neural Interaction Engine running on a Windows PC. "
            "You have a built-in Neural Intent Engine (NIE) that handles system commands "
            "INSTANTLY without needing you. The NIE handles: lock system, volume up/down, "
            "system status, open/close apps, screenshots, and brightness control. "
            "You handle general knowledge, chat, creative tasks, and anything the NIE can't do. "
            "You are also an expert AI coding assistant fueled by the Aider Agent. "
            "If the user asks you to write code, edit files, fix bugs, or start a project, "
            "do NOT output the code yourself. Instead, USE THE 'run_aider' TOOL to pass the "
            "task to the background autonomous coding agent."
            "Be concise, helpful, and friendly. Keep responses under 3 sentences when answering regular questions."
        )

        url = f"{self.base_url}/models/{self.model_id}:generateContent?key={self.api_key}"

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_input}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000,
            },
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": "run_aider",
                            "description": "Trigger the Advanced AI Aider engine to fulfill complex coding tasks. USE THIS if the user asks you to write code, modify files, refactor, or fix bugs. Hand the exact user prompt to this tool.",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {
                                    "coding_task": {
                                        "type": "STRING",
                                        "description": "The exact instructional prompt from the user detailing what code or files to create/edit"
                                    }
                                },
                                "required": ["coding_task"]
                            }
                        }
                    ]
                }
            ]
        }

        try:
            res = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return "Gemini returned an empty response."
                
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return "Gemini returned an empty response."

                # Check for Function Call (Aider Coder logic)
                for part in parts:
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        if fc["name"] == "run_aider":
                            args = fc.get("args", {})
                            coding_task = args.get("coding_task", "")
                            
                            try:
                                import sys
                                import io
                                from aider.coders import Coder
                                from aider.models import Model
                                from aider.io import InputOutput
                                
                                # Capture stdout so it doesn't pollute the prompt
                                old_stdout = sys.stdout
                                captured_output = io.StringIO()
                                sys.stdout = captured_output
                                
                                main_model = Model("gemini/" + self.model_id)
                                my_io = InputOutput(yes=True, pretty=False)
                                coder = Coder.create(main_model=main_model, io=my_io)
                                coder.run(coding_task)
                                
                                sys.stdout = old_stdout
                                out = captured_output.getvalue()
                                
                                # Log internally for debug, return clean status to user
                                return f"[bold green]✓ Aider successfully completed the task![/bold green]\n[dim]The code was written autonomously in the background.[/dim]"
                            except Exception as e:
                                if 'sys.stdout' in locals() and sys.stdout == captured_output:
                                    sys.stdout = old_stdout
                                return f"[bold red]✗ Aider encountered an error:[/bold red] {e}"

                # Normal text response
                return parts[0].get("text", "No response generated.")

            elif res.status_code == 400:
                return f"Gemini Error: Bad request — {res.json().get('error', {}).get('message', res.text)}"
            elif res.status_code == 403:
                return "Gemini Error: API key invalid or quota exceeded."
            elif res.status_code == 429:
                return "Gemini Error: Rate limited. Wait a moment and try again."
            elif res.status_code == 404:
                return f"Gemini Error: Model '{self.model_id}' not found. Try 'gemini-2.0-flash'."
            else:
                return f"Gemini API Error ({res.status_code}): {res.text[:200]}"

        except requests.exceptions.Timeout:
            return "Gemini Error: Request timed out. Check your internet connection."
        except requests.exceptions.ConnectionError:
            return "Gemini Error: Could not connect. Check your internet connection."
        except Exception as e:
            return f"Gemini Error: {e}"


# Global instance
gemini_chat_manager = GeminiChatManager()
