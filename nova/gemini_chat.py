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
            "You are also an expert AI coding assistant (like Aider). You have been given tools "
            "to create and edit files on the user's computer. If the user asks you to create a file "
            "or write code, USE THE 'create_file' TOOL instead of just returning text."
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
                            "name": "create_file",
                            "description": "Create a new file or overwrite an existing file with the provided content. Use this to write code.",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {
                                    "filename": {
                                        "type": "STRING",
                                        "description": "The name or path of the file to create (e.g. 'tiger.py')"
                                    },
                                    "content": {
                                        "type": "STRING",
                                        "description": "The exact code or text to write into the file"
                                    }
                                },
                                "required": ["filename", "content"]
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
                        if fc["name"] == "create_file":
                            args = fc.get("args", {})
                            filename = args.get("filename")
                            content = args.get("content", "")
                            try:
                                with open(filename, "w", encoding="utf-8") as f:
                                    f.write(content)
                                return f"[bold green]✓ Created file:[/bold green] {filename}\n[dim]NOVA automatically wrote the code![/dim]"
                            except Exception as e:
                                return f"[bold red]✗ Failed to create {filename}:[/bold red] {e}"

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
