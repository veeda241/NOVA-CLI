"""
NOVA Groq Chat Manager
======================
Uses Groq API via their OpenAI-compatible endpoint.
"""

import os
import requests
import json
from rich.console import Console

console = Console()


class GroqChatManager:
    """Chat manager for Groq models via REST API."""

    def __init__(self, model_id="llama-3.3-70b-versatile"):
        self.model_id = model_id
        # Temporarily hardcoding for testing as requested
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.history = []

    def set_model(self, model_id):
        self.model_id = model_id

    def is_available(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.api_key) and len(self.api_key) > 10

    def stream_response(self, user_input: str) -> str:
        """Send a message to Groq and return the response."""
        if not self.is_available():
            return "Error: GROQ_API_KEY not set. Get one at console.groq.com"

        system_prompt = (
            "You are NOVA, a Neural Interaction Engine running on a Windows PC. "
            "You have a built-in Neural Intent Engine (NIE) that handles system commands "
            "INSTANTLY without needing you. The NIE handles: lock system, volume up/down, "
            "system status, open/close apps, screenshots, and brightness control. "
            "You handle general knowledge, chat, creative tasks, and anything the NIE can't do. "
            "You are also an expert AI coding assistant fueled by the Aider Agent. "
            "If the user asks you to NEW tasks like writing code, editing files, fixing bugs, or starting a project, "
            "USE THE 'run_aider' TOOL. "
            "CRITICAL: If the user asks follow-up questions about a task you ALREADY finished (e.g., 'where is the file?', 'what did you write?'), "
            "do NOT use the tool. Answer directly from your memory."
            "Be concise, helpful, and friendly. Keep responses under 3 sentences when answering regular questions."
        )

        # Add user input to history
        self.history.append({"role": "user", "content": user_input})
        
        # Keep history manageable
        if len(self.history) > 20:
            self.history = self.history[-20:]

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + self.history,
            "temperature": 0.7,
            "max_tokens": 1000,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_aider",
                        "description": "Trigger the Advanced AI Aider engine to fulfill complex coding tasks. USE THIS if the user asks you to write code, modify files, refactor, or fix bugs. Hand the exact user prompt to this tool.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "coding_task": {
                                    "type": "string",
                                    "description": "The exact instructional prompt from the user detailing what code or files to create/edit"
                                }
                            },
                            "required": ["coding_task"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }

        try:
            res = requests.post(
                self.base_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                timeout=30,
            )

            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices", [])
                if not choices:
                    return "Groq returned an empty response."
                
                message = choices[0].get("message", {})
                
                # Check for Function Call (Aider Coder logic)
                if "tool_calls" in message and message["tool_calls"]:
                    for tool_call in message["tool_calls"]:
                        if tool_call.get("type") == "function" and tool_call["function"].get("name") == "run_aider":
                            args_str = tool_call["function"].get("arguments", "{}")
                            try:
                                args = json.loads(args_str)
                            except:
                                args = {}
                                
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
                                
                                # Append groq/ to model ID for litellm mapping inside Aider
                                main_model = Model("groq/" + self.model_id)
                                my_io = InputOutput(yes=True, pretty=False)
                                coder = Coder.create(main_model=main_model, io=my_io)
                                coder.run(coding_task)
                                
                                sys.stdout = old_stdout
                                out = captured_output.getvalue()
                                
                                # Extract files involved and location
                                files_involved = [os.path.basename(f) for f in coder.abs_fnames]
                                files_str = ", ".join(files_involved) if files_involved else "Project files"
                                location = os.getcwd()
                                
                                # Add tool call and tool response to history for proper context
                                self.history.append(message) # The assistant's tool call
                                self.history.append({
                                    "role": "tool",
                                    "name": "run_aider",
                                    "tool_call_id": tool_call.get("id"),
                                    "content": f"Aider successfully completed: {coding_task}. Modified files: {files_str}"
                                })
                                
                                success_msg = (
                                    f"[bold green]✓ Aider successfully completed the task on Groq![/bold green]\n"
                                    f"[bold white]Files:[/bold white] {files_str}\n"
                                    f"[bold white]Location:[/bold white] [dim]{location}[/dim]\n"
                                    f"[dim]The code for '{coding_task}' was written autonomously in the background.[/dim]"
                                )
                                return success_msg
                            except Exception as e:
                                if 'sys.stdout' in locals() and sys.stdout == captured_output:
                                    sys.stdout = old_stdout
                                return f"[bold red]✗ Aider encountered an error:[/bold red] {e}"

                # Normal text response
                reply = message.get("content", "No response generated.")
                self.history.append({"role": "assistant", "content": reply})
                return reply

            elif res.status_code == 400:
                return f"Groq Error: Bad request — {res.json().get('error', {}).get('message', res.text)}"
            elif res.status_code == 401:
                return "Groq Error: API key invalid or Unauthorized."
            elif res.status_code == 429:
                return "Groq Error: Rate limited. Wait a moment and try again."
            else:
                return f"Groq API Error ({res.status_code}): {res.text[:200]}"

        except requests.exceptions.Timeout:
            return "Groq Error: Request timed out. Check your internet connection."
        except requests.exceptions.ConnectionError:
            return "Groq Error: Could not connect. Check your internet connection."
        except Exception as e:
            return f"Groq Error: {e}"


# Global instance
groq_chat_manager = GroqChatManager()
