import ollama
from rich.console import Console
from rich.markdown import Markdown, Text

console = Console()

class ChatManager:
    def __init__(self, model_name="tinyllama"):
        self.model_name = model_name
        self.history = []

    def check_ollama_ready(self):
        """Checks if Ollama service feels reachable."""
        try:
            # Simple list call to check connection
            _ = ollama.list()
            return True
        except Exception:
            return False

    def get_available_models(self):
        """Returns a list of local model names."""
        try:
            models_info = ollama.list()
            return [m['name'] for m in models_info.get('models', [])]
        except:
            return []

    def stream_response(self, user_input):
        """Streams a response from the SLM model."""
        
        if not self.check_ollama_ready():
            return "Error: Could not connect to Ollama. Make sure it's running."

        # Check if model exists, if not, try to pick the first available one
        available = self.get_available_models()
        if self.model_name not in [m.split(':')[0] for m in available] and available:
            old_name = self.model_name
            self.model_name = available[0].split(':')[0]
            # No console print here, just return the info in the logic or handle it silently
        
        # Add user message to history
        self.history.append({'role': 'user', 'content': user_input})
        
        system_prompt = (
            "You are NOVA (NIE) on Windows. Trigger ONLY if asked. "
            "Use: [ACTION: type {}] "
            "Actions: check_cpu, check_ram, check_disk, check_battery, open_browser, set_volume, screenshot."
        )

        try:
            # We return the response as a string for main.py to handle rendering
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'system', 'content': system_prompt}] + self.history,
            )
            reply = response['message']['content']
            self.history.append({'role': 'assistant', 'content': reply})
            return reply
            
        except ollama.ResponseError as e:
            if e.status_code == 404:
                return f"Error: Local model '{self.model_name}' not found. Please run 'ollama pull {self.model_name}'"
            return f"Ollama Error: {e}"
        except Exception as e:
            return f"Unexpected Local AI Error: {e}"

# Global instance
chat_manager = ChatManager()
