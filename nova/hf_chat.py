import os
import requests
from rich.console import Console

console = Console()

try:
    from huggingface_hub import InferenceClient
    HAS_HF_CLIENT = True
except ImportError:
    HAS_HF_CLIENT = False

class HFChatManager:
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.model_id = model_id
        # Token is hardcoded by user
        self.api_token = os.getenv("HF_API_TOKEN", "your_token_here")
        
        if HAS_HF_CLIENT:
            self.client = InferenceClient(token=self.api_token)
        else:
            self.client = None

    def set_model(self, model_id):
        self.model_id = model_id
        if HAS_HF_CLIENT:
            self.client = InferenceClient(token=self.api_token)

    def stream_response(self, user_input):
        if not self.api_token or "YourTokenHere" in self.api_token:
            return "Error: HF_API_TOKEN not set. Visit https://huggingface.co/settings/tokens to get one."

        system_prompt = (
            "You are NOVA, a Neural Interaction Engine running on a Windows PC. "
            "You have a built-in Neural Intent Engine (NIE) that handles system commands "
            "INSTANTLY without needing you. The NIE handles: lock system, volume up/down, "
            "system status, open/close apps, screenshots, and brightness control. "
            "If the user asks about those, they are already handled — just respond conversationally. "
            "You handle general knowledge, chat, creative tasks, and anything the NIE can't do. "
            "Be concise, helpful, and friendly. Keep responses under 3 sentences when possible."
        )

        try:
            if HAS_HF_CLIENT:
                # Use official InferenceClient
                try:
                    response = self.client.chat_completion(
                        model=self.model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=500,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    # If modern chat_completion fails, try the general task-based inference
                    # as it might hit a different internal endpoint
                    return self.client.text_generation(
                        user_input,
                        model=self.model_id,
                        max_new_tokens=500,
                    )
            else:
                # Manual fallback with correct new router domain
                url = "https://router.huggingface.co/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": user_input}],
                    "max_tokens": 500
                }
                res = requests.post(url, headers=headers, json=payload, timeout=20)
                
                if res.status_code == 200:
                    return res.json()['choices'][0]['message']['content']
                elif res.status_code == 400 and "model_not_supported" in res.text:
                    # Fallback: Try direct inference endpoint if router doesn't have a provider
                    direct_url = f"https://api-inference.huggingface.co/models/{self.model_id}/v1/chat/completions"
                    direct_res = requests.post(direct_url, headers=headers, json=payload, timeout=20)
                    if direct_res.status_code == 200:
                        return direct_res.json()['choices'][0]['message']['content']
                    return f"API Error: Router doesn't support this model and direct inference failed ({direct_res.status_code})."
                else:
                    err_data = res.json() if res.text else {"error": res.status_code}
                    return f"API Error ({res.status_code}): {err_data.get('error', res.text)}"

        except Exception as e:
            error_str = str(e)
            if not error_str:
                error_str = "Unknown connection error (Check your internet or HF Token)"
                
            if "403" in error_str or "gated" in error_str.lower():
                return f"Error: This is a GATED model. You must visit https://huggingface.co/{self.model_id} and accept the terms to use it."
            if "loading" in error_str.lower():
                return "The model is currently starting up on Hugging Face. Please wait 20-30 seconds and try again."
            if "404" in error_str:
                return f"Error: Model '{self.model_id}' not found or not supported on the free inference tier."
            if "429" in error_str:
                return "Error: Too many requests. Hugging Face free tier is currently rate-limiting this model."
                
            return f"Hugging Face Error: {error_str}"

# Global instance
hf_chat_manager = HFChatManager()
