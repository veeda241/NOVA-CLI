import requests
import json
import os

token = os.getenv("HF_API_TOKEN", "your_token_here")
h={'Authorization': f'Bearer {token}'}
p={'model': 'Qwen/Qwen2.5-7B-Instruct', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 10}
try:
    r=requests.post('https://router.huggingface.co/v1/chat/completions', headers=h, json=p)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Exception: {e}")
