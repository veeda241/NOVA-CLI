import requests
import os

token = os.getenv("HF_API_TOKEN", "your_token_here")
headers = {"Authorization": f"Bearer {token}"}

try:
    r = requests.get("https://router.huggingface.co/v1/models", headers=headers)
    if r.status_code == 200:
        data = r.json()
        models = [m['id'] for m in data.get('data', [])]
        print("TOP 20 SUPPORTED MODELS:")
        for m in models[:30]:
            print(f"- {m}")
    else:
        print(f"Error {r.status_code}: {r.text}")
except Exception as e:
    print(f"Failed: {e}")
