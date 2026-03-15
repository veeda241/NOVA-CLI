from interpreter import interpreter
import os

# Expects GROQ_API_KEY to be set in environment
if not os.getenv("GROQ_API_KEY"):
    print("Warning: GROQ_API_KEY not set in environment")

interpreter.llm.model = "groq/llama-3.3-70b-versatile"
interpreter.auto_run = True

response = interpreter.chat("Print 'Hello from Open Interpreter'")
print(f"Response style: {type(response)}")
print(response)
