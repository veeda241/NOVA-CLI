#!/bin/bash
ollama serve &
sleep 2

if ! ollama list | grep -q "gemma3"; then
  ollama pull gemma3:4b
fi

cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
sleep 2
python device/laptop_agent.py &
cd ../cli && python wayne.py
