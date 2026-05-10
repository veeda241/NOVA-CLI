#!/bin/bash
ollama serve &
sleep 3
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 2
python device/laptop_agent.py &
python ../startup/wayne_daemon.py &
echo "W.A.Y.N.E daemon running. Say 'WAYNE' to initialize."
