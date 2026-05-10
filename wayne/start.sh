#!/bin/bash
echo "=================================="
echo "  W.A.Y.N.E SYSTEM BOOT SEQUENCE "
echo "  Wireless Artificial Yielding    "
echo "  Network Engine v1.0             "
echo "=================================="
echo ""

echo "Checking Ollama..."
if ! command -v ollama &> /dev/null; then
  echo "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Starting Ollama..."
ollama serve &
sleep 3

echo "Checking Gemma model..."
if ! ollama list | grep -q "gemma3"; then
  echo "Pulling Gemma 3 4B model..."
  ollama pull gemma3:4b
fi

echo "Installing Python dependencies..."
cd backend && pip install -r requirements.txt -q

echo "Starting W.A.Y.N.E backend..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
sleep 2

echo "Starting W.A.Y.N.E laptop agent..."
python device/laptop_agent.py &

echo "Starting file indexer..."
python -c "from tools.file_indexer import file_indexer; file_indexer.start_background_index()" &

echo "Starting W.A.Y.N.E web interface..."
cd ../web && npm install -q && npm run dev &

echo ""
echo "=================================="
echo "   W.A.Y.N.E FULLY OPERATIONAL   "
echo "   Local AI - Zero Cloud Needed  "
echo "=================================="
echo "Web:    http://localhost:3000"
echo "API:    http://localhost:8000"
echo "Docs:   http://localhost:8000/docs"
echo "Model:  Gemma (gemma3:4b)"
echo ""
echo "iPhone: Open W.A.Y.N.E app on iOS"
echo "=================================="
wait
