#!/usr/bin/env bash
set -e

echo "Setting up AI Web Research Agent..."

if [ ! -d ".venv" ]; then
  python3.11 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

mkdir -p data/chroma data/cache data/logs

echo "Setup complete."
echo "Activate the environment with: source .venv/bin/activate"
echo "Run the backend with: uvicorn app.main:app --reload"
echo "Run the frontend with: streamlit run frontend/streamlit_app.py"