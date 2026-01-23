#!/bin/bash

# Change to the directory containing the server.py file
cd "$(dirname "$0")"

# Add the current directory to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Start the FastAPI server with uvicorn in the background
python -m uvicorn server:app --host 0.0.0.0 --port 7860 &

# Keep container running for RunPod web terminal access
sleep infinity
