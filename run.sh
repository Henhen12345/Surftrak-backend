#!/bin/bash
set -e
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting SurfTrak Studio API..."
uvicorn main:app --reload --port 8000 --log-level info
