#!/bin/bash
# Rebuild tts image with torch.compile fix, restart container, run benchmark.
set -e
cd /home/joseph/projects/SoulX-Podcast

docker build -f Dockerfile.serve -t tts:latest .
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.dev.yml up -d tts

echo "Waiting for Application startup complete..."
until docker logs tts-tts-1 2>&1 | grep -q "Application startup complete"; do
  sleep 5
done

echo "Server up. Running benchmark (3 warm runs, chunk=150, long dialogue)..."
.venv/bin/python scripts/inference/bench_stream.py 3 --chunk 150 --long

echo "Also running short dialogue benchmark..."
.venv/bin/python scripts/inference/bench_stream.py 3 --chunk 150
