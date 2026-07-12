#!/bin/bash
# Entrypoint: download model from HuggingFace on first run, then start API.
set -e

MODEL_DIR="${MODEL_PATH:-/app/model}"
SENTINEL="${MODEL_DIR}/model.safetensors"

if [ ! -f "$SENTINEL" ]; then
    REPO="${HF_REPO:-indiejoseph/SoulX-Podcast-1.7B-AWQ-MeanFlow-InPaint}"
    echo "=== Model not found at ${MODEL_DIR} — downloading ${REPO} ==="
    mkdir -p "${MODEL_DIR}"

    DOWNLOAD_ARGS=(
        "--local-dir" "${MODEL_DIR}"
        "--local-dir-use-symlinks" "False"
    )
    if [ -n "${HF_TOKEN}" ]; then
        DOWNLOAD_ARGS+=("--token" "${HF_TOKEN}")
    fi

    huggingface-cli download "${REPO}" "${DOWNLOAD_ARGS[@]}"
    echo "=== Download complete ==="
else
    echo "=== Model already present at ${MODEL_DIR} — skipping download ==="
fi

exec python3 run_api.py "$@"
