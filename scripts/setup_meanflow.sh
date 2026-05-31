#!/usr/bin/env bash
# Convert Chatterbox MeanFlow weights into our flow.pt format and drop them
# into an existing MODEL_PATH dir.
#
# Idempotent: skips download/convert if outputs already exist.
#
# Usage:
#   bash scripts/setup_meanflow.sh <MODEL_DIR>
#
# Where MODEL_DIR is the path to an existing checkpoint dir that already
# contains hift.pt, campplus.onnx, soulxpodcast_config.json, tokenizer files,
# and the LLM weights (model.safetensors). The script will:
#   1. Download s3gen_meanflow.safetensors from HuggingFace into tmp/chatterbox/
#   2. Run scripts/inference/convert_chatterbox_meanflow.py to extract the
#      flow.* slice into a .pt file
#   3. Back up the existing MODEL_DIR/flow.pt to flow.pt.cfm.bak
#   4. Install the converted flow.pt into MODEL_DIR/
#
# After running, restart the TTS service. The model loader auto-detects
# MeanFlow weights from the checkpoint keys and switches to basic_euler.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <MODEL_DIR>" >&2
  echo "" >&2
  echo "  MODEL_DIR must already exist and contain a non-MeanFlow flow.pt." >&2
  echo "  Example: bash scripts/setup_meanflow.sh /app/runs/merged" >&2
  exit 1
fi

MODEL_DIR="$1"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[error] $MODEL_DIR is not a directory" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/flow.pt" ]]; then
  echo "[error] $MODEL_DIR/flow.pt does not exist — provide a directory that already has a CFM flow.pt" >&2
  exit 2
fi

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAFETENSORS="$PROJ_ROOT/tmp/chatterbox/s3gen_meanflow.safetensors"
CONVERTED="$PROJ_ROOT/tmp/chatterbox/flow_meanflow.pt"

# 1. Download (skip if cached)
if [[ -f "$SAFETENSORS" ]]; then
  echo "[1/4] safetensors already at $SAFETENSORS — skipping download"
else
  echo "[1/4] downloading s3gen_meanflow.safetensors (~1 GB) ..."
  mkdir -p "$PROJ_ROOT/tmp/chatterbox"
  huggingface-cli download ResembleAI/chatterbox-turbo s3gen_meanflow.safetensors \
    --local-dir "$PROJ_ROOT/tmp/chatterbox/"
fi

# 2. Convert (skip if cached)
if [[ -f "$CONVERTED" ]]; then
  echo "[2/4] converted flow.pt already at $CONVERTED — skipping conversion"
else
  echo "[2/4] extracting flow.* keys from safetensors ..."
  python "$PROJ_ROOT/scripts/inference/convert_chatterbox_meanflow.py" \
    --src "$SAFETENSORS" \
    --dst "$CONVERTED" \
    --reference "$MODEL_DIR/flow.pt"
fi

# 3. Backup existing flow.pt (skip if already a meanflow checkpoint)
if python -c "
import sys, torch
state = torch.load('$MODEL_DIR/flow.pt', map_location='cpu', weights_only=True)
sys.exit(0 if any('time_embed_mixer' in k for k in state.keys()) else 1)
" 2>/dev/null; then
  echo "[3/4] $MODEL_DIR/flow.pt is already MeanFlow — skipping backup"
else
  if [[ -L "$MODEL_DIR/flow.pt" ]]; then
    echo "[3/4] $MODEL_DIR/flow.pt is a symlink → leaving it alone"
  elif [[ -f "$MODEL_DIR/flow.pt.cfm.bak" ]]; then
    echo "[3/4] backup $MODEL_DIR/flow.pt.cfm.bak already exists — skipping"
  else
    echo "[3/4] backing up CFM flow.pt → flow.pt.cfm.bak"
    cp "$MODEL_DIR/flow.pt" "$MODEL_DIR/flow.pt.cfm.bak"
  fi
fi

# 4. Install
echo "[4/4] installing MeanFlow flow.pt at $MODEL_DIR/flow.pt"
cp "$CONVERTED" "$MODEL_DIR/flow.pt"

echo ""
echo "Done. Restart the TTS service; the model loader will auto-detect"
echo "MeanFlow and switch to basic_euler. Recommended env: FLOW_STEPS=1."
