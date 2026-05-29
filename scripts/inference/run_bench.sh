#!/bin/bash
# Usage: bash scripts/inference/run_bench.sh [config_name] [n_runs]
# Reads .env.local, optionally overrides LLM_ENGINE and ENABLE_MTP.

set -a
source /home/joseph/projects/SoulX-Podcast/.env.local
set +a

LABEL=${1:-hf_mtp}
N=${2:-3}

if [[ "$LABEL" == "vllm" ]]; then
    LLM_ENGINE=vllm
    ENABLE_MTP=false
elif [[ "$LABEL" == "hf_trunk" ]]; then
    LLM_ENGINE=hf
    ENABLE_MTP=false
fi

echo "Config: LLM_ENGINE=$LLM_ENGINE  ENABLE_MTP=$ENABLE_MTP"
cd /home/joseph/projects/SoulX-Podcast
/home/joseph/anaconda3/envs/w2v2bert-jyutping/bin/python \
    scripts/inference/bench_stream.py "$N" 2>&1
