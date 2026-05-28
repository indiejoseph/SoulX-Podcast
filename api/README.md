# Production TTS API

The production endpoint mirrors OpenAI's speech-generation shape:

```bash
curl http://localhost:8000/v1/audio/speech \
  -X POST \
  -H "Authorization: Bearer ${SOULX_API_KEY:-local-dev-key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "soulx-podcast-mtp",
    "prompt_audio": "file:///app/example/audios/female_mandarin.wav",
    "prompt_text": "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
    "input": "Maple est le meilleur golden retriever du monde entier.",
    "language": "fr",
    "format": "wav",
    "stream": true
  }' \
  --output sample.wav
```

`stream=true` uses chunked transfer. For exact WAV headers, use
`"stream": false`; for lowest-latency playback, request `"format": "pcm"`.

`prompt_audio` may be one of:

- `file:///app/path/to/reference.wav`
- `data:audio/wav;base64,<base64 bytes>`
- raw base64 audio bytes, assumed to be WAV

`prompt_text` is required whenever `prompt_audio` is set. The path form is a
server/container-local file URI, not a client filesystem path. Set
`PROMPT_AUDIO_ROOT` to restrict accepted `file://` prompt paths; the compose
default is `/app`.

## Docker Compose

The compose stack expects model, MTP, and exported flow runtime artifacts to be
mounted from the host:

```bash
python scripts/flow/export_flow_runtime.py \
  --model_path runs/merged \
  --output_dir exports/flow_runtime \
  --encoder_fp16 \
  --streaming

SOULX_API_KEY=local-dev-key \
MTP_CHECKPOINT=/app/runs/mtp_h100_v3_kl_refresh_from22000/mtp_step2000.pt \
docker compose up --build
```

Copy `.env.serve.example` to `.env` to keep deployment paths and latency knobs
out of the command line.

Default production knobs in `docker-compose.yml`:

- `ENABLE_MTP=true`
- `FLOW_STREAMING=true`
- `FLOW_STEPS=8`
- `STREAM_FIRST_CHUNK_SIZE=4`
- `STREAM_CHUNK_SIZE=100`
- `TRT_ESTIMATOR=true`

MTP serving currently forces `LLM_ENGINE=hf` because the speculative sampler
requires direct access to the Qwen trunk and KV cache. The image is still based
on the patched vLLM runtime so the container remains compatible with vLLM
fallback/baseline experiments. For those experiments, set `ENABLE_MTP=false`
and `LLM_ENGINE=vllm`.

## Voice Registry

Voice ids are still supported as a convenience for stable production voices.
If `prompt_audio` is omitted, the service resolves `voice.id` through the local
registry. The default registry is
`config/voices.example.json`:

```json
{
  "female_mandarin": {
    "prompt_audio": "example/audios/female_mandarin.wav",
    "prompt_text": "..."
  }
}
```

Use `VOICE_REGISTRY_PATH=/app/config/voices.json` to mount a production
registry.
