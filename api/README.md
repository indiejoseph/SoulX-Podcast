# Production TTS API

The production endpoint mirrors OpenAI's speech-generation shape:

```bash
curl http://localhost:8000/v1/audio/speech \
  -X POST \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts",
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
default is `/app`. Send `prompt_audio` + `prompt_text` on the first request,
then reuse `prompt_cache_id` when appropriate.

Every successful `/v1/audio/speech` response includes a `Prompt-Cache-Id`
header. A follow-up request can omit `prompt_audio` and `prompt_text` and reuse
the prepared prompt state:

```json
{
  "model": "tts",
  "prompt_cache_id": "pc_...",
  "input": "The next sentence to synthesize.",
  "format": "pcm",
  "stream": true
}
```

Cache ids use a local GPU tensor cache for the fast path. When Redis is
configured, file-based prompt metadata is also stored with a TTL so another API
process can rebuild its local GPU cache from the same mounted file. Inline
base64 prompt audio is not stored in Redis unless
`PROMPT_CACHE_STORE_INLINE_AUDIO=true`.

## Docker Compose

The compose stack expects model, MTP, and exported flow runtime artifacts to be
mounted from the host:

```bash
python scripts/flow/export_flow_runtime.py \
  --model_path runs/merged \
  --output_dir exports/flow_runtime \
  --encoder_fp16 \
  --streaming

API_KEY="$(openssl rand -hex 32)" \
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
- `PROMPT_CACHE_SIZE=16`
- `REDIS_URL=redis://redis:6379/0`

The compose service and image are named `tts`. The API binds to
`127.0.0.1:${API_PORT}` by default; put a reverse proxy or load balancer in
front of it for TLS, external access, request-size limits, and rate limiting.
Set `API_BIND_HOST=0.0.0.0` only when that exposure is intentional.

`API_KEY` is required by the compose file. The OpenAI-compatible endpoint,
legacy generation endpoints, task status endpoint, and download endpoint all
require `Authorization: Bearer <API_KEY>`. `/health` remains unauthenticated for
container health checks.

MTP serving currently forces `LLM_ENGINE=hf` because the speculative sampler
requires direct access to the Qwen trunk and KV cache. The image is still based
on the patched vLLM runtime so the container remains compatible with vLLM
fallback/baseline experiments. For those experiments, set `ENABLE_MTP=false`
and `LLM_ENGINE=vllm`.

## TTFA Measurement

Use streamed PCM for latency measurements. Streamed WAV responses send the WAV
header before generated audio, so first-byte timing does not equal TTFA.

```bash
python scripts/api/measure_ttfa.py \
  --url http://localhost:8000/v1/audio/speech \
  --api-key "$API_KEY" \
  --prompt-audio file:///app/example/audios/female_mandarin.wav \
  --prompt-text "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。" \
  --first-chunk-size 4 \
  --chunk-size 100 \
  --flow-steps 8 \
  --flow-streaming \
  --format pcm
```

Restart the API process immediately before running this script if you need a
true cold-cache number. Run 2 reuses the `Prompt-Cache-Id` returned by run 1 by
default. The script prints client-visible TTFA and total wall time; internal
first-LLM-token and first-flow-call fields are reported as `n/a` unless the
server exposes timing headers.
