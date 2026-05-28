# MTP Streaming Benchmarks

Use `mtp_bistream_test.py` to measure TTFA and audio quality for MTP plus
bi-stream decoding. Defaults are conservative: full-context flow attention and
15 CFM steps.

```bash
# Baseline-safe behavior.
python scripts/mtp/mtp_bistream_test.py CKPT 100 4 --flow_steps 15

# Low-TTFA candidate after audio A/B approval.
python scripts/mtp/mtp_bistream_test.py CKPT 100 4 --flow_streaming --flow_steps 8

# Quality/latency middle ground.
python scripts/mtp/mtp_bistream_test.py CKPT 100 4 --flow_streaming --flow_steps 10
```

Output directories include first chunk size, main chunk size, flow mode, and
flow step count, for example:

```text
outputs/mtp_bistream/first4_chunk100_flowstream_steps8/
```

## RTX 3090 Flow Runtime Sweep

Checkpoint: KL-refresh MTP deployable artifact. First chunk: 4 tokens. Main
chunk: 100 tokens. Audio A/B passed for chunk-masked streaming flow.

| Backend | Flow Streaming | Flow Steps | TTFA | First Flow | Wall | RTF |
|---|---:|---:|---:|---:|---:|---:|
| Eager fp16 | true | 15 | 1.054s | 0.883s | 10.58s | 0.764 |
| Eager fp16 | true | 10 | 0.826s | 0.654s | 9.71s | 0.702 |
| Eager fp16 | true | 8 | 0.704s | 0.533s | 9.41s | 0.680 |
| TRT fp16 | true | 15 | 0.406s | 0.239s | 9.07s | 0.655 |
| TRT fp16 | true | 10 | 0.357s | 0.194s | 8.77s | 0.633 |
| TRT fp16 | true | 8 | 0.331s | 0.159s | 8.46s | 0.611 |
| TRT fp16 | false | 8 | 0.339s | 0.168s | 8.70s | 0.629 |
| TRT fp16 | false | 15 | 0.414s | 0.243s | 9.26s | 0.669 |

Current production low-TTFA candidate:

```bash
python scripts/mtp/mtp_bistream_test.py CKPT 100 4 --flow_streaming --flow_steps 8
```

TRT is the main flow-side win. Even TRT fp16 with 15 CFM steps beats eager fp16
with 8 steps on TTFA. With TRT enabled, chunk-masked flow attention gives
similar TTFA but better total wall time across later chunks.

Remaining headroom:

- At 0.331s TTFA, LLM verification/commit time is roughly half of the budget.
- Smaller `first_chunk_size` targets the LLM side.
- Cleaning ONNX slice/export issues may recover another small flow-side gain.
- H100/H200 should have enough memory bandwidth to make sub-200ms TTFA realistic.
