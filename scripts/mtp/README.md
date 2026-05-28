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
