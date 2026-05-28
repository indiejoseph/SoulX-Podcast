"""Microbenchmark: flow.decoder.estimator on PyTorch eager (fp16) vs TensorRT (fp16).

Loads the estimator weights from runs/merged/flow.pt, runs the same fixed inputs
through both backends, reports per-call wall time and numerical agreement.

The estimator forward is the hot path inside every CFM Euler step — it runs
n_timesteps times per chunk synthesis. A 2× speedup here translates roughly
linearly to first-chunk-flow time.

Usage:
    python scripts/flow/bench_estimator_trt.py \\
        --onnx exports/flow_runtime/flow.decoder.estimator.fp32.streaming.onnx \\
        --model_path runs/merged
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def build_engine(onnx_path: Path, fp16: bool, mel_len: int):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[onnx-parse] {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("x",    (2, 80, 16), (2, 80, mel_len), (2, 80, 2048))
    profile.set_shape("mask", (2, 1, 16),  (2, 1, mel_len),  (2, 1, 2048))
    profile.set_shape("mu",   (2, 80, 16), (2, 80, mel_len), (2, 80, 2048))
    profile.set_shape("t",    (2,),        (2,),             (2,))
    profile.set_shape("spks", (2, 80),     (2, 80),          (2, 80))
    profile.set_shape("cond", (2, 80, 16), (2, 80, mel_len), (2, 80, 2048))
    config.add_optimization_profile(profile)

    print(f"[trt] building engine (fp16={fp16}, opt mel_len={mel_len})...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")
    print(f"[trt] built in {time.perf_counter()-t0:.1f}s, "
          f"engine size {serialized.nbytes/1e6:.0f} MB")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    return engine, logger


def run_pytorch_eager(model_path: Path, inputs_cpu, dtype, n_warmup, n_iters):
    """Time eager PyTorch flow.decoder.estimator forward in autocast(dtype)."""
    from soulxpodcast.models.modules.flow import CausalMaskedDiffWithXvec

    flow = CausalMaskedDiffWithXvec()
    state = torch.load(model_path / "flow.pt", map_location="cpu", weights_only=True)
    flow.load_state_dict(state, strict=True)
    estimator = flow.decoder.estimator.eval().cuda()

    x, mask, mu, t, spks, cond = [t_.cuda() for t_ in inputs_cpu]
    streaming = True  # match how the runtime invokes it

    # Warmup
    autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
    with torch.inference_mode():
        for _ in range(n_warmup):
            with autocast_ctx:
                _ = estimator(x, mask, mu, t, spks, cond, streaming)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            with autocast_ctx:
                out = estimator(x, mask, mu, t, spks, cond, streaming)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    out_np = out.detach().float().cpu().numpy()
    return wall / n_iters, out_np


def run_trt(engine, inputs_cpu, n_warmup, n_iters):
    """Time TRT engine forward on the same inputs."""
    import tensorrt as trt

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    # Move inputs to GPU.
    x, mask, mu, t, spks, cond = [t_.cuda().contiguous() for t_ in inputs_cpu]

    # Set runtime shapes.
    context.set_input_shape("x",    tuple(x.shape))
    context.set_input_shape("mask", tuple(mask.shape))
    context.set_input_shape("mu",   tuple(mu.shape))
    context.set_input_shape("t",    tuple(t.shape))
    context.set_input_shape("spks", tuple(spks.shape))
    context.set_input_shape("cond", tuple(cond.shape))

    # Allocate output. Output shape = x.shape, dtype = engine-determined (fp32 ONNX).
    out_shape = tuple(context.get_tensor_shape("dphi_dt"))
    out_dtype = trt.nptype(engine.get_tensor_dtype("dphi_dt"))
    out_tensor = torch.empty(out_shape, dtype=getattr(torch, str(np.dtype(out_dtype))), device="cuda")

    # Bind addresses.
    context.set_tensor_address("x", x.data_ptr())
    context.set_tensor_address("mask", mask.data_ptr())
    context.set_tensor_address("mu", mu.data_ptr())
    context.set_tensor_address("t", t.data_ptr())
    context.set_tensor_address("spks", spks.data_ptr())
    context.set_tensor_address("cond", cond.data_ptr())
    context.set_tensor_address("dphi_dt", out_tensor.data_ptr())

    # Warmup
    for _ in range(n_warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    wall = time.perf_counter() - t0
    return wall / n_iters, out_tensor.detach().float().cpu().numpy()


def make_inputs(mel_len, dtype):
    """Identical inputs across backends. Random but reproducible."""
    torch.manual_seed(198964)
    return [
        torch.randn(2, 80, mel_len, dtype=dtype),  # x
        torch.ones(2, 1, mel_len, dtype=dtype),    # mask
        torch.randn(2, 80, mel_len, dtype=dtype),  # mu
        torch.tensor([0.5, 0.5], dtype=dtype),     # t
        torch.randn(2, 80, dtype=dtype),           # spks
        torch.randn(2, 80, mel_len, dtype=dtype),  # cond
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="exports/flow_runtime/flow.decoder.estimator.fp32.streaming.onnx")
    ap.add_argument("--model_path", default="runs/merged")
    ap.add_argument("--mel_lens", default="64,256,512",
                    help="comma-separated mel lengths to test")
    ap.add_argument("--opt_mel_len", type=int, default=256,
                    help="optShape mel_len for engine build")
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--n_iters", type=int, default=30)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    model_path = Path(args.model_path)
    if not onnx_path.exists():
        raise SystemExit(f"ONNX not found: {onnx_path}")

    # Build TRT engine ONCE (fp16) with opt_mel_len optimized.
    engine, _ = build_engine(onnx_path, fp16=True, mel_len=args.opt_mel_len)

    mel_lens = [int(m) for m in args.mel_lens.split(",")]

    print(f"\n{'mel_len':>8}  {'eager fp16 (ms)':>16}  {'trt fp16 (ms)':>16}  "
          f"{'speedup':>8}  {'max|Δ|':>10}  {'cosine':>10}")
    print("-" * 80)
    for mel_len in mel_lens:
        inputs_fp32 = make_inputs(mel_len, dtype=torch.float32)
        # Eager runs with fp32 inputs but autocast to fp16 inside.
        eager_ms, eager_out = run_pytorch_eager(
            model_path, inputs_fp32, dtype=torch.float16,
            n_warmup=args.n_warmup, n_iters=args.n_iters,
        )
        # TRT runs with fp32 inputs (engine handles internal precision).
        trt_ms, trt_out = run_trt(engine, inputs_fp32,
                                  n_warmup=args.n_warmup, n_iters=args.n_iters)

        max_abs = np.abs(eager_out - trt_out).max()
        cos = float((eager_out.flatten() @ trt_out.flatten()) /
                    (np.linalg.norm(eager_out.flatten()) * np.linalg.norm(trt_out.flatten()) + 1e-12))
        print(f"{mel_len:>8d}  {eager_ms*1000:>16.3f}  {trt_ms*1000:>16.3f}  "
              f"{eager_ms/trt_ms:>7.2f}x  {max_abs:>10.4f}  {cos:>10.5f}")


if __name__ == "__main__":
    main()
