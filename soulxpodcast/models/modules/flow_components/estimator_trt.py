"""TensorRT-backed drop-in replacement for flow.decoder.estimator.

Same forward signature: (x, mask, mu, t, spks, cond, streaming) -> dphi_dt.
The exported ONNX has `streaming` baked in (we export with --streaming), so
the `streaming` argument here is ignored — callers must use the same mode
the engine was built for.

Build the engine once via `build_or_load_engine(...)`; serialized plan is
cached at the same path for subsequent runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


_TRT_LOADED = False


def _import_trt():
    global _TRT_LOADED
    import tensorrt as trt
    _TRT_LOADED = True
    return trt


def build_or_load_engine(
    onnx_path: Path,
    plan_path: Path,
    fp16: bool = True,
    opt_mel_len: int = 256,
    min_mel_len: int = 16,
    max_mel_len: int = 2048,
    workspace_gb: float = 2.0,
):
    """Build a TRT engine from ONNX, or load a previously-serialized .plan."""
    trt = _import_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    if plan_path.exists():
        print(f"[trt] loading cached engine: {plan_path}")
        with open(plan_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    print(f"[trt] building engine from {onnx_path} (fp16={fp16}, opt={opt_mel_len})...")
    builder = trt.Builder(logger)
    # EXPLICIT_BATCH removed in TRT 10+ (always-on); flag only needed for TRT < 10
    trt_major = int(trt.__version__.split(".")[0])
    if trt_major < 10:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    else:
        network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[onnx-parse] {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024 ** 3)
    )
    # TRT 10+ removed the global FP16 BuilderFlag — set precision per-layer instead.
    # TRT < 10: use the global FP16 permissive flag.
    if fp16:
        if trt_major < 10 and hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                try:
                    layer.precision = trt.DataType.HALF
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.DataType.HALF)
                except Exception:
                    pass  # some layers don't support FP16 — skip silently

    profile = builder.create_optimization_profile()
    profile.set_shape("x",    (2, 80, min_mel_len), (2, 80, opt_mel_len), (2, 80, max_mel_len))
    profile.set_shape("mask", (2, 1, min_mel_len),  (2, 1, opt_mel_len),  (2, 1, max_mel_len))
    profile.set_shape("mu",   (2, 80, min_mel_len), (2, 80, opt_mel_len), (2, 80, max_mel_len))
    profile.set_shape("t",    (2,),                 (2,),                 (2,))
    profile.set_shape("spks", (2, 80),              (2, 80),              (2, 80))
    profile.set_shape("cond", (2, 80, min_mel_len), (2, 80, opt_mel_len), (2, 80, max_mel_len))
    config.add_optimization_profile(profile)

    import time
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "wb") as f:
        f.write(memoryview(serialized))
    print(f"[trt] built in {time.perf_counter() - t0:.1f}s, "
          f"engine size {serialized.nbytes / 1e6:.0f} MB → {plan_path}")
    engine = runtime.deserialize_cuda_engine(memoryview(serialized))
    return engine


class FlowEstimatorTRT(torch.nn.Module):
    """Drop-in for `flow.decoder.estimator`.

    The wrapped engine takes fp32 inputs and emits fp32 outputs regardless of
    internal precision (set via the FP16 builder flag). Callers can run inside
    an autocast(fp16) context — we cast to fp32 before binding and the output
    flows back to the autocast scope as fp32 (the downstream `cfg` arithmetic
    in the CFM solver handles the type fine).
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.context = engine.create_execution_context()
        # Discover output tensor name from the engine (varies by ONNX exporter version).
        n = engine.num_io_tensors
        import tensorrt as trt
        self._output_name = next(
            engine.get_tensor_name(i) for i in range(n)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        )
        print(f"[trt] engine output tensor: {self._output_name}")

    def forward(self, x, mask, mu, t, spks, cond, streaming):
        # Cast everything to fp32 for the engine. The engine internally uses
        # fp16 tensor cores (when built with FP16 flag) but exposes fp32 IO.
        x32 = x.float().contiguous()
        mask32 = mask.float().contiguous()
        mu32 = mu.float().contiguous()
        t32 = t.float().contiguous()
        spks32 = spks.float().contiguous()
        cond32 = cond.float().contiguous()

        self.context.set_input_shape("x",    tuple(x32.shape))
        self.context.set_input_shape("mask", tuple(mask32.shape))
        self.context.set_input_shape("mu",   tuple(mu32.shape))
        self.context.set_input_shape("t",    tuple(t32.shape))
        self.context.set_input_shape("spks", tuple(spks32.shape))
        self.context.set_input_shape("cond", tuple(cond32.shape))

        out = torch.empty_like(x32)

        self.context.set_tensor_address("x", x32.data_ptr())
        self.context.set_tensor_address("mask", mask32.data_ptr())
        self.context.set_tensor_address("mu", mu32.data_ptr())
        self.context.set_tensor_address("t", t32.data_ptr())
        self.context.set_tensor_address("spks", spks32.data_ptr())
        self.context.set_tensor_address("cond", cond32.data_ptr())
        self.context.set_tensor_address(self._output_name, out.data_ptr())

        stream = torch.cuda.current_stream()
        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        return out


def install_trt_estimator(model, onnx_path: Optional[Path] = None,
                          plan_path: Optional[Path] = None,
                          fp16: bool = True, opt_mel_len: int = 256) -> None:
    """Replace model.flow.decoder.estimator with a TRT-backed version.

    Idempotent: if the estimator is already FlowEstimatorTRT, this is a no-op.
    """
    if isinstance(model.flow.decoder.estimator, FlowEstimatorTRT):
        return
    if onnx_path is None:
        onnx_path = Path("exports/flow_runtime/flow.decoder.estimator.fp32.streaming.onnx")
    if plan_path is None:
        suffix = "fp16" if fp16 else "fp32"
        plan_path = Path(f"exports/flow_runtime/flow.decoder.estimator.{suffix}.streaming.plan")
    engine = build_or_load_engine(
        Path(onnx_path), Path(plan_path), fp16=fp16, opt_mel_len=opt_mel_len,
    )
    model.flow.decoder.estimator = FlowEstimatorTRT(engine)
    print(f"[trt] installed FlowEstimatorTRT on model.flow.decoder.estimator")
