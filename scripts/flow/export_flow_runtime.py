"""Export SoulX flow pieces for JIT/ONNX/TensorRT experiments.

This follows the CosyVoice deployment split:
  - TorchScript/JIT for the flow encoder.
  - fp32 ONNX for flow.decoder.estimator, which can then be converted to
    fp16 TensorRT with trtexec --fp16.

The exported estimator is the hot path inside every CFM Euler step. TensorRT
acceleration there is the most realistic flow-side path to lower first-chunk
latency without changing the model architecture.

Example:
    python scripts/flow/export_flow_runtime.py \
        --model_path runs/merged \
        --output_dir exports/flow_runtime \
        --fp16
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class EncoderFinalWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, streaming: bool):
        super().__init__()
        self.encoder = encoder
        self.streaming = streaming

    def forward(self, token_embed: torch.Tensor, token_len: torch.Tensor):
        return self.encoder(token_embed, token_len, streaming=self.streaming)


class EncoderChunkWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, streaming: bool):
        super().__init__()
        self.encoder = encoder
        self.streaming = streaming

    def forward(
        self,
        token_embed: torch.Tensor,
        token_len: torch.Tensor,
        context: torch.Tensor,
    ):
        return self.encoder(
            token_embed,
            token_len,
            context=context,
            streaming=self.streaming,
        )


class EstimatorWrapper(torch.nn.Module):
    def __init__(self, estimator: torch.nn.Module, streaming: bool):
        super().__init__()
        self.estimator = estimator
        self.streaming = streaming

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ):
        return self.estimator(
            x,
            mask,
            mu,
            t,
            spks,
            cond,
            self.streaming,
        )


def load_flow(model_path: Path, device: torch.device, fp16: bool) -> Any:
    from soulxpodcast.models.modules.flow import CausalMaskedDiffWithXvec

    flow = CausalMaskedDiffWithXvec()
    state = torch.load(model_path / "flow.pt", map_location="cpu", weights_only=True)
    flow.load_state_dict(state, strict=True)
    flow.eval().to(device)
    if fp16:
        flow.half()
    return flow


def suffix(fp16: bool, streaming: bool) -> str:
    precision = "fp16" if fp16 else "fp32"
    mode = "streaming" if streaming else "full"
    return f"{precision}.{mode}"


def export_encoder(
    flow: Any,
    output_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    token_len: int,
    context_len: int,
    fp16: bool,
    streaming: bool,
    skip_trace_check: bool,
) -> None:
    token = torch.randn(1, token_len, flow.input_size, device=device, dtype=dtype)
    token_lens = torch.tensor([token_len], device=device, dtype=torch.long)
    context = torch.randn(1, context_len, flow.input_size, device=device, dtype=dtype)
    check_len = max(token_len + 17, 8)
    check_context_len = max(context_len, 1)
    check_token = torch.randn(1, check_len, flow.input_size, device=device, dtype=dtype)
    check_lens = torch.tensor([check_len], device=device, dtype=torch.long)
    check_context = torch.randn(
        1,
        check_context_len,
        flow.input_size,
        device=device,
        dtype=dtype,
    )
    check_tolerance = 1e-2 if fp16 else 1e-5

    encoder = copy.deepcopy(flow.encoder).to(device=device, dtype=dtype)
    final = EncoderFinalWrapper(encoder, streaming=streaming).eval()
    chunk = EncoderChunkWrapper(encoder, streaming=streaming).eval()

    final_ts = torch.jit.trace(
        final,
        (token, token_lens),
        strict=True,
        check_trace=not skip_trace_check,
        check_inputs=[] if skip_trace_check else [(check_token, check_lens)],
        check_tolerance=check_tolerance,
    )
    chunk_ts = torch.jit.trace(
        chunk,
        (token, token_lens, context),
        strict=True,
        check_trace=not skip_trace_check,
        check_inputs=[] if skip_trace_check else [(check_token, check_lens, check_context)],
        check_tolerance=check_tolerance,
    )

    final_path = output_dir / f"flow.encoder.final.{suffix(fp16, streaming)}.zip"
    chunk_path = output_dir / f"flow.encoder.chunk.{suffix(fp16, streaming)}.zip"
    final_ts.save(str(final_path))
    chunk_ts.save(str(chunk_path))
    print(f"[export] encoder final: {final_path}")
    print(f"[export] encoder chunk: {chunk_path}")
    if skip_trace_check:
        print("[warn] skipped JIT trace validation")
    else:
        print(f"[check] encoder trace matched eager on alternate shape, tol={check_tolerance:g}")


def export_estimator(
    flow: Any,
    output_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    mel_len: int,
    onnx_fp16: bool,
    streaming: bool,
    opset: int,
) -> None:
    estimator = copy.deepcopy(flow.decoder.estimator).to(device=device, dtype=dtype)
    wrapper = EstimatorWrapper(estimator, streaming=streaming).eval()

    # CFG doubles the batch: conditional + unconditional.
    batch = 2
    x = torch.randn(batch, flow.output_size, mel_len, device=device, dtype=dtype)
    mask = torch.ones(batch, 1, mel_len, device=device, dtype=dtype)
    mu = torch.randn(batch, flow.output_size, mel_len, device=device, dtype=dtype)
    t = torch.full((batch,), 0.5, device=device, dtype=dtype)
    spks = torch.randn(batch, flow.output_size, device=device, dtype=dtype)
    cond = torch.randn(batch, flow.output_size, mel_len, device=device, dtype=dtype)

    onnx_path = output_dir / f"flow.decoder.estimator.{suffix(onnx_fp16, streaming)}.onnx"
    torch.onnx.export(
        wrapper,
        (x, mask, mu, t, spks, cond),
        str(onnx_path),
        input_names=["x", "mask", "mu", "t", "spks", "cond"],
        output_names=["dphi_dt"],
        dynamic_axes={
            "x": {0: "B", 2: "T"},
            "mask": {0: "B", 2: "T"},
            "mu": {0: "B", 2: "T"},
            "t": {0: "B"},
            "spks": {0: "B"},
            "cond": {0: "B", 2: "T"},
            "dphi_dt": {0: "B", 2: "T"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[export] estimator onnx: {onnx_path}")


def print_trtexec_hint(
    output_dir: Path,
    onnx_fp16: bool,
    trt_fp16: bool,
    streaming: bool,
    mel_len: int,
) -> None:
    onnx_path = output_dir / f"flow.decoder.estimator.{suffix(onnx_fp16, streaming)}.onnx"
    engine_path = output_dir / f"flow.decoder.estimator.{suffix(trt_fp16, streaming)}.plan"
    fp16_flag = " --fp16" if trt_fp16 else ""
    print("\n[trt] starter command:")
    print(
        "trtexec"
        f" --onnx={onnx_path}"
        f" --saveEngine={engine_path}"
        f"{fp16_flag}"
        " --minShapes=x:2x80x16,mask:2x1x16,mu:2x80x16,t:2,spks:2x80,cond:2x80x16"
        f" --optShapes=x:2x80x{mel_len},mask:2x1x{mel_len},mu:2x80x{mel_len},t:2,spks:2x80,cond:2x80x{mel_len}"
        " --maxShapes=x:2x80x2048,mask:2x1x2048,mu:2x80x2048,t:2,spks:2x80,cond:2x80x2048"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="runs/merged")
    parser.add_argument("--output_dir", default="exports/flow_runtime")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true",
                        help="Export encoder JIT in fp16 and print a TensorRT fp16 build command.")
    parser.add_argument("--onnx_fp16", action="store_true",
                        help="Export estimator ONNX in fp16. Default is fp32; use TRT --fp16 for runtime precision.")
    parser.add_argument("--streaming", action="store_true",
                        help="Export chunk-masked streaming variants.")
    parser.add_argument("--skip_trace_check", action="store_true",
                        help="Skip alternate-shape eager-vs-traced encoder validation.")
    parser.add_argument("--token_len", type=int, default=128)
    parser.add_argument("--context_len", type=int, default=3)
    parser.add_argument("--mel_len", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder_dtype = torch.float16 if args.fp16 else torch.float32
    estimator_dtype = torch.float16 if args.onnx_fp16 else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flow = load_flow(Path(args.model_path), device=device, fp16=False)
    export_encoder(
        flow,
        output_dir,
        dtype=encoder_dtype,
        device=device,
        token_len=args.token_len,
        context_len=args.context_len,
        fp16=args.fp16,
        streaming=args.streaming,
        skip_trace_check=args.skip_trace_check,
    )
    export_estimator(
        flow,
        output_dir,
        dtype=estimator_dtype,
        device=device,
        mel_len=args.mel_len,
        onnx_fp16=args.onnx_fp16,
        streaming=args.streaming,
        opset=args.opset,
    )
    print_trtexec_hint(
        output_dir,
        onnx_fp16=args.onnx_fp16,
        trt_fp16=args.fp16,
        streaming=args.streaming,
        mel_len=args.mel_len,
    )


if __name__ == "__main__":
    main()
