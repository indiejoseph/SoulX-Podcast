"""Cross-request flow + HiFT batcher for concurrent /generate-stream serving.

Single shared worker thread + single dedicated CUDA stream. Multiple concurrent
producer threads (one per in-flight request) submit chunks; the worker drains
the queue opportunistically (no fixed wait window) and runs ONE batched flow
forward + ONE batched HiFT forward per drained batch group.

Why this exists
---------------
Today every active request creates its own ``torch.cuda.Stream()`` and calls
``model.flow(...)`` / ``model.hift(...)`` inline from its own thread. With N
concurrent /generate-stream clients we get N flow streams contending on the
GPU scheduler with chaotic interleaving and zero compute amortisation across
requests. vLLM batches LLM decode across requests; the post-LLM pipeline did
not. This module is the missing piece.

Architecture
------------
- One ``FlowHiftBatcher`` instance shared by the model.
- A single ``threading.Thread`` worker owns the dedicated flow CUDA stream and
  is the *only* thread that touches the flow and hift nn.Modules. This
  side-steps any concurrency hazards inside the modules themselves.
- ``submit(...)`` is called from per-request producer threads. It enqueues a
  job and returns a ``concurrent.futures.Future`` that resolves to the
  per-request output wav tensor (already moved off-GPU).

Batching strategy
-----------------
**Opportunistic, zero artificial latency:**

  1. Worker blocks on ``queue.get()`` for the first item.
  2. Then drains the queue with ``get_nowait()`` for items already waiting.
  3. Groups drained items by ``(streaming, finalize, flow_steps)`` — flow's
     forward signature requires these to be uniform across a batch.
  4. For each group, runs one batched flow forward + one batched HiFT forward.
  5. Slices outputs back per request, resolves each request's Future.

Single-request case ⇒ batch=[1], processed immediately, no wait.
Multi-request case ⇒ jobs that piled up while worker was busy form the next
batch. The "wait" is exactly however long the previous batch took to compute,
which is fundamental and unavoidable.

Padding
-------
Flow's signature already accepts B>1 with per-row length tensors. We pad to
the batch's max length on each variable axis (speech tokens, prompt mel) and
mask via the ``_len`` tensors that the model already consumes. ``spk_emb`` is
[192] so it stacks directly.

HiFT's mel input length varies per request after the per-row slice
``mels[i, :, prompt_mel_len:mels_lens[i]]``. We pad those mels to the batch
max and slice the wav output by the proportional sample count.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class _Job:
    """One submission. Inputs are kept pre-tensor (lists of ints) where possible
    so the batcher can pad on the GPU in one go instead of N small allocations."""
    prompt_speech_tokens: List[int]
    generated_speech_tokens: List[int]
    prompt_mel: torch.Tensor          # [1, T_mel, n_mels=80] OR [1, 80, T_mel] — caller-shape preserved
    prompt_mel_len_t: torch.Tensor    # [1] int — number of valid mel frames in prompt_mel
    spk_emb: torch.Tensor             # [1, 192]
    finalize: bool
    streaming: bool
    flow_steps: int
    future: Future = field(default_factory=Future)

    def group_key(self) -> Tuple[bool, bool, int]:
        return (self.streaming, self.finalize, self.flow_steps)


class FlowHiftBatcher:
    """Cross-request flow + HiFT scheduler.

    The expected mel layout matches the inline call in
    ``SoulXPodcast._stream_synth_chunk``: ``prompt_mel`` shape ``[B=1, 80, T_mel]``.
    """

    def __init__(
        self,
        flow,
        hift,
        *,
        fp16_flow: bool,
        max_batch_size: int = 4,
        profile: bool = False,
    ):
        self.flow = flow
        self.hift = hift
        self.fp16_flow = fp16_flow
        self.max_batch_size = max(1, int(max_batch_size))
        self.profile = profile

        # Dedicated stream for all flow + hift work. One stream owned by one
        # thread = no concurrency hazards inside the modules.
        self._cuda_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._inbox: "queue.Queue[_Job]" = queue.Queue()
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._loop,
            name="soulx-flow-hift-batcher",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------ public
    def submit(
        self,
        prompt_speech_tokens: List[int],
        generated_speech_tokens: List[int],
        prompt_mel: torch.Tensor,
        prompt_mel_len_t: torch.Tensor,
        spk_emb: torch.Tensor,
        *,
        finalize: bool,
        streaming: bool,
        flow_steps: int,
    ) -> Future:
        """Submit a chunk for batched flow + HiFT. Returns a Future resolving to
        the per-request wav tensor (shape ``[1, T_audio]``, on CPU)."""
        job = _Job(
            prompt_speech_tokens=prompt_speech_tokens,
            generated_speech_tokens=generated_speech_tokens,
            prompt_mel=prompt_mel,
            prompt_mel_len_t=prompt_mel_len_t,
            spk_emb=spk_emb,
            finalize=finalize,
            streaming=streaming,
            flow_steps=flow_steps,
        )
        self._inbox.put(job)
        return job.future

    def shutdown(self) -> None:
        self._shutdown = True
        # Wake the worker if it's blocked on the queue with a sentinel.
        self._inbox.put(_Job(  # type: ignore[arg-type]
            prompt_speech_tokens=[], generated_speech_tokens=[],
            prompt_mel=torch.empty(0), prompt_mel_len_t=torch.empty(0),
            spk_emb=torch.empty(0),
            finalize=False, streaming=False, flow_steps=0,
        ))
        self._worker.join(timeout=2.0)

    # ----------------------------------------------------------------- worker
    def _loop(self) -> None:
        while not self._shutdown:
            try:
                first = self._inbox.get()
            except Exception:
                continue
            if self._shutdown:
                return
            batch: List[_Job] = [first]
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self._inbox.get_nowait())
                except queue.Empty:
                    break

            # Group by signature; one flow forward per group.
            groups: dict = {}
            for job in batch:
                groups.setdefault(job.group_key(), []).append(job)

            for jobs in groups.values():
                try:
                    self._process_group(jobs)
                except BaseException as exc:
                    for j in jobs:
                        if not j.future.done():
                            j.future.set_exception(exc)

    # -------------------------------------------------------------- batched fwd
    def _process_group(self, jobs: List[_Job]) -> None:
        """Run one batched flow + one batched HiFT for `jobs` sharing
        ``(streaming, finalize, flow_steps)``."""
        if not jobs:
            return
        device = jobs[0].prompt_mel.device
        if device.type == "cuda" and self._cuda_stream is not None:
            stream_ctx = torch.cuda.stream(self._cuda_stream)
        else:
            from contextlib import nullcontext
            stream_ctx = nullcontext()

        with stream_ctx:
            self._run_batched(jobs, device)

    def _run_batched(self, jobs: List[_Job], device: torch.device) -> None:
        B = len(jobs)
        streaming = jobs[0].streaming
        finalize = jobs[0].finalize
        flow_steps = jobs[0].flow_steps

        # ---- pad flow_input (speech tokens) -------------------------------
        token_seqs = [j.prompt_speech_tokens + j.generated_speech_tokens for j in jobs]
        token_lens = [len(s) for s in token_seqs]
        T_tok_max = max(token_lens)
        flow_input = torch.zeros((B, T_tok_max), dtype=torch.long, device=device)
        for i, seq in enumerate(token_seqs):
            flow_input[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        flow_input_len = torch.tensor(token_lens, dtype=torch.long, device=device)

        # ---- pad prompt_mel -----------------------------------------------
        # Caller convention from _stream_synth_chunk: prompt_mel is already on
        # device, shape [1, T_mel, n_mels=80] (the flow.forward transposes
        # this to [B, 80, T] internally before the CFM decoder). We strip the
        # singleton batch dim, pad along the T axis, then restack.
        prompt_mels = [j.prompt_mel.squeeze(0) for j in jobs]   # each: [T_mel, 80]
        mel_lens = [int(j.prompt_mel_len_t.item()) for j in jobs]
        T_mel_max = max(p.shape[0] for p in prompt_mels)
        n_mels = prompt_mels[0].shape[1]
        prompt_mel = torch.zeros(
            (B, T_mel_max, n_mels), dtype=prompt_mels[0].dtype, device=device,
        )
        for i, m in enumerate(prompt_mels):
            prompt_mel[i, : m.shape[0], :] = m
        prompt_mel_len_t = torch.tensor(mel_lens, dtype=torch.long, device=device)

        # ---- stack spk_emb -------------------------------------------------
        spk_emb = torch.cat([j.spk_emb for j in jobs], dim=0)  # [B, 192]

        # ---- batched flow forward -----------------------------------------
        if self.profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.amp.autocast(
            "cuda", dtype=torch.float16 if self.fp16_flow else torch.float32
        ):
            mels, mels_lens = self.flow(
                flow_input, flow_input_len,
                prompt_mel, prompt_mel_len_t, spk_emb,
                streaming=streaming, finalize=finalize,
                n_timesteps=flow_steps,
            )
        if self.profile:
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        # ---- per-request mel slice + batched HiFT -------------------------
        # Flow output ``mels`` shape: [B, 80, T_total]. Slice each row's
        # generated mel: skip the prompt_mel prefix, keep up to mels_lens[i].
        # Batched HiFT requires padding these slices to a common length; we
        # then slice the wav back by the per-row generated length.
        sliced_mels: List[torch.Tensor] = []
        gen_mel_lens: List[int] = []
        for i in range(B):
            start = mel_lens[i]
            end = int(mels_lens[i].item())
            sliced = mels[i, :, start:end]                     # [80, T_gen_i]
            sliced_mels.append(sliced)
            gen_mel_lens.append(sliced.shape[1])
        T_gen_max = max(gen_mel_lens) if gen_mel_lens else 0
        if T_gen_max == 0:
            # All-empty generation (unlikely in practice); resolve trivially.
            for j in jobs:
                j.future.set_result(torch.zeros((1, 0), dtype=torch.float32))
            return
        mel_padded = torch.zeros(
            (B, n_mels, T_gen_max), dtype=mels.dtype, device=device,
        )
        for i, m in enumerate(sliced_mels):
            if m.shape[1] > 0:
                mel_padded[i, :, : m.shape[1]] = m
        # HiFT runs on the padded batch; we slice each output by the proportional
        # samples-per-mel-frame ratio (driven by the HiFT upsample stack —
        # cumulative product of upsample_rates).
        wavs, _ = self.hift(speech_feat=mel_padded)            # [B, T_audio_max]
        if self.profile:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(
                f"[FlowHiftBatcher] B={B} flow={t1-t0:.3f}s hift={t2-t1:.3f}s "
                f"T_tok_max={T_tok_max} T_mel_max={T_mel_max} T_gen_max={T_gen_max}",
                flush=True,
            )

        # samples-per-mel-frame: derive from output / padded mel length so we
        # don't hardcode the HiFT upsample ratio.
        samples_per_mel = wavs.shape[1] / T_gen_max
        for i, j in enumerate(jobs):
            gen_len = gen_mel_lens[i]
            n_samples = int(round(gen_len * samples_per_mel))
            wav_i = wavs[i : i + 1, : n_samples].detach()
            # Defer .cpu() to the caller; they may want to keep on-GPU for
            # subsequent processing. Most callers immediately .cpu() it.
            if not j.future.done():
                j.future.set_result(wav_i)


def get_global_batcher(flow, hift, *, fp16_flow: bool) -> Optional[FlowHiftBatcher]:
    """Return a process-wide batcher iff FLOW_HIFT_BATCHING env is truthy.

    The batcher is created lazily and reused across requests. Returning ``None``
    means callers should run the legacy inline path.
    """
    if os.environ.get("FLOW_HIFT_BATCHING", "").strip().lower() not in ("1", "true", "yes"):
        return None
    global _GLOBAL_BATCHER
    if _GLOBAL_BATCHER is None:
        max_batch = int(os.environ.get("FLOW_HIFT_BATCH_SIZE", "4"))
        profile = os.environ.get("PROFILE_FLOW_STAGES", "").strip().lower() in ("1", "true", "yes")
        _GLOBAL_BATCHER = FlowHiftBatcher(
            flow, hift, fp16_flow=fp16_flow,
            max_batch_size=max_batch, profile=profile,
        )
    return _GLOBAL_BATCHER


_GLOBAL_BATCHER: Optional[FlowHiftBatcher] = None
