"""Streaming utilities for SoulX-Podcast.

Provides a BaseStreamer that exposes generated speech tokens through a queue,
so the main thread can consume tokens in chunks while the LLM keeps generating
in a background thread.

This is the foundational piece for bi-streaming inference (PLAN.md Phase 0
task 2). Downstream consumers (flow + HiFT chunked synthesis) read from
SpeechTokenStreamer to drive incremental audio generation.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator, List, Optional

import torch
from transformers.generation.streamers import BaseStreamer


_SENTINEL = object()


class SpeechTokenStreamer(BaseStreamer):
    """Collects generated speech tokens into a queue, yieldable as chunks.

    The HF `generate()` loop calls `.put(tokens)` for every newly-sampled
    token (see `_ras_sample_hf_engine` in `models/modules/sampler.py:185`)
    and `.end()` once generation stops. We forward those to an internal queue,
    which the main thread drains via `iter_tokens()` / `iter_chunks()`.

    Tokens are stored as raw LLM-space ids (i.e. already offset by
    `speech_token_offset`). Strip the offset before feeding into flow.
    """

    def __init__(self, eos_token_id: Optional[int] = None, timeout: Optional[float] = None):
        self.eos_token_id = eos_token_id
        self.timeout = timeout
        self._q: "queue.Queue" = queue.Queue()
        self._closed = False
        # HF's `model.generate()` calls `streamer.put(input_ids)` once at the
        # start with the full prompt tensor (any shape) before any sampling.
        # Skip that first call — we only want generated tokens, not the prompt.
        self._got_prompt = False

    # ---- producer side (called by HF generate loop) ----------------------

    def put(self, value: torch.Tensor) -> None:
        """Receive newly-generated tokens from the generation loop.

        First call: full prompt tensor — discarded.
        Subsequent calls: usually shape [batch] with batch=1 from HF generate.
        MTP streaming may pass a 1-D tensor containing multiple already-
        verified committed tokens; those are enqueued in order.
        """
        if not self._got_prompt:
            self._got_prompt = True
            return
        if value.ndim == 0:
            toks = [int(value.item())]
        elif value.ndim == 1:
            toks = [int(tok.item()) for tok in value]
        else:
            # Shouldn't happen with HF's standard generate loop, but be defensive.
            raise ValueError(
                f"SpeechTokenStreamer expected scalar or 1-D tensor, got shape {tuple(value.shape)}"
            )
        # Drop EOS — consumers don't need it in the chunk stream; .end() signals stop.
        for tok in toks:
            if self.eos_token_id is not None and tok == self.eos_token_id:
                continue
            self._q.put(tok)

    def end(self) -> None:
        if not self._closed:
            self._closed = True
            self._q.put(_SENTINEL)

    # ---- consumer side ---------------------------------------------------

    def iter_tokens(self) -> Iterator[int]:
        """Yield speech tokens one at a time until generation ends."""
        while True:
            item = self._q.get(timeout=self.timeout)
            if item is _SENTINEL:
                return
            yield item

    def iter_chunks(
        self,
        chunk_size: int,
        final_partial: bool = True,
        first_chunk_size: Optional[int] = None,
    ) -> Iterator[List[int]]:
        """Yield speech tokens in chunks.

        Args:
            chunk_size: how many tokens per chunk after the first chunk.
            final_partial: if True, yield a final shorter chunk when generation
                ends mid-chunk. If False, drop the trailing partial chunk.
            first_chunk_size: optional smaller first chunk. This reduces TTFA
                without forcing every later chunk to pay high flow overhead.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        first_target = first_chunk_size if first_chunk_size is not None else chunk_size
        if first_target <= 0:
            raise ValueError(f"first_chunk_size must be > 0, got {first_target}")
        buf: List[int] = []
        target = first_target
        yielded_first = False
        for tok in self.iter_tokens():
            buf.append(tok)
            if len(buf) >= target:
                yield buf
                buf = []
                if not yielded_first:
                    yielded_first = True
                    target = chunk_size
        if buf and final_partial:
            yield buf


def run_llm_in_thread(
    llm_engine,
    prompt: List[int],
    sampling_param,
    past_key_values=None,
    streamer: Optional[SpeechTokenStreamer] = None,
    cuda_stream: Optional["torch.cuda.Stream"] = None,
) -> threading.Thread:
    """Run `llm_engine.generate(...)` in a background thread.

    The caller passes a `SpeechTokenStreamer` and drains it from the main
    thread while generation proceeds. Returns the started Thread so the caller
    can `.join()` to retrieve final state.

    The thread stashes the LLM's output dict on `thread.result` for retrieval
    after `.join()`.

    If `cuda_stream` is provided, all CUDA ops in the worker thread run on
    that stream — letting the main thread's flow+HiFT work overlap with LLM
    generation on a different stream (PLAN.md Phase 0 task B3).
    """

    class _LLMThread(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.result: Optional[dict] = None
            self.exc: Optional[BaseException] = None

        def run(self):
            try:
                if cuda_stream is not None:
                    with torch.cuda.stream(cuda_stream):
                        self.result = llm_engine.generate(
                            prompt,
                            sampling_param,
                            past_key_values=past_key_values,
                            streamer=streamer,
                        )
                else:
                    self.result = llm_engine.generate(
                        prompt,
                        sampling_param,
                        past_key_values=past_key_values,
                        streamer=streamer,
                    )
            except BaseException as e:
                self.exc = e
                # Make sure the consumer's blocking iter_tokens() doesn't hang
                # if the generation thread dies mid-run.
                if streamer is not None:
                    streamer.end()
                raise

    t = _LLMThread()
    t.start()
    return t
