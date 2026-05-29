"""
SoulXPodcast Model Service Layer
"""
import base64
import binascii
import copy
import hashlib
import json
import re
import logging
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Optional
from urllib.parse import unquote, urlparse
import torch
import numpy as np
import random
import gc
import threading

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.utils.dataloader import (
    AUDIO_START,
    SPK_DICT,
    TEXT_END,
    TEXT_START,
    PodcastInferHandler,
)
from soulxpodcast.utils.infer_utils import process_single_input
from soulxpodcast.utils.streaming import SpeechTokenStreamer
from soulxpodcast.utils.text import normalize_text
from soulxpodcast.training.mtp_inference import mtp_speculative_sample_cached
from soulxpodcast.training.mtp_module import MtpConfig, SequentialMTP

from api.audio import tensor_to_pcm16_bytes, wav_bytes_from_pcm, wav_header
from api.config import config as api_config
from api.models import SpeechRequest
from api.redis_state import get_redis_client, redis_key
from api.utils import parse_dialogue_text

logger = logging.getLogger(__name__)


LANGUAGE_PREFIX = {
    "yue": "<|Yue|>",
    "zh-yue": "<|Yue|>",
    "cantonese": "<|Yue|>",
    "sichuan": "<|Sichuan|>",
    "sichuanese": "<|Sichuan|>",
    "henan": "<|Henan|>",
    "henanese": "<|Henan|>",
}

ALLOWED_PROMPT_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}
PROMPT_AUDIO_MIME_EXTENSIONS = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
}


class SoulXPodcastService:
    """Singleton service for the SoulXPodcast model."""

    _instance: Optional['SoulXPodcastService'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                logger.info("Creating new SoulXPodcastService instance")
                cls._instance = super(SoulXPodcastService, cls).__new__(cls)
                cls._instance._initialized = False
                cls._instance._generation_lock = threading.Lock()
                cls._instance._speech_lock = threading.Lock()
                cls._instance._prompt_cache_lock = threading.Lock()
                cls._instance._prompt_cache = OrderedDict()
                cls._instance._prompt_cache_ids = OrderedDict()
        return cls._instance

    def __init__(self):
        """Initialize the singleton model once."""
        if not self._initialized:
            logger.info("Initializing SoulXPodcastService (first time)")
            self._load_model()
            self._initialized = True
        else:
            logger.info("SoulXPodcastService already initialized, skipping model load")

    def _load_model(self):
        """Load model components."""
        try:
            logger.info(f"Loading SoulXPodcast model from {api_config.model_path}...")
            logger.info(f"Using LLM engine: {api_config.llm_engine}")

            hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
                initial_values={"fp16_flow": api_config.fp16_flow},
                json_file=f"{api_config.model_path}/soulxpodcast_config.json"
            )

            model_config = Config(
                model=api_config.model_path,
                enforce_eager=api_config.vllm_enforce_eager,
                llm_engine=api_config.llm_engine,
                hf_config=hf_config
            )

            self.model = SoulXPodcast(model_config)
            self.dataset = PodcastInferHandler(
                self.model.llm.tokenizer,
                None,
                model_config
            )
            self.config = model_config
            self.mtp = self._load_mtp()
            self.trt_streaming_mode = None
            if api_config.trt_estimator:
                self._install_trt_estimator()

            logger.info(f"Model loaded successfully with {api_config.llm_engine} engine!")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model load failed: {str(e)}")

    def _load_mtp(self):
        if not api_config.enable_mtp:
            logger.info("MTP disabled by ENABLE_MTP=false")
            return None
        if not api_config.mtp_checkpoint:
            logger.warning("MTP_CHECKPOINT not set; /v1/audio/speech will fall back to trunk-only synthesis")
            return None
        ckpt_path = Path(api_config.mtp_checkpoint)
        if not ckpt_path.exists():
            raise RuntimeError(f"MTP_CHECKPOINT does not exist: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        mtp_config = MtpConfig(**ckpt["mtp_config"])
        base = self.model.llm.model
        decoder_layer_cls = base.model.layers[0].__class__
        mtp = SequentialMTP(mtp_config, decoder_layer_cls, base.config)
        mtp.load_state_dict(ckpt["mtp_state"])
        mtp = mtp.to(device="cuda").eval()
        logger.info("Loaded MTP checkpoint %s with %d layers", ckpt_path, len(mtp.layers))
        return mtp

    def _install_trt_estimator(self) -> None:
        from soulxpodcast.models.modules.flow_components.estimator_trt import install_trt_estimator

        mode = "streaming" if api_config.flow_streaming else "full"
        onnx_path = api_config.trt_onnx or f"exports/flow_runtime/flow.decoder.estimator.fp32.{mode}.onnx"
        plan_path = api_config.trt_plan or f"exports/flow_runtime/flow.decoder.estimator.fp16.{mode}.plan"
        install_trt_estimator(
            self.model,
            onnx_path=Path(onnx_path),
            plan_path=Path(plan_path),
            opt_mel_len=api_config.trt_opt_mel_len,
        )
        self.trt_streaming_mode = api_config.flow_streaming
        logger.info("Installed TRT estimator for flow_streaming=%s", self.trt_streaming_mode)

    def is_loaded(self) -> bool:
        """Return whether the model has been loaded."""
        return hasattr(self, 'model') and self.model is not None

    def generate(
        self,
        prompt_audio_paths: List[str],
        prompt_texts: List[str],
        dialogue_text: str,
        seed: int = 1988,
        temperature: float = 0.6,
        top_k: int = 100,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate speech.

        Args:
            prompt_audio_paths: Reference audio paths.
            prompt_texts: Reference transcripts.
            dialogue_text: Dialogue text.
            seed: Random seed.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty.

        Returns:
            Tuple[int, np.ndarray]: sample rate and audio array.
        """
        logger.info(f"Generate called - Instance ID: {id(self)}, Model loaded: {self.is_loaded()}")

        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")

        with self._generation_lock:
            logger.info("Acquired generation lock")
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                num_speakers = len(prompt_audio_paths)
                logger.info(f"Generating audio for {num_speakers} speaker(s)")

                target_text_list = parse_dialogue_text(dialogue_text, num_speakers)
                logger.info(f"Parsed dialogue into {len(target_text_list)} segments")

                spks, texts = [], []
                for target_text in target_text_list:
                    pattern = r'(\[S[1-9]\])(.+)'
                    match = re.match(pattern, target_text)
                    if match:
                        text, spk = match.group(2), int(match.group(1)[2]) - 1
                        spks.append(spk)
                        texts.append(text)
                    else:
                        raise ValueError(f"Invalid dialogue text format: {target_text}")

                dataitem = {
                    "key": "api_001",
                    "prompt_text": prompt_texts,
                    "prompt_wav": prompt_audio_paths,
                    "text": texts,
                    "spk": spks,
                }

                self.dataset.update_datasource([dataitem])

                data = self.dataset[0]

                import s3tokenizer
                prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])
                spk_emb_for_flow = torch.tensor(data["spk_emb"])
                prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
                    data["mel"], batch_first=True, padding_value=0
                )
                prompt_mels_lens_for_flow = torch.tensor(data['mel_len'])
                text_tokens_for_llm = data["text_tokens"]
                prompt_text_tokens_for_llm = data["prompt_text_tokens"]
                spk_ids = data["spks_list"]

                sampling_params = SamplingParams(
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p,
                    use_ras=True,
                    win_size=25,
                    tau_r=0.2,
                    restrict_speech_vocab=api_config.restrict_speech_vocab,
                    speech_vocab_size=api_config.speech_vocab_size,
                )

                infos = [data["info"]]
                processed_data = {
                    "prompt_mels_for_llm": prompt_mels_for_llm,
                    "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
                    "prompt_text_tokens_for_llm": prompt_text_tokens_for_llm,
                    "text_tokens_for_llm": text_tokens_for_llm,
                    "prompt_mels_for_flow_ori": prompt_mels_for_flow,
                    "prompt_mels_lens_for_flow": prompt_mels_lens_for_flow,
                    "spk_emb_for_flow": spk_emb_for_flow,
                    "sampling_params": sampling_params,
                    "spk_ids": spk_ids,
                    "infos": infos,
                    "use_dialect_prompt": False,
                }

                logger.info("Running model inference...")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                import concurrent.futures
                import signal

                def run_inference():
                    """Run model inference in a worker thread."""
                    with torch.no_grad():
                        return self.model.forward_longform(**processed_data)

                num_segments = len(texts)
                timeout_seconds = max(1200, num_segments * 120)

                logger.info(f"Starting inference with timeout: {timeout_seconds}s for {num_segments} segments")

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_inference)
                    try:
                        results_dict = future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Model inference timeout after {timeout_seconds} seconds")
                        future.cancel()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise TimeoutError(
                            f"Model inference timed out after {timeout_seconds}s. "
                            "The audio may be too long or GPU memory may be insufficient."
                        )
                    except Exception as e:
                        logger.error(f"Model inference failed: {e}")
                        raise RuntimeError(f"Model inference failed: {str(e)}")

                target_audio = None
                for i in range(len(results_dict['generated_wavs'])):
                    if target_audio is None:
                        target_audio = results_dict['generated_wavs'][i]
                    else:
                        target_audio = torch.concat(
                            [target_audio, results_dict['generated_wavs'][i]], axis=1
                        )

                audio_array = target_audio.cpu().squeeze(0).numpy()
                sample_rate = 24000

                del target_audio
                del results_dict
                if 'processed_data' in locals():
                    del processed_data

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                gc.collect()

                logger.info(f"Audio generation completed. Duration: {len(audio_array) / sample_rate:.2f}s")

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                    logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

                return sample_rate, audio_array

            except Exception as e:
                logger.error(f"Generation failed: {e}", exc_info=True)
                raise RuntimeError(f"Speech generation failed: {str(e)}")
            finally:
                logger.info("Released generation lock")

    # ------------------------------------------------------------------ #
    # OpenAI-compatible `/v1/audio/speech` path.
    # ------------------------------------------------------------------ #

    def _validate_prompt_audio_path(self, audio_path: Path, *, label: str) -> Path:
        if api_config.prompt_audio_root:
            root = Path(api_config.prompt_audio_root).resolve()
            resolved = audio_path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"{label} prompt_audio must be under PROMPT_AUDIO_ROOT={root}")
        if not audio_path.exists():
            raise ValueError(f"{label} prompt_audio does not exist: {audio_path}")
        if audio_path.suffix.lower() not in ALLOWED_PROMPT_AUDIO_EXTENSIONS:
            raise ValueError(
                f"{label} prompt_audio format is unsupported: {audio_path.suffix}. "
                f"Supported: {', '.join(sorted(ALLOWED_PROMPT_AUDIO_EXTENSIONS))}"
            )
        return audio_path

    def _prompt_file_cache_key(self, audio_path: Path, prompt_text: str) -> str:
        resolved = audio_path.resolve()
        stat = resolved.stat()
        text_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        return f"file:{resolved}:{stat.st_size}:{stat.st_mtime_ns}:text:{text_hash}"

    def _prompt_bytes_cache_key(self, payload: bytes, prompt_text: str) -> str:
        audio_hash = hashlib.sha256(payload).hexdigest()
        text_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        return f"bytes:{audio_hash}:text:{text_hash}"

    def _prompt_cache_id_from_key(self, cache_key: str) -> str:
        return f"pc_{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:32]}"

    def _write_inline_prompt_audio(self, payload: bytes, suffix: str) -> Path:
        if not payload:
            raise ValueError("prompt_audio base64 decoded to empty bytes")
        if len(payload) > api_config.max_upload_size:
            raise ValueError(
                f"prompt_audio exceeds max size "
                f"({api_config.max_upload_size / 1024 / 1024:.0f}MB)"
            )
        if suffix.lower() not in ALLOWED_PROMPT_AUDIO_EXTENSIONS:
            suffix = ".wav"
        api_config.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="speech_prompt_",
                suffix=suffix,
                dir=api_config.temp_dir,
                delete=False,
            ) as f:
                temp_path = Path(f.name)
                f.write(payload)
                return temp_path
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

    def _resolve_prompt_audio(self, prompt_audio: str, prompt_text: str) -> Dict[str, Any]:
        value = prompt_audio.strip()
        if not value:
            raise ValueError("prompt_audio must not be empty")

        if value.startswith("file://"):
            parsed = urlparse(value)
            if parsed.netloc not in ("", "localhost"):
                raise ValueError("prompt_audio file:// URI must reference a local file")
            path = Path(unquote(parsed.path))
            path = self._validate_prompt_audio_path(path, label="inline")
            cache_key = self._prompt_file_cache_key(path, prompt_text)
            return {
                "prompt_audio": str(path),
                "_cache_key": cache_key,
                "_cache_id": self._prompt_cache_id_from_key(cache_key),
                "_prompt_audio_kind": "file",
            }

        if value.startswith("data:"):
            try:
                header, encoded = value.split(",", 1)
            except ValueError as e:
                raise ValueError("prompt_audio data URI must contain a comma separator") from e
            if ";base64" not in header.lower():
                raise ValueError("prompt_audio data URI must be base64 encoded")
            mime = header[5:].split(";", 1)[0].lower()
            suffix = PROMPT_AUDIO_MIME_EXTENSIONS.get(mime, ".wav")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except binascii.Error as e:
                raise ValueError("prompt_audio contains invalid base64 data") from e
            cache_key = self._prompt_bytes_cache_key(payload, prompt_text)
            return {
                "_cache_key": cache_key,
                "_cache_id": self._prompt_cache_id_from_key(cache_key),
                "_prompt_audio_kind": "inline",
                "_inline_audio_payload": payload,
                "_inline_audio_suffix": suffix,
            }

        try:
            payload = base64.b64decode(value, validate=True)
        except binascii.Error as e:
            raise ValueError(
                "prompt_audio must be a file:// URI, data:audio/*;base64 URI, or raw base64 audio"
            ) from e
        cache_key = self._prompt_bytes_cache_key(payload, prompt_text)
        return {
            "_cache_key": cache_key,
            "_cache_id": self._prompt_cache_id_from_key(cache_key),
            "_prompt_audio_kind": "inline",
            "_inline_audio_payload": payload,
            "_inline_audio_suffix": ".wav",
        }

    def _resolve_speech_prompt(self, request: SpeechRequest) -> Dict[str, Any]:
        if request.prompt_cache_id:
            if request.prompt_audio or request.prompt_text:
                raise ValueError("prompt_cache_id cannot be combined with prompt_audio or prompt_text")
            if not request.prompt_cache_id.startswith("pc_"):
                raise ValueError("prompt_cache_id must be an opaque id returned by Prompt-Cache-Id")
            return {
                "id": "cached_prompt",
                "_cache_id": request.prompt_cache_id,
            }
        if request.prompt_audio:
            if not request.prompt_text or not request.prompt_text.strip():
                raise ValueError("prompt_text is required when prompt_audio is provided")
            prompt_text = request.prompt_text.strip()
            prompt = self._resolve_prompt_audio(
                request.prompt_audio,
                prompt_text,
            )
            prompt.update({
                "id": "inline_prompt",
                "prompt_text": prompt_text,
            })
            return prompt
        if request.prompt_text:
            raise ValueError("prompt_audio is required when prompt_text is provided")
        raise ValueError("prompt_audio and prompt_text are required unless prompt_cache_id is provided")

    def _apply_language_prefix(self, text: str, language: Optional[str]) -> str:
        if not language:
            return text
        prefix = LANGUAGE_PREFIX.get(language.lower())
        if prefix and not text.lstrip().startswith(prefix):
            return f"{prefix}{text}"
        return text

    def _get_prompt_cache_entry(
        self,
        *,
        cache_key: Optional[str] = None,
        cache_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if api_config.prompt_cache_size <= 0:
            return None
        with self._prompt_cache_lock:
            if cache_key is None and cache_id is not None:
                cache_key = self._prompt_cache_ids.get(cache_id)
            if cache_key is None:
                return None
            entry = self._prompt_cache.get(cache_key)
            if entry is not None:
                self._prompt_cache.move_to_end(cache_key)
                prompt_cache_id = entry.get("prompt_cache_id")
                if prompt_cache_id is not None:
                    self._prompt_cache_ids[prompt_cache_id] = cache_key
                    self._prompt_cache_ids.move_to_end(prompt_cache_id)
            return entry

    def _store_prompt_cache_entry(
        self,
        cache_key: str,
        cache_id: str,
        entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        entry["prompt_cache_id"] = cache_id
        entry["prompt_cache_key"] = cache_key
        if api_config.prompt_cache_size <= 0:
            return entry
        with self._prompt_cache_lock:
            self._prompt_cache[cache_key] = entry
            self._prompt_cache.move_to_end(cache_key)
            self._prompt_cache_ids[cache_id] = cache_key
            self._prompt_cache_ids.move_to_end(cache_id)
            while len(self._prompt_cache) > api_config.prompt_cache_size:
                _, evicted = self._prompt_cache.popitem(last=False)
                evicted_id = evicted.get("prompt_cache_id")
                if evicted_id is not None:
                    self._prompt_cache_ids.pop(evicted_id, None)
        return entry

    def _prompt_cache_metadata_key(self, cache_id: str) -> str:
        return redis_key("prompt_cache", cache_id)

    def _store_prompt_cache_metadata(
        self,
        *,
        cache_key: str,
        cache_id: str,
        prompt: Dict[str, Any],
    ) -> None:
        redis = get_redis_client()
        if redis is None:
            return

        metadata: Dict[str, Any] = {
            "cache_key": cache_key,
            "cache_id": cache_id,
            "prompt_text": prompt.get("prompt_text"),
            "kind": prompt.get("_prompt_audio_kind"),
        }
        if prompt.get("_prompt_audio_kind") == "file":
            metadata["prompt_audio"] = prompt.get("prompt_audio")
        elif (
            prompt.get("_prompt_audio_kind") == "inline"
            and api_config.prompt_cache_store_inline_audio
            and "_inline_audio_payload" in prompt
        ):
            metadata["payload_b64"] = base64.b64encode(prompt["_inline_audio_payload"]).decode("ascii")
            metadata["suffix"] = prompt.get("_inline_audio_suffix", ".wav")
        else:
            metadata["rebuildable"] = False

        try:
            redis.set(
                self._prompt_cache_metadata_key(cache_id),
                json.dumps(metadata, ensure_ascii=False),
                ex=api_config.prompt_cache_ttl_seconds,
            )
        except Exception:
            logger.exception("Failed to store prompt cache metadata in Redis")

    def _restore_prompt_from_cache_metadata(self, cache_id: str) -> Optional[Dict[str, Any]]:
        redis = get_redis_client()
        if redis is None:
            return None
        try:
            raw = redis.get(self._prompt_cache_metadata_key(cache_id))
        except Exception:
            logger.exception("Failed to read prompt cache metadata from Redis")
            return None
        if not raw:
            return None

        metadata = json.loads(raw)
        cache_key = metadata.get("cache_key")
        prompt_text = metadata.get("prompt_text")
        if not cache_key or not prompt_text:
            return None

        if metadata.get("kind") == "file" and metadata.get("prompt_audio"):
            return {
                "id": "cached_prompt",
                "_cache_key": cache_key,
                "_cache_id": cache_id,
                "_prompt_audio_kind": "file",
                "prompt_audio": metadata["prompt_audio"],
                "prompt_text": prompt_text,
            }

        if metadata.get("kind") == "inline" and metadata.get("payload_b64"):
            try:
                payload = base64.b64decode(metadata["payload_b64"], validate=True)
            except binascii.Error:
                logger.warning("Prompt cache metadata has invalid inline audio payload")
                return None
            return {
                "id": "cached_prompt",
                "_cache_key": cache_key,
                "_cache_id": cache_id,
                "_prompt_audio_kind": "inline",
                "_inline_audio_payload": payload,
                "_inline_audio_suffix": metadata.get("suffix", ".wav"),
                "prompt_text": prompt_text,
            }

        return None

    def _encode_target_text(self, text: str) -> List[int]:
        # Keep this template in sync with PodcastInferHandler.__getitem__
        # target-text preprocessing in soulxpodcast/utils/dataloader.py.
        normalized = normalize_text(text)
        token_text = f"{SPK_DICT[0]}{TEXT_START}{normalized}{TEXT_END}{AUDIO_START}"
        return self.dataset.text_tokenizer.encode(token_text)

    def _fork_dynamic_cache(self, cache):
        """Copy cache containers while sharing immutable prefix KV tensors."""
        forked = copy.copy(cache)
        if hasattr(cache, "layers"):
            forked.layers = [copy.copy(layer) for layer in cache.layers]
        return forked

    @torch.inference_mode()
    def _build_prompt_prefix_kv(self, prompt_prefix_ids: List[int]) -> Tuple[Any, torch.Tensor]:
        input_ids = torch.tensor([prompt_prefix_ids], dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.llm.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
        return out.past_key_values, out.last_hidden_state.detach()

    def _build_prompt_cache_entry(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        cache_id = prompt["_cache_id"]
        cache_key = prompt.get("_cache_key")
        cached = self._get_prompt_cache_entry(cache_key=cache_key, cache_id=cache_id)
        if cached is not None:
            return cached
        if cache_key is None:
            restored_prompt = self._restore_prompt_from_cache_metadata(cache_id)
            if restored_prompt is None:
                raise ValueError(f"Unknown or expired prompt_cache_id: {cache_id}")
            prompt = restored_prompt
            cache_key = prompt["_cache_key"]

        temp_prompt_audio = None
        if "_inline_audio_payload" in prompt:
            temp_prompt_audio = self._write_inline_prompt_audio(
                prompt["_inline_audio_payload"],
                prompt.get("_inline_audio_suffix", ".wav"),
            )
            prompt_audio_path = temp_prompt_audio
        else:
            prompt_audio_path = Path(prompt["prompt_audio"])
        try:
            prepared = process_single_input(
                self.dataset,
                ["[S1]cache"],
                [str(prompt_audio_path)],
                [prompt["prompt_text"]],
                False,
                [""],
            )
        finally:
            if temp_prompt_audio is not None:
                try:
                    temp_prompt_audio.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to delete temporary prompt audio: %s", temp_prompt_audio)

        with torch.inference_mode():
            prompt_speech_tokens, prompt_lens = self.model.audio_tokenizer.quantize(
                prepared["prompt_mels_for_llm"].cuda(),
                prepared["prompt_mels_lens_for_llm"].cuda(),
            )
        prompt_len = prompt_lens[0].item()
        prompt_tokens = prompt_speech_tokens[0, :prompt_len].detach().cpu()
        prompt_mel_ori = prepared["prompt_mels_for_flow_ori"][0]
        prompt_mel_len = prompt_mel_ori.shape[0]
        if prompt_len * 2 > prompt_mel_len:
            prompt_len = int(prompt_mel_len / 2)
            prompt_tokens = prompt_tokens[:prompt_len]
            prompt_mel = prompt_mel_ori.detach().clone()
        else:
            prompt_mel = prompt_mel_ori[: prompt_len * 2].detach().clone()

        speech_token_offset = self.model.config.hf_config.speech_token_offset
        eos_id = self.model.config.hf_config.eos_token_id
        prompt_token_list = prompt_tokens.tolist()
        spk_tokens = [t + speech_token_offset for t in prompt_token_list] + [eos_id]
        prompt_prefix_ids = prepared["prompt_text_tokens_for_llm"][0] + spk_tokens

        entry = {
            "prompt_mels_for_llm": prepared["prompt_mels_for_llm"],
            "prompt_mels_lens_for_llm": prepared["prompt_mels_lens_for_llm"],
            "prompt_text_tokens_for_llm": prepared["prompt_text_tokens_for_llm"],
            "prompt_mels_for_flow_ori": prepared["prompt_mels_for_flow_ori"],
            "prompt_mels_lens_for_flow": prepared["prompt_mels_lens_for_flow"],
            "spk_emb_for_flow": prepared["spk_emb_for_flow"],
            "prompt_prefix_ids": prompt_prefix_ids,
            "prompt_tokens": prompt_token_list,
            "prompt_mel": prompt_mel[None],
            "prompt_mel_len": int(prompt_mel.shape[0]),
            "spk_emb": prepared["spk_emb_for_flow"][0:1],
            "speech_token_offset": speech_token_offset,
            "eos_id": eos_id,
            "info": prepared["infos"][0],
            "prompt_prefix_cache": None,
            "prompt_prefix_hidden": None,
            "prompt_prefix_len": len(prompt_prefix_ids),
        }
        should_cache_hf_prefix = (
            self.config.llm_engine == "hf"
            and (self.mtp is not None or api_config.hf_prompt_prefix_cache)
        )
        if should_cache_hf_prefix:
            prefix_cache, prefix_hidden = self._build_prompt_prefix_kv(prompt_prefix_ids)
            entry["prompt_prefix_cache"] = prefix_cache
            entry["prompt_prefix_hidden"] = prefix_hidden
        logger.info("Cached prompt audio entry %s", prompt.get("id", "inline_prompt"))
        stored = self._store_prompt_cache_entry(cache_key, cache_id, entry)
        self._store_prompt_cache_metadata(cache_key=cache_key, cache_id=cache_id, prompt=prompt)
        return stored

    def _build_speech_prepared(self, request: SpeechRequest):
        prompt = self._resolve_speech_prompt(request)
        text = self._apply_language_prefix(request.input.strip(), request.language)
        prompt_entry = self._build_prompt_cache_entry(prompt)
        prepared = {
            "prompt_mels_for_llm": prompt_entry["prompt_mels_for_llm"],
            "prompt_mels_lens_for_llm": prompt_entry["prompt_mels_lens_for_llm"],
            "prompt_text_tokens_for_llm": prompt_entry["prompt_text_tokens_for_llm"],
            "text_tokens_for_llm": [self._encode_target_text(text)],
            "prompt_mels_for_flow_ori": prompt_entry["prompt_mels_for_flow_ori"],
            "prompt_mels_lens_for_flow": prompt_entry["prompt_mels_lens_for_flow"],
            "spk_emb_for_flow": prompt_entry["spk_emb_for_flow"],
            "spk_ids": [0],
            "infos": [prompt_entry["info"]],
            "use_dialect_prompt": False,
            "prompt_cache_entry": prompt_entry,
        }
        prepared["sampling_params"] = SamplingParams(
            temperature=request.temperature if request.temperature is not None else api_config.default_temperature,
            repetition_penalty=request.repetition_penalty if request.repetition_penalty is not None else 1.25,
            top_k=request.top_k if request.top_k is not None else api_config.default_top_k,
            top_p=request.top_p if request.top_p is not None else api_config.default_top_p,
            use_ras=True,
            win_size=25,
            tau_r=0.2,
            restrict_speech_vocab=api_config.restrict_speech_vocab,
            speech_vocab_size=api_config.speech_vocab_size,
        )
        return prepared

    def prepare_speech_context(self, request: SpeechRequest) -> Dict[str, Any]:
        prepared = self._build_speech_prepared(request)
        prompt_entry = prepared["prompt_cache_entry"]
        return {
            "prepared": prepared,
            "prompt_cache_id": prompt_entry["prompt_cache_id"],
        }

    def _resolve_seed(self, request: SpeechRequest) -> int:
        if request.seed is not None:
            return request.seed
        if api_config.default_seed is not None:
            return api_config.default_seed
        return random.SystemRandom().randrange(0, 2**31)

    def _seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @torch.inference_mode()
    def _build_first_turn_prompt(self, prepared):
        prompt_entry = prepared.get("prompt_cache_entry")
        if prompt_entry is not None:
            return (
                prompt_entry["prompt_prefix_ids"] + prepared["text_tokens_for_llm"][0],
                prompt_entry["eos_id"],
                prompt_entry["speech_token_offset"],
            )

        prompt_mels = prepared["prompt_mels_for_llm"]
        prompt_mels_lens = prepared["prompt_mels_lens_for_llm"]
        prompt_text_tokens = prepared["prompt_text_tokens_for_llm"]
        text_tokens = prepared["text_tokens_for_llm"]

        speech_token_offset = self.model.config.hf_config.speech_token_offset
        eos_id = self.model.config.hf_config.eos_token_id
        prompt_speech_tokens, prompt_lens = self.model.audio_tokenizer.quantize(
            prompt_mels.cuda(), prompt_mels_lens.cuda()
        )
        spk_tokens = prompt_speech_tokens[0, : prompt_lens[0].item()].tolist()
        spk_tokens = [t + speech_token_offset for t in spk_tokens] + [eos_id]
        return prompt_text_tokens[0] + spk_tokens + text_tokens[0], eos_id, speech_token_offset

    @torch.inference_mode()
    def _prepare_synth_state(self, prepared):
        prompt_entry = prepared.get("prompt_cache_entry")
        if prompt_entry is not None:
            return {
                "prompt_tokens": list(prompt_entry["prompt_tokens"]),
                "prompt_mel": prompt_entry["prompt_mel"].cuda(),
                "prompt_mel_len": torch.tensor(
                    [prompt_entry["prompt_mel_len"]],
                    device="cuda",
                ),
                "spk_emb": prompt_entry["spk_emb"].cuda(),
            }

        prompt_mels = prepared["prompt_mels_for_llm"]
        prompt_mels_lens = prepared["prompt_mels_lens_for_llm"]
        prompt_spk_tokens, prompt_lens = self.model.audio_tokenizer.quantize(
            prompt_mels.cuda(), prompt_mels_lens.cuda()
        )
        prompt_len = prompt_lens[0].item()
        prompt_mel = prepared["prompt_mels_for_flow_ori"][0]
        prompt_mel = prompt_mel[: prompt_len * 2].cuda()
        return {
            "prompt_tokens": prompt_spk_tokens[0, :prompt_len].tolist(),
            "prompt_mel": prompt_mel[None],
            "prompt_mel_len": torch.tensor([prompt_mel.shape[0]], device="cuda"),
            "spk_emb": prepared["spk_emb_for_flow"][0:1].cuda(),
        }

    @torch.inference_mode()
    def _synthesize_chunk(
        self,
        synth_state: Dict[str, Any],
        all_speech_tokens: List[int],
        *,
        finalize: bool,
        streaming: bool,
        flow_steps: int,
    ) -> torch.Tensor:
        flow_input = torch.tensor(
            [synth_state["prompt_tokens"] + all_speech_tokens],
            device="cuda",
        )
        flow_input_len = torch.tensor([flow_input.shape[1]], device="cuda")
        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16 if self.model.config.hf_config.fp16_flow else torch.float32,
        ):
            mels, mels_lens = self.model.flow(
                flow_input,
                flow_input_len,
                synth_state["prompt_mel"],
                synth_state["prompt_mel_len"],
                synth_state["spk_emb"],
                streaming=streaming,
                finalize=finalize,
                n_timesteps=flow_steps,
            )
        mel = mels[:, :, synth_state["prompt_mel_len"][0].item(): mels_lens[0].item()]
        wav, _ = self.model.hift(speech_feat=mel)
        return wav

    def _run_mtp_in_thread(
        self,
        input_ids,
        sampling_params,
        eos_id,
        streamer,
        cuda_stream,
        seed: int,
        *,
        prefix_cache=None,
        prefix_hidden=None,
        prefix_len: int = 0,
    ):
        class _MTPThread(threading.Thread):
            def __init__(inner_self):
                super().__init__(daemon=True)
                inner_self.result = None
                inner_self.exc = None

            def run(inner_self):
                try:
                    with torch.cuda.stream(cuda_stream), torch.autocast("cuda", dtype=torch.bfloat16):
                        inner_self.result = mtp_speculative_sample_cached(
                            self.model.llm.model,
                            self.mtp,
                            input_ids,
                            max_new_tokens=sampling_params.max_tokens,
                            min_new_tokens=sampling_params.min_tokens,
                            eos_token_id=eos_id,
                            temperature=sampling_params.temperature,
                            top_k=sampling_params.top_k,
                            top_p=sampling_params.top_p,
                            repetition_penalty=sampling_params.repetition_penalty,
                            use_ras=sampling_params.use_ras,
                            ras_win_size=sampling_params.win_size,
                            ras_tau_r=sampling_params.tau_r,
                            allow_eos_from_drafts=False,
                            seed=seed,
                            streamer=streamer,
                            prefix_cache=prefix_cache,
                            prefix_hidden=prefix_hidden,
                            prefix_len=prefix_len,
                        )
                except BaseException as e:
                    inner_self.exc = e
                    streamer.end()

        thread = _MTPThread()
        thread.start()
        return thread

    def _stream_speech_pcm_mtp(
        self,
        request: SpeechRequest,
        prepared: Optional[Dict[str, Any]] = None,
    ) -> Iterator[bytes]:
        flow_streaming = request.flow_streaming if request.flow_streaming is not None else api_config.flow_streaming
        flow_steps = request.flow_steps if request.flow_steps is not None else api_config.flow_steps
        chunk_size = request.chunk_size or api_config.stream_chunk_size
        first_chunk_size = request.first_chunk_size or api_config.stream_first_chunk_size
        seed = self._resolve_seed(request)

        if self.trt_streaming_mode is not None and flow_streaming != self.trt_streaming_mode:
            raise ValueError(
                "TRT estimator was built for flow_streaming="
                f"{self.trt_streaming_mode}; request used {flow_streaming}"
            )

        self._seed_all(seed)

        prepared = prepared or self._build_speech_prepared(request)
        prompt_ids, eos_id, offset = self._build_first_turn_prompt(prepared)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        synth_state = self._prepare_synth_state(prepared)
        sampling_params = prepared["sampling_params"]
        prompt_entry = prepared["prompt_cache_entry"]
        prefix_cache = prompt_entry.get("prompt_prefix_cache")
        prefix_hidden = prompt_entry.get("prompt_prefix_hidden")
        prefix_len = prompt_entry.get("prompt_prefix_len", 0)
        if prefix_cache is not None:
            prefix_cache = self._fork_dynamic_cache(prefix_cache)

        streamer = SpeechTokenStreamer(eos_token_id=eos_id)
        mtp_stream = torch.cuda.Stream()
        flow_stream = torch.cuda.Stream()
        mtp_thread = self._run_mtp_in_thread(
            input_ids,
            sampling_params,
            eos_id,
            streamer,
            mtp_stream,
            seed,
            prefix_cache=prefix_cache,
            prefix_hidden=prefix_hidden,
            prefix_len=prefix_len,
        )

        accumulated_speech_tokens: List[int] = []
        prev_audio_len = 0
        try:
            for cur_chunk in streamer.iter_chunks(
                chunk_size=chunk_size,
                first_chunk_size=first_chunk_size,
            ):
                accumulated_speech_tokens.extend([t - offset for t in cur_chunk])
                with torch.cuda.stream(flow_stream):
                    wav_full = self._synthesize_chunk(
                        synth_state,
                        accumulated_speech_tokens,
                        finalize=False,
                        streaming=flow_streaming,
                        flow_steps=flow_steps,
                    )
                flow_stream.synchronize()
                new_audio = wav_full[:, prev_audio_len:].detach().cpu()
                prev_audio_len = wav_full.shape[-1]
                chunk_bytes = tensor_to_pcm16_bytes(new_audio)
                if chunk_bytes:
                    yield chunk_bytes

            mtp_thread.join()
            if mtp_thread.exc:
                raise mtp_thread.exc

            with torch.cuda.stream(flow_stream):
                wav_full = self._synthesize_chunk(
                    synth_state,
                    accumulated_speech_tokens,
                    finalize=True,
                    streaming=flow_streaming,
                    flow_steps=flow_steps,
                )
            flow_stream.synchronize()
            final_audio = wav_full[:, prev_audio_len:].detach().cpu()
            final_bytes = tensor_to_pcm16_bytes(final_audio)
            if final_bytes:
                yield final_bytes
        finally:
            if mtp_thread.is_alive():
                streamer.cancel()
                mtp_thread.join(timeout=1.0)

    def _generate_speech_pcm_trunk_with_prefix_cache(
        self,
        request: SpeechRequest,
        prepared: Dict[str, Any],
    ) -> Optional[bytes]:
        if not api_config.hf_prompt_prefix_cache or self.config.llm_engine != "hf":
            return None

        prompt_entry = prepared.get("prompt_cache_entry")
        if prompt_entry is None:
            return None
        prefix_cache = prompt_entry.get("prompt_prefix_cache")
        prefix_len = prompt_entry.get("prompt_prefix_len", 0)
        if prefix_cache is None or prefix_len <= 0:
            return None

        prompt_ids, eos_id, offset = self._build_first_turn_prompt(prepared)
        if prefix_len > len(prompt_ids):
            raise ValueError("Cached prompt prefix is longer than the generation prompt")
        target_tail_ids = prompt_ids[prefix_len:]
        if not target_tail_ids:
            return None

        sampling_params = prepared["sampling_params"]
        prefix_cache = self._fork_dynamic_cache(prefix_cache)
        with torch.no_grad():
            llm_outputs = self.model.llm.generate(
                target_tail_ids,
                sampling_params,
                past_key_values=prefix_cache,
            )

        generated_ids = list(llm_outputs["token_ids"])
        if generated_ids and generated_ids[-1] == eos_id:
            generated_ids = generated_ids[:-1]
        generated_speech_tokens = [token - offset for token in generated_ids]
        synth_state = self._prepare_synth_state(prepared)
        wav = self._synthesize_chunk(
            synth_state,
            generated_speech_tokens,
            finalize=True,
            streaming=False,
            flow_steps=15,
        )
        return tensor_to_pcm16_bytes(wav.detach().cpu())

    def _generate_speech_pcm_trunk(
        self,
        request: SpeechRequest,
        prepared: Optional[Dict[str, Any]] = None,
    ) -> Iterator[bytes]:
        self._seed_all(self._resolve_seed(request))
        prepared = prepared or self._build_speech_prepared(request)
        if request.stream:
            yield from self._stream_speech_pcm_trunk(request, prepared)
            return

        cached_pcm = None
        if self.config.llm_engine == "hf":
            cached_pcm = self._generate_speech_pcm_trunk_with_prefix_cache(request, prepared)
        if cached_pcm is not None:
            yield cached_pcm
            return

        with torch.no_grad():
            results_dict = self.model.forward_longform(**prepared)
        target_audio = None
        for wav in results_dict["generated_wavs"]:
            target_audio = wav if target_audio is None else torch.concat([target_audio, wav], axis=1)
        if target_audio is not None:
            yield tensor_to_pcm16_bytes(target_audio)

    def _stream_speech_pcm_trunk(
        self,
        request: SpeechRequest,
        prepared: Optional[Dict[str, Any]] = None,
    ) -> Iterator[bytes]:
        flow_streaming = request.flow_streaming if request.flow_streaming is not None else api_config.flow_streaming
        flow_steps = request.flow_steps if request.flow_steps is not None else api_config.flow_steps
        chunk_size = request.chunk_size or api_config.stream_chunk_size
        first_chunk_size = request.first_chunk_size or api_config.stream_first_chunk_size

        if self.trt_streaming_mode is not None and flow_streaming != self.trt_streaming_mode:
            raise ValueError(
                "TRT estimator was built for flow_streaming="
                f"{self.trt_streaming_mode}; request used {flow_streaming}"
            )

        prepared = prepared or self._build_speech_prepared(request)
        for event in self.model.forward_longform_streaming(
            **prepared,
            chunk_size=chunk_size,
            first_chunk_size=first_chunk_size,
            flow_streaming=flow_streaming,
            flow_steps=flow_steps,
        ):
            chunk_bytes = tensor_to_pcm16_bytes(event["audio"])
            if chunk_bytes:
                yield chunk_bytes

    def stream_speech_pcm(
        self,
        request: SpeechRequest,
        prepared: Optional[Dict[str, Any]] = None,
    ) -> Iterator[bytes]:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        with self._speech_lock:
            # MTP speculative sampling calls mtp_speculative_sample_cached with
            # self.model.llm.model, which must be an nn.Module. VLLMEngine.model
            # is an LLMEngine object, not an nn.Module, so MTP+vLLM is not yet
            # supported. Fall back to trunk synthesis when vLLM is active.
            use_mtp = self.mtp is not None and self.config.llm_engine == "hf"
            if use_mtp:
                yield from self._stream_speech_pcm_mtp(request, prepared)
            else:
                yield from self._generate_speech_pcm_trunk(request, prepared)

    def stream_speech_bytes(
        self,
        request: SpeechRequest,
        prepared: Optional[Dict[str, Any]] = None,
    ) -> Iterator[bytes]:
        if request.output_format == "wav":
            yield wav_header(None)
        elif request.output_format != "pcm":
            raise ValueError(f"Unsupported format: {request.output_format}")
        for pcm in self.stream_speech_pcm(request, prepared):
            yield pcm

    def generate_speech_bytes(
        self,
        request: SpeechRequest,
        prepared: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        pcm = b"".join(self.stream_speech_pcm(request, prepared))
        if request.output_format == "pcm":
            return pcm
        return wav_bytes_from_pcm(pcm)

    def stream_speech_podcast(
        self,
        prompt_audio_paths: List[str],
        prompt_texts: List[str],
        dialogue_text: str,
        seed: int = 1988,
        temperature: float = 0.6,
        top_k: int = 100,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
        output_format: str = "wav",
        chunk_size: Optional[int] = None,
        first_chunk_size: Optional[int] = None,
    ) -> Iterator[bytes]:
        """Stream multi-speaker podcast synthesis as WAV/PCM chunks.

        Yields the WAV header immediately (unknown-length placeholder), then
        PCM16 chunks as each speech token window is decoded by the flow model.
        First chunk arrives after `STREAM_FIRST_CHUNK_SIZE` speech tokens are
        generated (~160 ms at 25 Hz with the default value of 4).
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")

        with self._generation_lock:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            num_speakers = len(prompt_audio_paths)
            target_text_list = parse_dialogue_text(dialogue_text, num_speakers)

            spks, texts = [], []
            for target_text in target_text_list:
                pattern = r'(\[S[1-9]\])(.+)'
                match = re.match(pattern, target_text)
                if match:
                    text, spk = match.group(2), int(match.group(1)[2]) - 1
                    spks.append(spk)
                    texts.append(text)
                else:
                    raise ValueError(f"Invalid dialogue text format: {target_text}")

            dataitem = {
                "key": "api_stream",
                "prompt_text": prompt_texts,
                "prompt_wav": prompt_audio_paths,
                "text": texts,
                "spk": spks,
            }
            self.dataset.update_datasource([dataitem])
            data = self.dataset[0]

            import s3tokenizer
            prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])
            spk_emb_for_flow = torch.tensor(data["spk_emb"])
            # Move flow mels to CUDA so _stream_synth_chunk always gets CUDA
            # tensors regardless of which branch the per-prompt mel-alignment
            # check takes (forward_longform does .cuda() at its flow call site;
            # forward_longform_streaming infers device from prompt_mel instead).
            device = "cuda" if torch.cuda.is_available() else "cpu"
            prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
                data["mel"], batch_first=True, padding_value=0
            ).to(device)
            text_tokens_for_llm = data["text_tokens"]
            prompt_text_tokens_for_llm = data["prompt_text_tokens"]
            spk_ids = data["spks_list"]

            sampling_params = SamplingParams(
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                use_ras=True,
                win_size=25,
                tau_r=0.2,
            )

            processed_data = {
                "prompt_mels_for_llm": prompt_mels_for_llm,
                "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
                "prompt_text_tokens_for_llm": prompt_text_tokens_for_llm,
                "text_tokens_for_llm": text_tokens_for_llm,
                "prompt_mels_for_flow_ori": prompt_mels_for_flow,
                "spk_emb_for_flow": spk_emb_for_flow,
                "sampling_params": sampling_params,
                "spk_ids": spk_ids,
                "use_dialect_prompt": False,
            }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if output_format == "wav":
                yield wav_header(None)

            for event in self.model.forward_longform_streaming(
                **processed_data,
                chunk_size=chunk_size if chunk_size is not None else api_config.stream_chunk_size,
                first_chunk_size=first_chunk_size if first_chunk_size is not None else api_config.stream_first_chunk_size,
                flow_streaming=api_config.flow_streaming,
                flow_steps=api_config.flow_steps,
            ):
                chunk_bytes = tensor_to_pcm16_bytes(event["audio"])
                if chunk_bytes:
                    yield chunk_bytes

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


_service: Optional[SoulXPodcastService] = None


def get_service() -> SoulXPodcastService:
    """Get the global service instance."""
    global _service
    if _service is None:
        _service = SoulXPodcastService()
    return _service
