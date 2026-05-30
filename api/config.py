"""
API Configuration Management
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass
class APIConfig:
    """API configuration."""

    # Model configuration
    model_path: str = os.getenv(
        "MODEL_PATH",
        "pretrained_models/SoulX-Podcast-1.7B"
    )
    llm_engine: str = os.getenv("LLM_ENGINE", "hf")  # hf or vllm
    vllm_enforce_eager: bool = _env_bool("VLLM_ENFORCE_EAGER", False)
    # vLLM's default 0.9 leaves ~2.4GiB headroom on 24GB cards, which is not
    # enough for the flow chunk cache + flow model + HiFT on long content
    # (causes OOM on subsequent requests). 0.7 leaves ~7GiB which is safe.
    vllm_gpu_memory_utilization: float = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.7"))
    hf_prompt_prefix_cache: bool = _env_bool("HF_PROMPT_PREFIX_CACHE", False)

    def validate_llm_engine(self):
        """Validate LLM engine configuration."""
        if self.enable_mtp and self.mtp_checkpoint and self.llm_engine != "hf":
            import logging
            logging.warning("MTP serving requires direct HF trunk access; forcing LLM_ENGINE=hf")
            self.llm_engine = "hf"

        if self.llm_engine not in ["hf", "vllm"]:
            raise ValueError(f"Invalid llm_engine: {self.llm_engine}. Must be 'hf' or 'vllm'")

        if self.llm_engine == "vllm":
            try:
                import vllm
            except ImportError:
                import logging
                logging.warning("vLLM not installed, falling back to HuggingFace engine")
                self.llm_engine = "hf"
    fp16_flow: bool = os.getenv("FP16_FLOW", "false").lower() == "true"

    # API security
    # SOULX_API_KEY is the legacy name; API_KEY takes precedence.
    api_key: str = (os.getenv("API_KEY") or os.getenv("SOULX_API_KEY", "")).strip()
    require_api_key: bool = _env_bool("REQUIRE_API_KEY", True)
    cors_allowed_origins: tuple[str, ...] = _env_csv("CORS_ALLOWED_ORIGINS")
    cors_allow_credentials: bool = _env_bool("CORS_ALLOW_CREDENTIALS", False)

    # Redis-backed coordination. GPU-resident prompt tensors still stay local to
    # each API process; Redis stores task state and prompt cache metadata.
    redis_url: str = os.getenv("REDIS_URL", "").strip()
    redis_key_prefix: str = os.getenv("REDIS_KEY_PREFIX", "tts").strip() or "tts"
    task_ttl_seconds: int = int(os.getenv("TASK_TTL_SECONDS", "86400"))
    prompt_cache_ttl_seconds: int = int(os.getenv("PROMPT_CACHE_TTL_SECONDS", "3600"))
    prompt_cache_store_inline_audio: bool = _env_bool("PROMPT_CACHE_STORE_INLINE_AUDIO", False)

    # OpenAI-compatible speech endpoint
    prompt_audio_root: Optional[str] = os.getenv("PROMPT_AUDIO_ROOT")
    prompt_cache_size: int = int(os.getenv("PROMPT_CACHE_SIZE", "16"))
    mtp_checkpoint: str = os.getenv("MTP_CHECKPOINT", "")
    enable_mtp: bool = os.getenv("ENABLE_MTP", "true").lower() == "true"
    stream_chunk_size: int = int(os.getenv("STREAM_CHUNK_SIZE", "100"))
    stream_first_chunk_size: int = int(os.getenv("STREAM_FIRST_CHUNK_SIZE", "4"))
    flow_streaming: bool = os.getenv("FLOW_STREAMING", "true").lower() == "true"
    flow_steps: int = int(os.getenv("FLOW_STEPS", "8"))
    trt_estimator: bool = os.getenv("TRT_ESTIMATOR", "false").lower() == "true"
    trt_onnx: str = os.getenv("TRT_ONNX", "")
    trt_plan: str = os.getenv("TRT_PLAN", "")
    trt_opt_mel_len: int = int(os.getenv("TRT_OPT_MEL_LEN", "256"))

    # torch.compile optimisation for flow estimator + HiFT vocoder.
    # First inference triggers JIT compilation (~30-60s); warmup runs at startup.
    torch_compile: bool = _env_bool("TORCH_COMPILE", False)

    # Service configuration
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"

    # File configuration
    temp_dir: Path = Path("api/temp")
    output_dir: Path = Path("api/outputs")
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    file_cleanup_minutes: int = 30

    # Concurrency control
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))

    # Default generation parameters
    default_seed: Optional[int] = (
        int(os.environ["DEFAULT_SEED"]) if os.getenv("DEFAULT_SEED") else None
    )
    default_temperature: float = 0.6
    default_top_k: int = 100
    default_top_p: float = 0.9

    def __post_init__(self):
        """Create required directories and validate configuration."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validate_llm_engine()
        self.validate_cors_config()

    def validate_cors_config(self):
        """Validate CORS settings that would be unsafe in browsers."""
        if "*" in self.cors_allowed_origins and self.cors_allow_credentials:
            raise ValueError("CORS_ALLOW_CREDENTIALS=true cannot be used with CORS_ALLOWED_ORIGINS=*")

    def validate_runtime_security(self):
        """Validate settings that should fail server startup in production mode.

        Called from the FastAPI lifespan, not __post_init__, so that scripts and
        tests that import config without starting a server are not required to
        set API_KEY.
        """
        if not self.require_api_key:
            return
        if not self.api_key:
            raise RuntimeError("API_KEY is required when REQUIRE_API_KEY=true")
        weak_key = self.api_key.lower()
        if (
            weak_key in {"changeme", "change-me", "dev", "test"}
            or weak_key.startswith("change-me")
            or weak_key.endswith("-dev-key")
        ):
            raise RuntimeError("API_KEY must be changed from the development placeholder")


# Global configuration instance
config = APIConfig()
