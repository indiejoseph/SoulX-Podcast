"""
API Configuration Management
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """API configuration."""

    # Model configuration
    model_path: str = os.getenv(
        "MODEL_PATH",
        "pretrained_models/SoulX-Podcast-1.7B"
    )
    llm_engine: str = os.getenv("LLM_ENGINE", "hf")  # hf or vllm

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

    # OpenAI-compatible speech endpoint
    api_key: str = os.getenv("SOULX_API_KEY", "")
    require_api_key: bool = os.getenv(
        "REQUIRE_API_KEY",
        "true" if os.getenv("SOULX_API_KEY") else "false",
    ).lower() == "true"
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


# Global configuration instance
config = APIConfig()
