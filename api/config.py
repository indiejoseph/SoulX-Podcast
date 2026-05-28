"""
API Configuration Management
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """API配置类"""

    # 模型配置
    model_path: str = os.getenv(
        "MODEL_PATH",
        "pretrained_models/SoulX-Podcast-1.7B"
    )
    llm_engine: str = os.getenv("LLM_ENGINE", "hf")  # hf or vllm

    def validate_llm_engine(self):
        """验证LLM引擎配置"""
        if self.enable_mtp and self.mtp_checkpoint and self.llm_engine != "hf":
            import logging
            logging.warning("MTP serving requires direct HF trunk access; forcing LLM_ENGINE=hf")
            self.llm_engine = "hf"

        if self.llm_engine not in ["hf", "vllm"]:
            raise ValueError(f"Invalid llm_engine: {self.llm_engine}. Must be 'hf' or 'vllm'")

        # 如果选择vllm，检查是否安装
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
    voice_registry_path: Optional[str] = os.getenv("VOICE_REGISTRY_PATH")
    default_voice_id: str = os.getenv("DEFAULT_VOICE_ID", "female_mandarin")
    default_voice_prompt_audio: str = os.getenv(
        "DEFAULT_VOICE_PROMPT_AUDIO",
        "example/audios/female_mandarin.wav",
    )
    default_voice_prompt_text: str = os.getenv(
        "DEFAULT_VOICE_PROMPT_TEXT",
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
    )
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

    # 服务配置
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"

    # 文件配置
    temp_dir: Path = Path("api/temp")
    output_dir: Path = Path("api/outputs")
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    file_cleanup_minutes: int = 30  # 文件过期时间（分钟）

    # 并发控制
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))

    # 默认生成参数
    default_seed: Optional[int] = (
        int(os.environ["DEFAULT_SEED"]) if os.getenv("DEFAULT_SEED") else None
    )
    default_temperature: float = 0.6
    default_top_k: int = 100
    default_top_p: float = 0.9

    def __post_init__(self):
        """确保目录存在并验证配置"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validate_llm_engine()


# 全局配置实例
config = APIConfig()
