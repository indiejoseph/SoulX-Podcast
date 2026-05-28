"""
Pydantic Data Models for API
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    """Generation request model for JSON bodies used with file uploads."""
    prompt_texts: List[str] = Field(
        ...,
        description="Reference transcripts. Must match the number of uploaded prompt audio files.",
        min_items=1,
        max_items=4
    )
    dialogue_text: str = Field(
        ...,
        description="Dialogue text to synthesize. Single-speaker text may be plain; multi-speaker text uses [S1], [S2], etc.",
        min_length=1
    )
    seed: Optional[int] = Field(
        default=1988,
        description="Random seed for reproducible generation."
    )
    temperature: Optional[float] = Field(
        default=0.6,
        ge=0.1,
        le=2.0,
        description="Sampling temperature."
    )
    top_k: Optional[int] = Field(
        default=100,
        ge=1,
        le=500,
        description="Top-k sampling parameter."
    )
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter."
    )
    repetition_penalty: Optional[float] = Field(
        default=1.25,
        ge=1.0,
        le=2.0,
        description="Repetition penalty."
    )

    @validator('dialogue_text')
    def validate_dialogue_text(cls, v):
        """Validate dialogue text."""
        if not v.strip():
            raise ValueError("dialogue_text must not be empty")
        return v.strip()


class SpeechRequest(BaseModel):
    """OpenAI-compatible `/v1/audio/speech` request body."""
    model: str = Field(..., description="Model name. Accepted for API compatibility.")
    prompt_cache_id: Optional[str] = Field(
        default=None,
        description=(
            "Opaque prompt cache id returned in the Prompt-Cache-Id response header. "
            "When set, omit prompt_audio and prompt_text."
        ),
    )
    prompt_audio: Optional[str] = Field(
        default=None,
        description=(
            "Prompt audio as trusted server-local file:// URI, "
            "data:audio/*;base64 URI, or raw base64 audio bytes. Requires prompt_text. "
            "file:// paths may be restricted by PROMPT_AUDIO_ROOT."
        ),
    )
    prompt_text: Optional[str] = Field(
        default=None,
        description="Transcript for prompt_audio. Required when prompt_audio is set.",
    )
    input: str = Field(..., min_length=1, description="Text to synthesize")
    language: Optional[str] = Field(default=None, description="Optional BCP-47 language code")
    format: Literal["wav", "pcm"] = Field(default="wav", description="Output audio format")
    response_format: Optional[Literal["wav", "pcm"]] = Field(
        default=None,
        description="OpenAI-compatible alias. Overrides `format` when set.",
    )
    stream: bool = Field(
        default=True,
        description=(
            "Stream audio chunks as they are synthesized. WAV streaming uses a "
            "placeholder-length header; use format=pcm or stream=false for stricter clients."
        ),
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional sampling seed. If omitted, the service picks a random seed per request.",
    )
    temperature: Optional[float] = Field(default=0.6, ge=0.1, le=2.0)
    top_k: Optional[int] = Field(default=100, ge=1, le=500)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.25, ge=1.0, le=2.0)
    chunk_size: Optional[int] = Field(default=None, ge=1)
    first_chunk_size: Optional[int] = Field(default=None, ge=1)
    flow_streaming: Optional[bool] = Field(default=None)
    flow_steps: Optional[int] = Field(default=None, ge=1)

    @property
    def output_format(self) -> str:
        return self.response_format or self.format


class TaskCreateResponse(BaseModel):
    """Async task creation response."""
    task_id: str = Field(..., description="Unique task identifier.")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status.")
    created_at: datetime = Field(..., description="Task creation timestamp.")
    message: str = Field(default="Task created", description="Human-readable status message.")


class TaskStatusResponse(BaseModel):
    """Task status response."""
    task_id: str = Field(..., description="Unique task identifier.")
    status: TaskStatus = Field(..., description="Task status.")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage.")
    result_url: Optional[str] = Field(None, description="Download URL for the generated result.")
    error: Optional[str] = Field(None, description="Error message, if the task failed.")
    created_at: datetime = Field(..., description="Task creation timestamp.")
    started_at: Optional[datetime] = Field(None, description="Task start timestamp.")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp.")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "progress": 100,
                "result_url": "/download/123e4567-e89b-12d3-a456-426614174000.wav",
                "error": None,
                "created_at": "2025-11-01T12:00:00Z",
                "started_at": "2025-11-01T12:00:01Z",
                "completed_at": "2025-11-01T12:00:15Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status.")
    model_loaded: bool = Field(..., description="Whether the model is loaded.")
    gpu_available: bool = Field(..., description="Whether CUDA/GPU is available.")
    llm_engine: str = Field(..., description="Active LLM engine: hf or vllm.")
    active_tasks: int = Field(default=0, description="Number of active async tasks.")
    version: str = Field(default="1.0.0", description="API version.")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type.")
    message: str = Field(..., description="Detailed error message.")
    task_id: Optional[str] = Field(None, description="Related task id, when available.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp.")
