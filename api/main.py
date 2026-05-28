"""
FastAPI main application for the TTS API.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import json
import threading

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import scipy.io.wavfile as wavfile

from api.config import config
from api.models import (
    TaskCreateResponse,
    TaskStatusResponse,
    HealthResponse,
    ErrorResponse,
    TaskStatus,
    SpeechRequest,
)
from api.service import get_service
from api.tasks import get_task_manager
from api.redis_state import async_ping_redis
from api.utils import (
    generate_task_id,
    save_upload_file,
    validate_audio_files,
    validate_dialogue_format,
    cleanup_old_files,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global lock for synchronous inference concurrency.
inference_lock = threading.Lock()
active_inferences = 0
MAX_CONCURRENT_SYNC_INFERENCES = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Starting TTS API...")
    config.validate_runtime_security()

    logger.info("Loading model...")
    service = get_service()
    if not service.is_loaded():
        raise RuntimeError("Failed to load model")

    task_manager = get_task_manager()
    task_manager.start_workers(config.max_concurrent_tasks)

    async def cleanup_task():
        while True:
            await asyncio.sleep(600)
            count = cleanup_old_files(config.temp_dir, config.file_cleanup_minutes)
            count += cleanup_old_files(config.output_dir, config.file_cleanup_minutes)
            if count > 0:
                logger.info("Cleaned up %d old files", count)

    cleanup_task_handle = asyncio.create_task(cleanup_task())

    logger.info("API started successfully!")

    yield

    logger.info("Shutting down API...")
    cleanup_task_handle.cancel()

    try:
        await asyncio.wait_for(task_manager.shutdown(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Task manager shutdown timeout, forcing exit")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("API shutdown completed")


app = FastAPI(
    title="TTS API",
    description="Voice cloning and text-to-speech API.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS is disabled unless explicitly configured. This keeps the API safe when
# exposed behind a proxy and lets deployments opt in per frontend origin.
if config.cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_allowed_origins),
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Prompt-Cache-Id"],
    )


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token guard for all generation and artifact endpoints."""
    if not config.require_api_key:
        return
    if not config.api_key:
        raise HTTPException(status_code=500, detail="API_KEY is required but not configured")
    expected = f"Bearer {config.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "name": "TTS API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check."""
    service = get_service()
    task_manager = get_task_manager()
    redis_available = await async_ping_redis()
    model_loaded = service.is_loaded()
    gpu_available = torch.cuda.is_available()
    healthy = model_loaded and gpu_available and redis_available is not False

    response = HealthResponse(
        status="healthy" if healthy else "unhealthy",
        model_loaded=model_loaded,
        gpu_available=gpu_available,
        redis_available=redis_available,
        llm_engine=config.llm_engine,
        active_tasks=task_manager.get_active_task_count(),
        version="1.0.0"
    )
    if not healthy:
        raise HTTPException(status_code=503, detail=response.dict())
    return response


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_audio_speech(
    request: SpeechRequest,
    _: None = Depends(require_api_key),
):
    """OpenAI-compatible TTS endpoint.

    Request shape follows the `/v1/audio/speech` convention: JSON in, audio
    bytes out. `stream=true` returns chunked transfer from the MTP bi-stream
    path when `MTP_CHECKPOINT` is configured.
    """
    try:
        service = get_service()
        speech_context = service.prepare_speech_context(request)
        media_type = "audio/wav" if request.output_format == "wav" else "audio/pcm"
        headers = {
            "Content-Disposition": f'attachment; filename="speech.{request.output_format}"',
            "X-TTS-Model": request.model,
            "Prompt-Cache-Id": speech_context["prompt_cache_id"],
        }
        if request.stream:
            return StreamingResponse(
                service.stream_speech_bytes(request, speech_context["prepared"]),
                media_type=media_type,
                headers=headers,
            )
        return Response(
            content=service.generate_speech_bytes(request, speech_context["prepared"]),
            media_type=media_type,
            headers=headers,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("OpenAI-compatible speech request failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", tags=["Generation"])
async def generate_sync(
    prompt_audio: List[UploadFile] = File(..., description="Reference prompt audio files, 1-4 files."),
    prompt_texts: List[str] = Form(..., description='Reference transcripts, e.g. ["text 1", "text 2"].'),
    dialogue_text: str = Form(..., description="Dialogue text to synthesize."),
    seed: int = Form(default=1988, description="Random seed."),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="Sampling temperature."),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-k sampling parameter."),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter."),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="Repetition penalty."),
    _: None = Depends(require_api_key),
):
    """
    Synchronously generate speech and return the audio file.

    Intended for short outputs.
    """
    task_id = generate_task_id()

    try:
        validate_audio_files(prompt_audio)

        try:
            prompt_text_list = prompt_texts
            if not isinstance(prompt_text_list, list):
                raise ValueError("prompt_texts must be a JSON array")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid prompt_texts JSON: {str(e)}")

        if len(prompt_audio) != len(prompt_text_list):
            raise HTTPException(
                status_code=400,
                detail=f"Number of prompt audio files ({len(prompt_audio)}) does not match number of prompt transcripts ({len(prompt_text_list)})"
            )

        is_valid, error_msg = validate_dialogue_format(dialogue_text, len(prompt_audio))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        audio_paths = []
        for i, file in enumerate(prompt_audio):
            path = save_upload_file(file, task_id, i)
            audio_paths.append(str(path))

        logger.info("Sync generation started: task_id=%s, speakers=%d", task_id, len(audio_paths))

        service = get_service()
        sample_rate, audio_array = service.generate(
            prompt_audio_paths=audio_paths,
            prompt_texts=prompt_text_list,
            dialogue_text=dialogue_text,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        output_filename = f"{task_id}.wav"
        output_path = config.output_dir / output_filename
        wavfile.write(str(output_path), sample_rate, audio_array)

        logger.info("Sync generation completed: task_id=%s", task_id)

        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=output_filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Sync generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-async", response_model=TaskCreateResponse, tags=["Generation"])
async def generate_async(
    prompt_audio: List[UploadFile] = File(..., description="Reference prompt audio files, 1-4 files."),
    prompt_texts: str = Form(..., description="Reference transcripts as a JSON array."),
    dialogue_text: str = Form(..., description="Dialogue text to synthesize."),
    seed: int = Form(default=1988, description="Random seed."),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="Sampling temperature."),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-k sampling parameter."),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter."),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="Repetition penalty."),
    _: None = Depends(require_api_key),
):
    """
    Asynchronously generate speech and return a task id.

    Intended for longer outputs or batch jobs.
    """
    task_id = generate_task_id()

    try:
        validate_audio_files(prompt_audio)

        try:
            prompt_text_list = json.loads(prompt_texts)
            if not isinstance(prompt_text_list, list):
                raise ValueError("prompt_texts must be a JSON array")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid prompt_texts JSON: {str(e)}")

        if len(prompt_audio) != len(prompt_text_list):
            raise HTTPException(
                status_code=400,
                detail=f"Number of prompt audio files ({len(prompt_audio)}) does not match number of prompt transcripts ({len(prompt_text_list)})"
            )

        is_valid, error_msg = validate_dialogue_format(dialogue_text, len(prompt_audio))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        audio_paths = []
        for i, file in enumerate(prompt_audio):
            path = save_upload_file(file, task_id, i)
            audio_paths.append(str(path))

        task_manager = get_task_manager()
        task = await task_manager.create_task(
            task_id=task_id,
            prompt_audio_paths=audio_paths,
            prompt_texts=prompt_text_list,
            dialogue_text=dialogue_text,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        logger.info("Async task created: task_id=%s", task_id)

        return TaskCreateResponse(
            task_id=task_id,
            status=task.status,
            created_at=task.created_at,
            message=f"Task created. Queue size: {task_manager.queue_size()}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task creation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str, _: None = Depends(require_api_key)):
    """Get async task status."""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    result_url = None
    if task.status == TaskStatus.COMPLETED and task.result_path:
        result_url = f"/download/{task.result_path.name}"

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        result_url=result_url,
        error=task.error,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


@app.get("/download/{filename}", tags=["Download"])
async def download_file(filename: str, _: None = Depends(require_api_key)):
    """Download a generated audio file."""
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    output_root = config.output_dir.resolve()
    file_path = (config.output_dir / filename).resolve()
    if not file_path.is_relative_to(output_root):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level="info"
    )
