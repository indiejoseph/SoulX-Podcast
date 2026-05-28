"""
Async task management.

Redis is used for durable task metadata and queue coordination when REDIS_URL is
configured. The in-process queue remains available for local development.
"""
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import scipy.io.wavfile as wavfile

from api.models import TaskStatus
from api.config import config
from api.redis_state import (
    close_async_redis_client,
    get_async_redis_client,
    get_redis_client,
    redis_enabled,
    redis_key,
)
from api.service import get_service

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Async generation task."""
    task_id: str
    prompt_audio_paths: List[str]
    prompt_texts: List[str]
    dialogue_text: str
    seed: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float

    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    result_path: Optional[Path] = None
    error: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


def _dt_to_str(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _dt_from_str(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _task_to_json(task: Task) -> str:
    return json.dumps(
        {
            "task_id": task.task_id,
            "prompt_audio_paths": task.prompt_audio_paths,
            "prompt_texts": task.prompt_texts,
            "dialogue_text": task.dialogue_text,
            "seed": task.seed,
            "temperature": task.temperature,
            "top_k": task.top_k,
            "top_p": task.top_p,
            "repetition_penalty": task.repetition_penalty,
            "status": task.status.value if isinstance(task.status, TaskStatus) else task.status,
            "progress": task.progress,
            "result_path": str(task.result_path) if task.result_path else None,
            "error": task.error,
            "created_at": _dt_to_str(task.created_at),
            "started_at": _dt_to_str(task.started_at),
            "completed_at": _dt_to_str(task.completed_at),
        },
        ensure_ascii=False,
    )


def _task_from_json(payload: str) -> Task:
    data = json.loads(payload)
    return Task(
        task_id=data["task_id"],
        prompt_audio_paths=list(data.get("prompt_audio_paths") or []),
        prompt_texts=list(data.get("prompt_texts") or []),
        dialogue_text=data.get("dialogue_text", ""),
        seed=int(data.get("seed", 1988)),
        temperature=float(data.get("temperature", 0.6)),
        top_k=int(data.get("top_k", 100)),
        top_p=float(data.get("top_p", 0.9)),
        repetition_penalty=float(data.get("repetition_penalty", 1.25)),
        status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
        progress=int(data.get("progress", 0)),
        result_path=Path(data["result_path"]) if data.get("result_path") else None,
        error=data.get("error"),
        created_at=_dt_from_str(data.get("created_at")) or datetime.now(),
        started_at=_dt_from_str(data.get("started_at")),
        completed_at=_dt_from_str(data.get("completed_at")),
    )


class TaskManager:
    """Singleton async task manager."""

    _instance: Optional['TaskManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.tasks: Dict[str, Task] = {}
            self.queue: asyncio.Queue = asyncio.Queue(maxsize=100)
            self.queue_key = redis_key("task_queue")
            self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
            self.workers: List[asyncio.Task] = []
            self.startup_tasks: List[asyncio.Task] = []
            self._initialized = True
            logger.info(
                "TaskManager initialized with %d concurrent task(s), redis=%s",
                config.max_concurrent_tasks,
                redis_enabled(),
            )

    def _task_key(self, task_id: str) -> str:
        return redis_key("task", task_id)

    async def _persist_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task
        redis = await get_async_redis_client()
        if redis is None:
            return
        await redis.set(
            self._task_key(task.task_id),
            _task_to_json(task),
            ex=config.task_ttl_seconds,
        )

    def _persist_task_sync_best_effort(self, task: Task) -> None:
        self.tasks[task.task_id] = task
        redis = get_redis_client()
        if redis is None:
            return
        try:
            redis.set(
                self._task_key(task.task_id),
                _task_to_json(task),
                ex=config.task_ttl_seconds,
            )
        except Exception:
            logger.exception("Failed to persist task %s to Redis", task.task_id)

    def start_workers(self, num_workers: int = None):
        """Start background workers."""
        if num_workers is None:
            num_workers = config.max_concurrent_tasks

        if redis_enabled():
            self.startup_tasks.append(asyncio.create_task(self._recover_interrupted_tasks()))

        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            logger.info("Started worker-%d", i)

    async def _recover_interrupted_tasks(self) -> None:
        """Mark tasks left processing by a previous crashed API process as failed."""
        redis = await get_async_redis_client()
        if redis is None:
            return
        pattern = redis_key("task", "*")
        async for key in redis.scan_iter(match=pattern, count=100):
            raw = await redis.get(key)
            if not raw:
                continue
            try:
                task = _task_from_json(raw)
            except Exception:
                logger.warning("Skipping unreadable task state at %s", key)
                continue
            if task.status == TaskStatus.PROCESSING:
                task.status = TaskStatus.FAILED
                task.error = "Task was interrupted by API restart"
                task.completed_at = datetime.now()
                await self._persist_task(task)

    async def _next_task_id(self) -> tuple[Optional[str], bool]:
        redis = await get_async_redis_client()
        if redis is not None:
            # timeout=1 keeps the loop responsive to CancelledError; each idle
            # worker wakes at most once per second when the queue is empty.
            item = await redis.blpop(self.queue_key, timeout=1)
            if item is None:
                return None, False
            return item[1], False
        return await self.queue.get(), True

    async def _worker(self, worker_name: str):
        """Background worker loop."""
        logger.info("%s started", worker_name)

        while True:
            local_queue_item = False
            try:
                task_id, local_queue_item = await self._next_task_id()
                if task_id is None:
                    continue

                task = self.get_task(task_id)
                if task is None:
                    logger.warning("%s: Task %s not found", worker_name, task_id)
                    continue

                async with self.semaphore:
                    logger.info("%s: Processing task %s", worker_name, task_id)
                    await self._process_task(task)

            except asyncio.CancelledError:
                logger.info("%s cancelled", worker_name)
                break
            except Exception as e:
                logger.error("%s error: %s", worker_name, e, exc_info=True)
            finally:
                if local_queue_item:
                    self.queue.task_done()

    async def _process_task(self, task: Task):
        """Process a single task."""
        try:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            task.progress = 10
            await self._persist_task(task)
            logger.info("Task %s started processing", task.task_id)

            loop = asyncio.get_event_loop()
            service = get_service()

            task.progress = 20
            await self._persist_task(task)

            sample_rate, audio_array = await loop.run_in_executor(
                None,
                service.generate,
                task.prompt_audio_paths,
                task.prompt_texts,
                task.dialogue_text,
                task.seed,
                task.temperature,
                task.top_k,
                task.top_p,
                task.repetition_penalty,
            )

            task.progress = 80
            await self._persist_task(task)
            logger.info("Task %s generation completed", task.task_id)

            output_filename = f"{task.task_id}.wav"
            output_path = config.output_dir / output_filename
            wavfile.write(str(output_path), sample_rate, audio_array)

            task.progress = 100
            task.result_path = output_path
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            await self._persist_task(task)

            duration = (task.completed_at - task.started_at).total_seconds()
            logger.info("Task %s completed in %.2fs", task.task_id, duration)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            await self._persist_task(task)
            logger.error("Task %s failed: %s", task.task_id, e, exc_info=True)

    async def create_task(
        self,
        task_id: str,
        prompt_audio_paths: List[str],
        prompt_texts: List[str],
        dialogue_text: str,
        seed: int = 1988,
        temperature: float = 0.6,
        top_k: int = 100,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
    ) -> Task:
        """Create a task and enqueue it."""
        task = Task(
            task_id=task_id,
            prompt_audio_paths=prompt_audio_paths,
            prompt_texts=prompt_texts,
            dialogue_text=dialogue_text,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        await self._persist_task(task)
        redis = await get_async_redis_client()
        if redis is not None:
            await redis.rpush(self.queue_key, task_id)
            queue_size = await redis.llen(self.queue_key)
        else:
            await self.queue.put(task_id)
            queue_size = self.queue.qsize()
        logger.info("Task %s added to queue. Queue size: %s", task_id, queue_size)

        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task metadata."""
        task = self.tasks.get(task_id)
        if task is not None:
            return task

        redis = get_redis_client()
        if redis is None:
            return None
        try:
            raw = redis.get(self._task_key(task_id))
        except Exception:
            logger.exception("Failed to read task %s from Redis", task_id)
            return None
        if not raw:
            return None
        task = _task_from_json(raw)
        self.tasks[task.task_id] = task
        return task

    def get_active_task_count(self) -> int:
        """Get the number of active tasks."""
        redis = get_redis_client()
        if redis is None:
            return sum(
                1 for task in self.tasks.values()
                if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]
            )

        active = 0
        try:
            for key in redis.scan_iter(match=redis_key("task", "*"), count=100):
                raw = redis.get(key)
                if not raw:
                    continue
                try:
                    status = json.loads(raw).get("status")
                except json.JSONDecodeError:
                    continue
                if status in {TaskStatus.PENDING.value, TaskStatus.PROCESSING.value}:
                    active += 1
        except Exception:
            logger.exception("Failed to read active task count from Redis")
        return active

    def queue_size(self) -> int:
        """Return current queue size."""
        redis = get_redis_client()
        if redis is None:
            return self.queue.qsize()
        try:
            return int(redis.llen(self.queue_key))
        except Exception:
            logger.exception("Failed to read queue size from Redis")
            return self.queue.qsize()

    async def shutdown(self):
        """Shut down the task manager."""
        logger.info("Shutting down TaskManager...")

        if not redis_enabled():
            await self.queue.join()

        for worker in self.workers:
            worker.cancel()
        for task in self.startup_tasks:
            task.cancel()

        await asyncio.gather(*self.workers, *self.startup_tasks, return_exceptions=True)
        await close_async_redis_client()
        logger.info("TaskManager shutdown completed")


_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
