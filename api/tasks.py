"""
Async Task Management System
"""
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import scipy.io.wavfile as wavfile

from api.models import TaskStatus
from api.config import config
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
            self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
            self.workers: List[asyncio.Task] = []
            self._initialized = True
            logger.info(f"TaskManager initialized with {config.max_concurrent_tasks} concurrent tasks")

    def start_workers(self, num_workers: int = None):
        """Start background workers."""
        if num_workers is None:
            num_workers = config.max_concurrent_tasks

        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            logger.info(f"Started worker-{i}")

    async def _worker(self, worker_name: str):
        """Background worker loop."""
        logger.info(f"{worker_name} started")

        while True:
            try:
                task_id = await self.queue.get()

                if task_id not in self.tasks:
                    logger.warning(f"{worker_name}: Task {task_id} not found")
                    self.queue.task_done()
                    continue

                task = self.tasks[task_id]

                async with self.semaphore:
                    logger.info(f"{worker_name}: Processing task {task_id}")
                    await self._process_task(task)

                self.queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"{worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"{worker_name} error: {e}", exc_info=True)
                self.queue.task_done()

    async def _process_task(self, task: Task):
        """Process a single task."""
        try:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            task.progress = 10
            logger.info(f"Task {task.task_id} started processing")

            loop = asyncio.get_event_loop()
            service = get_service()

            task.progress = 20

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
            logger.info(f"Task {task.task_id} generation completed")

            output_filename = f"{task.task_id}.wav"
            output_path = config.output_dir / output_filename
            wavfile.write(str(output_path), sample_rate, audio_array)

            task.progress = 100
            task.result_path = output_path
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            duration = (task.completed_at - task.started_at).total_seconds()
            logger.info(f"Task {task.task_id} completed in {duration:.2f}s")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)

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

        self.tasks[task_id] = task
        await self.queue.put(task_id)
        logger.info(f"Task {task_id} added to queue. Queue size: {self.queue.qsize()}")

        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task metadata."""
        return self.tasks.get(task_id)

    def get_active_task_count(self) -> int:
        """Get the number of active tasks."""
        return sum(
            1 for task in self.tasks.values()
            if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]
        )

    async def shutdown(self):
        """Shut down the task manager."""
        logger.info("Shutting down TaskManager...")

        await self.queue.join()

        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("TaskManager shutdown completed")


_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
