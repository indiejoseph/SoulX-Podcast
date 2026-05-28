"""
Utility Functions for API
"""
import os
import re
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
from fastapi import UploadFile, HTTPException
import logging

from api.config import config

logger = logging.getLogger(__name__)


def generate_task_id() -> str:
    """Generate a unique task id."""
    return str(uuid.uuid4())


def save_upload_file(upload_file: UploadFile, task_id: str, index: int) -> Path:
    """Save an uploaded file to the temporary directory."""
    file_path = None
    try:
        file_extension = Path(upload_file.filename).suffix or ".wav"

        filename = f"{task_id}_prompt_{index}{file_extension}"
        file_path = config.temp_dir / filename

        bytes_written = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > config.max_upload_size:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File {upload_file.filename} exceeds the maximum size limit "
                            f"({config.max_upload_size / 1024 / 1024:.0f}MB)"
                        ),
                    )
                buffer.write(chunk)

        logger.info("Saved upload file to %s", file_path)
        return file_path

    except HTTPException:
        raise
    except Exception as e:
        if file_path is not None:
            file_path.unlink(missing_ok=True)
        logger.error("Failed to save upload file: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    finally:
        upload_file.file.close()


def validate_audio_files(files: List[UploadFile]) -> None:
    """Validate uploaded audio files."""
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one reference audio file is required")

    if len(files) > 4:
        raise HTTPException(status_code=400, detail="At most 4 speakers / audio files are supported")

    allowed_extensions = {".wav", ".mp3", ".flac", ".m4a"}
    for i, file in enumerate(files):
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} has an unsupported format. Supported formats: {', '.join(allowed_extensions)}"
            )

        if hasattr(file, 'size') and file.size and file.size > config.max_upload_size:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds the maximum size limit ({config.max_upload_size / 1024 / 1024}MB)"
            )


def validate_dialogue_format(dialogue_text: str, num_speakers: int) -> Tuple[bool, str]:
    """Validate dialogue text format."""
    dialogue_text = dialogue_text.strip()

    if num_speakers == 1:
        if len(dialogue_text) == 0:
            return False, "dialogue_text must not be empty"
        return True, ""

    speaker_pattern = r'\[S[1-4]\]'
    matches = re.findall(speaker_pattern, dialogue_text)

    if not matches:
        return False, "Multi-speaker dialogue must use speaker markers, e.g. [S1]Hello[S2]Hi"

    used_speakers = set()
    for match in matches:
        speaker_id = int(match[2])
        used_speakers.add(speaker_id)

        if speaker_id > num_speakers:
            return False, f"Dialogue uses speaker [S{speaker_id}], but only {num_speakers} reference audio file(s) were provided"

    return True, ""


def cleanup_old_files(directory: Path, minutes: int = 30) -> int:
    """Clean up expired files and return the number removed."""
    if not directory.exists():
        return 0

    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    cleaned_count = 0

    try:
        for file_path in directory.glob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if file_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info("Cleaned up old file: %s", file_path)
                    except Exception as e:
                        logger.warning("Failed to delete %s: %s", file_path, e)

    except Exception as e:
        logger.error("Error during cleanup: %s", e)

    return cleaned_count


def format_audio_duration(seconds: float) -> str:
    """Format an audio duration as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def parse_dialogue_text(dialogue_text: str, num_speakers: int) -> List[str]:
    """Parse dialogue text into per-speaker segments."""
    if num_speakers == 1:
        if not dialogue_text.startswith("[S1]"):
            return [f"[S1]{dialogue_text}"]
        else:
            return [dialogue_text]

    pattern = r'(\[S[1-4]\][^\[\]]*)'
    segments = re.findall(pattern, dialogue_text)

    segments = [seg.strip() for seg in segments if seg.strip()]

    return segments
