from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from redis.asyncio import Redis

from rag_assistant.config import Settings
from rag_assistant.models.api import JobStatus

_JOB_PREFIX = "job:"
_JOB_TTL = 60 * 60 * 24  # 24 hours


def _job_key(job_id: str) -> str:
    return f"{_JOB_PREFIX}{job_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RedisJobStore:
    """Stores background indexing job state in Redis hashes.

    Each job is a Redis Hash at job:{job_id} with string fields
    matching the JobStatus model. Jobs expire after 24 hours.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @classmethod
    def from_settings(cls, settings: Settings) -> RedisJobStore:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        return cls(redis)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_job(self, repo_url: str) -> JobStatus:
        """Create a new job in 'pending' state and return its JobStatus."""
        job_id = str(uuid.uuid4())
        now = _now_iso()
        status = JobStatus(
            job_id=job_id,
            status="pending",
            repo_url=repo_url,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )
        await self._save(status)
        return status

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Return the current JobStatus, or None if not found."""
        data = await self._redis.hgetall(_job_key(job_id))
        if not data:
            return None
        return self._deserialize(data)

    async def get_running_job_for_repo(self, repo_url: str) -> Optional[str]:
        """Return the job_id of any currently running job for repo_url, or None."""
        # Scan all job keys — acceptable since job count is small
        async for key in self._redis.scan_iter(f"{_JOB_PREFIX}*"):
            data = await self._redis.hgetall(key)
            if data.get("repo_url") == repo_url and data.get("status") == "running":
                return data.get("job_id")
        return None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def mark_running(self, job_id: str) -> None:
        await self._update(job_id, {"status": "running"})

    async def update_progress(
        self, job_id: str, files_indexed: int, chunks_upserted: int
    ) -> None:
        await self._update(job_id, {
            "files_indexed": str(files_indexed),
            "chunks_upserted": str(chunks_upserted),
        })

    async def mark_completed(
        self, job_id: str, files_indexed: int, chunks_upserted: int
    ) -> None:
        await self._update(job_id, {
            "status": "completed",
            "files_indexed": str(files_indexed),
            "chunks_upserted": str(chunks_upserted),
        })

    async def mark_failed(self, job_id: str, error: str) -> None:
        await self._update(job_id, {"status": "failed", "error": error})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _save(self, status: JobStatus) -> None:
        key = _job_key(status.job_id)
        await self._redis.hset(key, mapping=self._serialize(status))
        await self._redis.expire(key, _JOB_TTL)

    async def _update(self, job_id: str, fields: dict[str, str]) -> None:
        key = _job_key(job_id)
        fields["updated_at"] = _now_iso()
        await self._redis.hset(key, mapping=fields)

    @staticmethod
    def _serialize(status: JobStatus) -> dict[str, str]:
        return {
            "job_id": status.job_id,
            "status": status.status,
            "repo_url": status.repo_url,
            "created_at": status.created_at.isoformat(),
            "updated_at": status.updated_at.isoformat(),
            "files_indexed": str(status.files_indexed),
            "chunks_upserted": str(status.chunks_upserted),
            "error": status.error or "",
        }

    @staticmethod
    def _deserialize(data: dict[str, str]) -> JobStatus:
        return JobStatus(
            job_id=data["job_id"],
            status=data["status"],  # type: ignore[arg-type]
            repo_url=data["repo_url"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            files_indexed=int(data.get("files_indexed", 0)),
            chunks_upserted=int(data.get("chunks_upserted", 0)),
            error=data.get("error") or None,
        )

    async def close(self) -> None:
        await self._redis.aclose()
