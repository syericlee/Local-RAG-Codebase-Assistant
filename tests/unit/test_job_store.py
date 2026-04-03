from __future__ import annotations

import pytest
import fakeredis.aioredis

from rag_assistant.jobs.job_store import RedisJobStore

REPO_URL = "https://github.com/foo/bar"


@pytest.fixture
async def store() -> RedisJobStore:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    s = RedisJobStore(redis)
    yield s
    await redis.aclose()


class TestJobStoreCreate:
    @pytest.mark.asyncio
    async def test_create_returns_pending_status(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        assert job.status == "pending"
        assert job.repo_url == REPO_URL
        assert job.job_id != ""

    @pytest.mark.asyncio
    async def test_create_stores_in_redis(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        fetched = await store.get_job(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_each_job_has_unique_id(self, store: RedisJobStore) -> None:
        job1 = await store.create_job(REPO_URL)
        job2 = await store.create_job(REPO_URL)
        assert job1.job_id != job2.job_id


class TestJobStoreRead:
    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: RedisJobStore) -> None:
        result = await store.get_job("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_correct_repo_url(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        fetched = await store.get_job(job.job_id)
        assert fetched is not None
        assert fetched.repo_url == REPO_URL


class TestJobStoreUpdates:
    @pytest.mark.asyncio
    async def test_mark_running(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        await store.mark_running(job.job_id)
        fetched = await store.get_job(job.job_id)
        assert fetched is not None
        assert fetched.status == "running"

    @pytest.mark.asyncio
    async def test_update_progress(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        await store.update_progress(job.job_id, files_indexed=5, chunks_upserted=42)
        fetched = await store.get_job(job.job_id)
        assert fetched is not None
        assert fetched.files_indexed == 5
        assert fetched.chunks_upserted == 42

    @pytest.mark.asyncio
    async def test_mark_completed(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        await store.mark_completed(job.job_id, files_indexed=10, chunks_upserted=80)
        fetched = await store.get_job(job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.files_indexed == 10
        assert fetched.chunks_upserted == 80

    @pytest.mark.asyncio
    async def test_mark_failed(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        await store.mark_failed(job.job_id, error="something went wrong")
        fetched = await store.get_job(job.job_id)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.error == "something went wrong"

    @pytest.mark.asyncio
    async def test_error_is_none_when_not_set(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        fetched = await store.get_job(job.job_id)
        assert fetched is not None
        assert fetched.error is None


class TestJobStoreRunningCheck:
    @pytest.mark.asyncio
    async def test_no_running_job_returns_none(self, store: RedisJobStore) -> None:
        result = await store.get_running_job_for_repo(REPO_URL)
        assert result is None

    @pytest.mark.asyncio
    async def test_running_job_found(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        await store.mark_running(job.job_id)
        result = await store.get_running_job_for_repo(REPO_URL)
        assert result == job.job_id

    @pytest.mark.asyncio
    async def test_completed_job_not_returned(self, store: RedisJobStore) -> None:
        job = await store.create_job(REPO_URL)
        await store.mark_running(job.job_id)
        await store.mark_completed(job.job_id, 5, 20)
        result = await store.get_running_job_for_repo(REPO_URL)
        assert result is None
