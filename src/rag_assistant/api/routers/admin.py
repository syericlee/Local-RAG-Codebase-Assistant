from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from rag_assistant.api.dependencies import JobStoreDep, PipelineDep
from rag_assistant.models.api import IndexRequest, JobStatus

router = APIRouter(prefix="/admin", tags=["admin"])


async def _run_pipeline(
    pipeline,
    job_store,
    job_id: str,
    request: IndexRequest,
) -> None:
    """Background task: run ingestion and update job state throughout."""
    await job_store.mark_running(job_id)
    try:
        def on_progress(progress):
            import asyncio
            asyncio.create_task(
                job_store.update_progress(
                    job_id,
                    files_indexed=progress.files_processed,
                    chunks_upserted=progress.chunks_upserted,
                )
            )

        result = await pipeline.run(
            repo_url=request.repo_url,
            branch=request.branch,
            force_reindex=request.force_reindex,
            progress_callback=on_progress,
        )
        await job_store.mark_completed(
            job_id,
            files_indexed=result.files_indexed,
            chunks_upserted=result.chunks_upserted,
        )
    except Exception as exc:
        await job_store.mark_failed(job_id, error=str(exc))
        raise


@router.post("/index", response_model=JobStatus, status_code=202)
async def index_repo(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    pipeline: PipelineDep,
    job_store: JobStoreDep,
) -> JobStatus:
    """Kick off background indexing for a GitHub repo.

    Returns HTTP 202 immediately with a job_id. Poll
    GET /admin/status/{job_id} to track progress.
    Returns HTTP 409 if a job for this repo is already running.
    """
    running_id = await job_store.get_running_job_for_repo(request.repo_url)
    if running_id:
        raise HTTPException(
            status_code=409,
            detail=f"A job is already running for this repo: {running_id}",
        )

    job = await job_store.create_job(request.repo_url)
    background_tasks.add_task(
        _run_pipeline, pipeline, job_store, job.job_id, request
    )
    return job


@router.get("/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str, job_store: JobStoreDep) -> JobStatus:
    """Return the current status of an indexing job."""
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job
