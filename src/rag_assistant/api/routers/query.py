from __future__ import annotations

import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from rag_assistant.api.dependencies import (
    CacheDep,
    CitationParserDep,
    EmbedderDep,
    OllamaClientDep,
    PromptBuilderDep,
    RetrieverDep,
    SettingsDep,
)
from rag_assistant.models.api import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    settings: SettingsDep,
    embedder: EmbedderDep,
    cache: CacheDep,
    retriever: RetrieverDep,
    prompt_builder: PromptBuilderDep,
    ollama: OllamaClientDep,
    citation_parser: CitationParserDep,
) -> QueryResponse:
    """Answer a natural language question about an indexed codebase.

    Checks both cache levels before hitting the retrieval + generation path.
    Returns the answer with structured citations and a cache_hit indicator.
    """
    # Embed query once — used for both cache lookup and retrieval
    query_embedding = embedder.embed_query(request.query)

    # Cache lookup
    cached, hit = await cache.get(request.query, query_embedding)
    if cached is not None:
        return QueryResponse(
            answer=cached.answer,
            citations=cached.citations,
            cache_hit=hit,  # type: ignore[arg-type]
            retrieval=None,
        )

    # Retrieval
    top_k = request.top_k or settings.rerank_top_n
    retrieval = await retriever.retrieve(
        request.query,
        repo_url=request.repo_url,
        top_k=top_k,
    )

    # Generation
    messages = prompt_builder.build(request.query, retrieval.results)
    answer = await ollama.generate(messages)
    citations = citation_parser.parse(answer)

    response = QueryResponse(
        answer=answer,
        citations=citations,
        cache_hit="miss",
        retrieval=retrieval,
    )

    # Store in cache for future requests
    await cache.set(request.query, query_embedding, response)

    return response


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    settings: SettingsDep,
    embedder: EmbedderDep,
    cache: CacheDep,
    retriever: RetrieverDep,
    prompt_builder: PromptBuilderDep,
    ollama: OllamaClientDep,
    citation_parser: CitationParserDep,
) -> EventSourceResponse:
    """Stream the answer token by token as Server-Sent Events.

    Each SSE event carries one text fragment. A final 'done' event is
    sent with the full citations payload once generation is complete.
    """
    query_embedding = embedder.embed_query(request.query)

    # Cache hit — stream cached answer as a single event
    cached, hit = await cache.get(request.query, query_embedding)
    if cached is not None:
        async def cached_stream():
            yield {"event": "token", "data": cached.answer}
            yield {
                "event": "done",
                "data": json.dumps({
                    "citations": [c.model_dump() for c in cached.citations],
                    "cache_hit": hit,
                }),
            }
        return EventSourceResponse(cached_stream())

    # Retrieval
    top_k = request.top_k or settings.rerank_top_n
    retrieval = await retriever.retrieve(
        request.query,
        repo_url=request.repo_url,
        top_k=top_k,
    )
    messages = prompt_builder.build(request.query, retrieval.results)

    async def generate_stream():
        full_answer: list[str] = []

        async for token in ollama.stream_generate(messages):
            full_answer.append(token)
            yield {"event": "token", "data": token}

        answer = "".join(full_answer)
        citations = citation_parser.parse(answer)

        complete_response = QueryResponse(
            answer=answer,
            citations=citations,
            cache_hit="miss",
            retrieval=retrieval,
        )
        await cache.set(request.query, query_embedding, complete_response)

        yield {
            "event": "done",
            "data": json.dumps({
                "citations": [c.model_dump() for c in citations],
                "cache_hit": "miss",
            }),
        }

    return EventSourceResponse(generate_stream())
