"""Research before forecasting: news search and a model with native web search.

For a backtest (`as_of` set) only the news archive up to that moment is used and web search
is off, since a live web search would find how the question resolved.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from . import logs
from .budget import BudgetExhausted
from .llm import Member, ModelClient
from .metaculus import Question
from .prompts import research_prompt

ASKNEWS_BASE = "https://api.asknews.app"


class NewsClient:
    def __init__(self, api_key: str, timeout: float = 60.0):
        self._client = httpx.AsyncClient(
            base_url=ASKNEWS_BASE, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, n_articles: int, strategy: str,
                     end_timestamp: int | None = None) -> str:
        params: dict = {
            "query": query[:400],
            "n_articles": n_articles,
            "return_type": "string",
            "method": "nl",
            "strategy": strategy,
        }
        if end_timestamp is not None:
            params.update({"historical": "true", "end_timestamp": end_timestamp,
                           "start_timestamp": end_timestamp - 60 * 86400, "strategy": "default"})
        response = await self._client.get("/v1/news/search", params=params)
        response.raise_for_status()
        return (response.json() or {}).get("as_string") or ""


async def gather(question: Question, models: ModelClient, news: NewsClient | None,
                 search_member: Member | None, as_of: int | None = None) -> str:
    """Research text for a question; a source that fails is skipped."""
    tasks: dict[str, asyncio.Future] = {}
    if news is not None:
        if as_of is None:
            tasks["Latest news"] = asyncio.ensure_future(news.search(question.title, 8, "latest news"))
            tasks["Background news"] = asyncio.ensure_future(news.search(question.title, 10, "news knowledge"))
        else:
            tasks["News archive"] = asyncio.ensure_future(news.search(question.title, 12, "default", end_timestamp=as_of))
    if search_member is not None and as_of is None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tasks["Web research"] = asyncio.ensure_future(
            models.complete(question.id, search_member, research_prompt(question, today), web_search=True)
        )
    sections = []
    for name, task in tasks.items():
        try:
            result = await task
        except BudgetExhausted:
            raise
        except Exception as exc:  # one source failing never stops the forecast
            logs.error(exc, stage="research", question_id=question.id, question_type=question.type)
            continue
        text = result.text if hasattr(result, "text") else result
        if text and text.strip():
            sections.append(f"## {name}\n{text.strip()[:12000]}")
    return "\n\n".join(sections) if sections else "No research could be gathered for this question."
