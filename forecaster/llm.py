"""Language-model calls through OpenRouter, with each call's cost read and charged to the budget."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from .budget import Budget

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class ModelError(Exception):
    def __init__(self, status: int = 0):
        super().__init__(f"model call failed ({status})")
        self.status = status

    @property
    def is_auth(self) -> bool:
        return self.status in (401, 403)

    @property
    def is_credit(self) -> bool:
        return self.status == 402


@dataclass
class Reply:
    text: str
    cost: float
    model: str


@dataclass
class Member:
    """One forecaster in an ensemble: a model and how hard it should think."""
    model: str
    effort: str = "high"
    max_tokens: int = 24000
    estimate_usd: float = 0.15


class ModelClient:
    def __init__(self, api_key: str, budget: Budget, max_concurrency: int = 8, timeout: float = 600.0):
        self.budget = budget
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "forecast-bot"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(self, question_id: int, member: Member, prompt: str,
                       web_search: bool = False) -> Reply:
        self.budget.reserve(question_id, member.estimate_usd)
        cost = 0.0
        try:
            data = await self._call(member, prompt, web_search)
            usage = data.get("usage") or {}
            # A reply without a reported cost is charged at the member's estimate.
            cost = float(usage.get("cost") or 0.0) or member.estimate_usd
        except ModelError as exc:
            # A failed call may still have been billed; charge a share of the estimate only
            # when the provider answered (status known), never for transport failures.
            cost = 0.0 if exc.status in (0, 401, 402, 403, 429) else member.estimate_usd * 0.25
            raise
        finally:
            self.budget.settle(question_id, member.estimate_usd, cost)
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        if not text.strip():
            raise ModelError(-1)
        return Reply(text=text, cost=cost, model=member.model)

    async def _call(self, member: Member, prompt: str, web_search: bool) -> dict:
        body: dict = {
            "model": member.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": member.max_tokens,
            "usage": {"include": True},
        }
        if member.effort:
            body["reasoning"] = {"effort": member.effort}
        if web_search:
            body["plugins"] = [{"id": "web", "engine": "native", "max_results": 6}]
        async with self._sem:
            return await self._post(body)

    async def _post(self, body: dict) -> dict:
        delay = 5.0
        for attempt in range(3):
            try:
                response = await self._client.post("/chat/completions", json=body)
            except httpx.TransportError:
                if attempt == 2:
                    raise ModelError(0)
                await asyncio.sleep(delay)
                delay *= 3
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    raise ModelError(response.status_code)
                await asyncio.sleep(delay)
                delay *= 3
                continue
            if response.status_code >= 400:
                raise ModelError(response.status_code)
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                code = data["error"].get("code") if isinstance(data["error"], dict) else 0
                raise ModelError(int(code) if isinstance(code, int) else 0)
            return data
        raise ModelError(0)

    async def credit_left(self) -> float | None:
        """What the key may still spend, if the key has a limit."""
        try:
            response = await self._client.get("/key")
            if response.status_code != 200:
                return None
            data = response.json().get("data") or {}
            remaining = data.get("limit_remaining")
            return float(remaining) if remaining is not None else None
        except (httpx.HTTPError, ValueError):
            return None
