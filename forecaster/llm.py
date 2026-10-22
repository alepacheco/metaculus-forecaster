"""Language-model calls through OpenRouter, with each call's cost read and charged to the budget."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date

import httpx

from .budget import Budget

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
TIMED_OUT = 408
NOT_FOUND = 404  # the model is not offered (any more)
EMPTY_REPLY = -1
EMPTY_REPLY_TRIES = 2  # an empty reply is asked for once more


class ModelError(Exception):
    def __init__(self, status: int = 0, reason: str | None = None):
        super().__init__(f"model call failed ({status})")
        self.status = status
        self.reason = reason  # for an empty reply, the provider's finish reason

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
    def __init__(self, api_key: str, budget: Budget, max_concurrency: int = 16, timeout: float = 600.0):
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
        """The member's reply. An empty one (a refusal, or reasoning that used up its tokens) is
        asked for once more before it counts as a failure."""
        cost = 0.0
        finish = None
        for _ in range(EMPTY_REPLY_TRIES):
            text, call_cost, finish = await self._complete_once(question_id, member, prompt, web_search)
            cost += call_cost
            if text.strip():
                return Reply(text=text, cost=cost, model=member.model)
        raise ModelError(EMPTY_REPLY, reason=finish)

    async def _complete_once(self, question_id: int, member: Member, prompt: str,
                             web_search: bool) -> tuple[str, float, str | None]:
        self.budget.reserve(question_id, member.estimate_usd)
        cost = 0.0
        try:
            data = await self._call(member, prompt, web_search)
            usage = data.get("usage") or {}
            # A reply without a reported cost is charged at the member's estimate.
            cost = float(usage.get("cost") or 0.0) or member.estimate_usd
        except ModelError as exc:
            # A refused request is not billed; one that timed out may have been, so it is
            # charged its estimate (the pass's reconciliation with the key corrects either way).
            cost = member.estimate_usd if exc.status == TIMED_OUT else 0.0
            raise
        except asyncio.CancelledError:
            cost = member.estimate_usd
            raise
        finally:
            self.budget.settle(question_id, member.estimate_usd, cost)
        choices = data.get("choices") or []
        if not choices:
            return "", cost, None
        text = (choices[0].get("message") or {}).get("content") or ""
        return text, cost, choices[0].get("finish_reason")

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
            except httpx.ReadTimeout:
                raise ModelError(TIMED_OUT)  # not retried: the timed-out call may be billed
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

    async def listed_models(self) -> set[str] | None:
        """The models the provider offers now (its public list, less any past its listed expiry),
        or None when the list cannot be read."""
        try:
            response = await self._client.get("/models")
            if response.status_code != 200:
                return None
            entries = response.json().get("data") or []
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        today = date.today().isoformat()
        listed = set()
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            expires = entry.get("expiration_date")
            if isinstance(expires, str) and expires[:10] < today:
                continue
            listed.add(str(entry["id"]))
        return listed or None

    async def _key_info(self) -> dict:
        try:
            response = await self._client.get("/key")
            if response.status_code != 200:
                return {}
            return response.json().get("data") or {}
        except (httpx.HTTPError, ValueError):
            return {}

    async def credit_left(self) -> float | None:
        """What the key may still spend, if the key has a limit."""
        remaining = (await self._key_info()).get("limit_remaining")
        return float(remaining) if remaining is not None else None

    async def key_usage(self) -> float | None:
        """What the key has spent in total, as the provider counts it."""
        usage = (await self._key_info()).get("usage")
        return float(usage) if usage is not None else None
