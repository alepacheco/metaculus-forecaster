"""The Metaculus API: list questions, read one, post a forecast and its comment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

API_BASE = "https://www.metaculus.com/api"
SUPPORTED_TYPES = ("binary", "multiple_choice", "numeric", "discrete", "date")


class ApiError(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status

    @property
    def is_auth(self) -> bool:
        return self.status in (401, 403)


@dataclass
class Question:
    id: int
    post_id: int
    type: str
    title: str
    tournament: str
    status: str = "open"
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    unit: str = ""
    options: list[str] = field(default_factory=list)
    open_time: str | None = None
    scheduled_close_time: str | None = None
    actual_close_time: str | None = None
    resolution: object = None
    range_min: float | None = None
    range_max: float | None = None
    nominal_min: float | None = None
    nominal_max: float | None = None
    zero_point: float | None = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    inbound_outcome_count: int | None = None
    has_my_forecast: bool = False

    @property
    def cdf_size(self) -> int:
        return (self.inbound_outcome_count or 200) + 1

    @property
    def is_continuous(self) -> bool:
        return self.type in ("numeric", "discrete", "date")

    def open_timestamp(self) -> int | None:
        if not self.open_time:
            return None
        return int(datetime.fromisoformat(self.open_time.replace("Z", "+00:00")).timestamp())

    @classmethod
    def from_post(cls, post: dict, tournament: str) -> "Question | None":
        q = post.get("question")
        if not q or q.get("type") not in SUPPORTED_TYPES:
            return None
        scaling = q.get("scaling") or {}
        my = q.get("my_forecasts") or {}
        latest = my.get("latest") if isinstance(my, dict) else None
        return cls(
            id=int(q["id"]),
            post_id=int(post["id"]),
            type=q["type"],
            title=q.get("title") or post.get("title") or "",
            tournament=tournament,
            status=q.get("status") or post.get("status") or "",
            description=q.get("description") or "",
            resolution_criteria=q.get("resolution_criteria") or "",
            fine_print=q.get("fine_print") or "",
            unit=q.get("unit") or "",
            options=list(q.get("options") or []),
            open_time=q.get("open_time") or post.get("open_time"),
            scheduled_close_time=q.get("scheduled_close_time") or post.get("scheduled_close_time"),
            actual_close_time=q.get("actual_close_time"),
            resolution=q.get("resolution"),
            range_min=scaling.get("range_min"),
            range_max=scaling.get("range_max"),
            nominal_min=scaling.get("nominal_min"),
            nominal_max=scaling.get("nominal_max"),
            zero_point=scaling.get("zero_point"),
            open_lower_bound=bool(scaling.get("open_lower_bound", q.get("open_lower_bound", False))),
            open_upper_bound=bool(scaling.get("open_upper_bound", q.get("open_upper_bound", False))),
            inbound_outcome_count=scaling.get("inbound_outcome_count") or q.get("inbound_outcome_count"),
            has_my_forecast=bool(latest and latest.get("forecast_values")),
        )


def forecast_payload(question: Question, value: object) -> dict:
    """The body of one forecast for /questions/forecast/."""
    body = {"question": question.id, "source": "api", "probability_yes": None,
            "probability_yes_per_category": None, "continuous_cdf": None}
    if question.type == "binary":
        body["probability_yes"] = value
    elif question.type == "multiple_choice":
        body["probability_yes_per_category"] = value
    else:
        body["continuous_cdf"] = value
    return body


class MetaculusClient:
    def __init__(self, token: str, base: str = API_BASE, timeout: float = 60.0):
        self._client = httpx.AsyncClient(
            base_url=base, timeout=timeout, headers={"Authorization": f"Token {token}"}
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> dict | list:
        delay = 2.0
        for attempt in range(4):
            response = await self._client.request(method, path, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 3:
                    raise ApiError(response.status_code)
                await asyncio.sleep(delay)
                delay *= 3
                continue
            if response.status_code >= 400:
                raise ApiError(response.status_code)
            return response.json() if response.content else {}
        raise ApiError(0)

    async def me(self) -> dict:
        return await self._request("GET", "/users/me/")

    async def list_posts(self, tournament: str, statuses: str = "open", limit: int = 100,
                         max_posts: int = 1000, order_by: str = "-open_time") -> list[dict]:
        posts: list[dict] = []
        offset = 0
        # A page can come back with fewer results than asked (or none) and still have a next page.
        for _ in range(50):
            if len(posts) >= max_posts:
                break
            page = await self._request("GET", "/posts/", params={
                "tournaments": tournament,
                "statuses": statuses,
                "forecast_type": ",".join(SUPPORTED_TYPES),
                "include_description": "true",
                "order_by": order_by,
                "limit": limit,
                "offset": offset,
            })
            results = page.get("results", []) if isinstance(page, dict) else []
            posts.extend(results)
            if not isinstance(page, dict) or not page.get("next"):
                break
            offset += limit
        return posts[:max_posts]

    async def open_questions(self, tournament: str) -> tuple[list[Question], int]:
        """Open questions of a tournament, and how many open posts were not plain questions."""
        posts = await self.list_posts(tournament, statuses="open")
        questions, unsupported = [], 0
        for post in posts:
            question = Question.from_post(post, tournament)
            if question is None:
                unsupported += 1
            elif question.status == "open":
                questions.append(question)
        return questions, unsupported

    async def get_post(self, post_id: int) -> dict:
        return await self._request("GET", f"/posts/{post_id}/")

    async def question(self, post_id: int, tournament: str) -> Question | None:
        return Question.from_post(await self.get_post(post_id), tournament)

    async def has_my_forecast(self, post_id: int) -> bool:
        question = await self.question(post_id, "")
        return bool(question and question.has_my_forecast)

    async def post_forecast(self, question: Question, value: object) -> None:
        await self._request("POST", "/questions/forecast/", json=[forecast_payload(question, value)])

    async def post_comment(self, post_id: int, text: str) -> None:
        await self._request("POST", "/comments/create/", json={
            "text": text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        })


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
