"""Test doubles for the Metaculus API and the model provider."""

from __future__ import annotations

import copy

from forecaster.llm import ModelError, Reply
from forecaster.metaculus import Question

__all__ = ["Question", "FakeApi", "FakeModels"]


def binary_q(qid=1, tournament="fall-test", **kw) -> Question:
    return Question(id=qid, post_id=qid + 1000, type="binary", title="Will it rain?", tournament=tournament, **kw)


def choice_q(qid=2, tournament="fall-test") -> Question:
    return Question(id=qid, post_id=qid + 1000, type="multiple_choice", title="Which party?",
                    tournament=tournament, options=["Red Party", "Blue Party", "Other"])


def numeric_q(qid=3, tournament="fall-test", **kw) -> Question:
    base = dict(range_min=0.0, range_max=100.0, nominal_min=0.0, nominal_max=100.0,
                open_lower_bound=True, open_upper_bound=True, inbound_outcome_count=200, unit="units")
    base.update(kw)
    return Question(id=qid, post_id=qid + 1000, type="numeric", title="How many?", tournament=tournament, **base)


def log_q(qid=4) -> Question:
    return numeric_q(qid, range_min=1.0, range_max=10000.0, zero_point=0.0,
                     open_lower_bound=False, open_upper_bound=False)


def discrete_q(qid=5) -> Question:
    return Question(id=qid, post_id=qid + 1000, type="discrete", title="How many officials?", tournament="fall-test",
                    range_min=-0.5, range_max=15.5, nominal_min=0, nominal_max=15,
                    open_lower_bound=False, open_upper_bound=False, inbound_outcome_count=16)


def date_q(qid=6) -> Question:
    return Question(id=qid, post_id=qid + 1000, type="date", title="When?", tournament="fall-test",
                    range_min=1767225600.0, range_max=1830297600.0,  # 2026-01-01 .. 2028-01-01
                    open_lower_bound=False, open_upper_bound=True, inbound_outcome_count=200)


def reply_for(question: Question, marker: str = "") -> str:
    """A well-formed model reply for the question's type."""
    head = f"Outside view and inside view considered. {marker}\nSummary: Rain is common this time of year. The forecast follows recent reports.\n"
    if question.type == "binary":
        return head + "Probability: 43.21%"
    if question.type == "multiple_choice":
        return head + "Red Party: 52.5%\nBlue Party: 40.25%\nOther: 7.25%"
    if question.type == "date":
        dates = ["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01",
                 "2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01", "2026-12-01", "2027-02-01", "2027-06-01"]
        return head + "\n".join(f"Percentile {p}: {d}" for p, d in zip((1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99), dates))
    if question.type == "discrete":
        vals = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 8, 9, 12]
    elif question.zero_point is not None:
        vals = [3, 8, 15, 30, 50, 80, 120, 170, 240, 350, 600, 900, 2000]
    else:
        vals = [5, 12, 18, 27, 33, 38, 43.21, 48, 53, 60, 70, 77, 90]
    return head + "\n".join(f"Percentile {p}: {v}" for p, v in zip((1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99), vals))


class FakeModels:
    def __init__(self, budget, replies=None, fail_models=(), cost=0.01, gone_models=(), listed=None):
        self.budget = budget
        self.replies = replies  # callable(question_id, member, prompt) -> text, or None for well-formed
        self.fail_models = set(fail_models)
        self.gone_models = set(gone_models)  # answer "not found", as a withdrawn model does
        self.listed = listed  # the provider's list of models, or None when it cannot be read
        self.cost = cost
        self.calls = []
        self.questions: dict[int, Question] = {}

    async def complete(self, question_id, member, prompt, web_search=False):
        self.calls.append((question_id, member.model, web_search))
        self.budget.reserve(question_id, member.estimate_usd)
        cost = 0.0
        try:
            if member.model in self.gone_models:
                raise ModelError(404)
            if member.model in self.fail_models:
                raise ModelError(500)
            cost = self.cost
            if self.replies is not None:
                text = self.replies(question_id, member, prompt)
            elif web_search:
                text = "Recent reports say rain is likely."
            else:
                text = reply_for(self.questions[question_id])
        finally:
            self.budget.settle(question_id, member.estimate_usd, cost)
        return Reply(text=text, cost=cost, model=member.model)

    async def listed_models(self):
        return set(self.listed) if self.listed is not None else None

    async def credit_left(self):
        return 42.0

    async def key_usage(self):
        return None

    async def close(self):
        pass


class FakeApi:
    def __init__(self, questions=(), on_site=()):
        self.questions = {q.id: q for q in questions}
        self.on_site = set(on_site)  # question ids with a forecast already on the site
        self.forecasts = []
        self.comments = []
        self.list_error = None

    async def open_questions(self, tournament):
        if self.list_error:
            raise self.list_error
        found = []
        for q in self.questions.values():
            if q.tournament == tournament:
                q = copy.copy(q)
                q.has_my_forecast = q.id in self.on_site
                found.append(q)
        return found, 0

    async def has_my_forecast(self, post_id):
        return any(q.post_id == post_id and q.id in self.on_site for q in self.questions.values())

    async def post_forecast(self, question, value):
        self.forecasts.append((question.id, value))
        self.on_site.add(question.id)

    async def post_comment(self, post_id, text):
        self.comments.append((post_id, text))

    async def close(self):
        pass
