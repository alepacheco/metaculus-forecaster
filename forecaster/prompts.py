"""The prompts. Changing one is a new release of the bot."""

from __future__ import annotations

from .metaculus import Question

PERCENTILES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)


def _question_block(q: Question) -> str:
    parts = [f"Question: {q.title}"]
    if q.description:
        parts.append(f"Background:\n{q.description.strip()}")
    if q.resolution_criteria:
        parts.append(f"Resolution criteria:\n{q.resolution_criteria.strip()}")
    if q.fine_print:
        parts.append(f"Fine print:\n{q.fine_print.strip()}")
    if q.scheduled_close_time:
        parts.append(f"The question closes for forecasting at {q.scheduled_close_time}.")
    return "\n\n".join(parts)


def research_prompt(q: Question, today: str) -> str:
    return f"""You are helping a professional forecaster. Today is {today}.
Search the web and write a concise research brief (at most 500 words) for the question below.
Cover: the current status of the matter, the most recent relevant developments with their dates,
scheduled events before the question resolves, relevant base rates or historical frequencies,
and what prediction markets or expert forecasts say if anything. Cite sources inline.
Do not give your own forecast.

{_question_block(q)}"""


_METHOD = """Work as a careful, well-calibrated superforecaster:
1. Restate exactly what must happen for each outcome under the resolution criteria and fine print.
2. Note how much time is left until the question resolves.
3. Outside view: find a reference class and its base rate. The status quo usually persists over short horizons.
4. Inside view: adjust from the base rate using the specific evidence in the research, weighing how reliable and recent it is.
5. Consider what would make you wrong, in both directions, and avoid overconfidence. Extreme forecasts are costly when wrong."""


def _bounds_text(q: Question) -> str:
    unit = f" {q.unit}" if q.unit else ""
    if q.type == "date":
        from datetime import datetime, timezone
        lo = datetime.fromtimestamp(q.range_min, timezone.utc).strftime("%Y-%m-%d") if q.range_min is not None else "?"
        hi = datetime.fromtimestamp(q.range_max, timezone.utc).strftime("%Y-%m-%d") if q.range_max is not None else "?"
    else:
        lo = q.nominal_min if q.nominal_min is not None else q.range_min
        hi = q.nominal_max if q.nominal_max is not None else q.range_max
    lower = "can resolve below this (open bound)" if q.open_lower_bound else "cannot resolve below this (closed bound)"
    upper = "can resolve above this (open bound)" if q.open_upper_bound else "cannot resolve above this (closed bound)"
    return f"The question's range runs from {lo}{unit}, which {lower}, to {hi}{unit}, which {upper}."


def forecast_prompt(q: Question, research: str, today: str) -> str:
    head = f"""Today is {today}. You are forecasting the question below.

{_question_block(q)}

Research gathered for you:
{research}

{_METHOD}
"""
    summary = ("\nEnd with a line starting 'Summary:' that gives your main reasons in two to four plain sentences, "
               "written for a reader of the question.")
    if q.type == "binary":
        return head + summary + """
Then give your final answer on the last line exactly as: Probability: NN%"""
    if q.type == "multiple_choice":
        options = "\n".join(f"{o}: NN%" for o in q.options)
        return head + f"""
The options are: {", ".join(q.options)}.
Give every option a probability; they must add up to 100%, and no option that could still happen should get 0%.""" + summary + f"""
Then give your final answer in exactly this format, one option per line, in this order:
{options}"""
    if q.type == "date":
        lines = "\n".join(f"Percentile {p}: YYYY-MM-DD" for p in PERCENTILES)
        return head + f"""
{_bounds_text(q)}
Give your forecast as a distribution of dates. Set wide intervals: unknown unknowns are common.""" + summary + f"""
Then give your final answer in exactly this format, with dates increasing:
{lines}"""
    unit = f" in {q.unit}" if q.unit else ""
    discrete = ""
    if q.type == "discrete" and q.discrete_step:
        step = q.discrete_step
        discrete = (" The answer is a whole number." if abs(step - round(step)) < 1e-9 and round(step) == 1
                    else f" The answer is a multiple of {step:g}.")
    lines = "\n".join(f"Percentile {p}: X" for p in PERCENTILES)
    return head + f"""
{_bounds_text(q)}{discrete}
Give your forecast as a distribution{unit}. Set wide intervals: unknown unknowns are common.""" + summary + f"""
Then give your final answer in exactly this format, with plain numbers (no units, no thousands separators) increasing:
{lines}"""
