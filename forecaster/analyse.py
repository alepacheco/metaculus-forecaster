"""How the bot did on questions that have resolved.

A record is opened only once the API says its question has closed; scores are computed only
for resolved questions. The report compares each forecast with the outcome and, where the API
shows it, with the community forecast at close (a rough stand-in for the field).
"""

from __future__ import annotations

import math
import statistics

import numpy as np

from .metaculus import MetaculusClient
from .records import Records

CLOSED = ("closed", "resolved")


def _prob_of_outcome(qtype: str, value, resolution, question: dict) -> float | None:
    if resolution in (None, "", "annulled", "ambiguous"):
        return None
    if qtype == "binary":
        p = float(value)
        return p if str(resolution).lower() == "yes" else 1.0 - p
    if qtype == "multiple_choice":
        return float(value.get(str(resolution), 0.0)) or None
    cdf = np.asarray(value, dtype=float)
    scaling = question.get("scaling") or {}
    if resolution == "below_lower_bound":
        return float(cdf[0])
    if resolution == "above_upper_bound":
        return float(1.0 - cdf[-1])
    try:
        if qtype == "date":
            from datetime import datetime
            x = datetime.fromisoformat(str(resolution).replace("Z", "+00:00")).timestamp()
        else:
            x = float(resolution)
    except ValueError:
        return None
    lo, hi, zero = scaling.get("range_min"), scaling.get("range_max"), scaling.get("zero_point")
    if zero is None:
        loc = (x - lo) / (hi - lo)
    else:
        ratio = (hi - zero) / (lo - zero)
        loc = math.log((x - lo) * (ratio - 1) + (hi - lo), ratio) - math.log(hi - lo, ratio)
    n = len(cdf)
    i = min(max(int(loc * (n - 1)), 0), n - 2)
    return float(cdf[i + 1] - cdf[i])


def _community(qtype: str, question: dict):
    aggs = question.get("aggregations") or {}
    for method in ("recency_weighted", "unweighted"):
        latest = (aggs.get(method) or {}).get("latest") or {}
        values = latest.get("forecast_values")
        if not values:
            continue
        if qtype == "binary":
            return values[-1] if len(values) == 2 else None
        if qtype == "multiple_choice":
            return dict(zip(question.get("options") or [], values))
        return values
    return None


async def analyse(api: MetaculusClient, records: Records) -> dict:
    rows = []
    open_skipped = 0
    for qid, meta in records.index().items():
        try:
            post = await api.get_post(int(meta["post_id"]))
        except Exception:
            continue
        question = post.get("question") or {}
        if question.get("status") not in CLOSED:
            open_skipped += 1
            continue
        record = records.load(int(qid))
        if not record or record.get("value") is None:
            continue
        row = {"qid": int(qid), "tournament": meta.get("tournament"), "type": meta.get("type"),
               "ensemble": record.get("ensemble"), "status": question.get("status"),
               "cost_usd": record.get("cost_usd")}
        if question.get("status") == "resolved":
            p = _prob_of_outcome(meta["type"], record["value"], question.get("resolution"), question)
            if p is not None:
                row["log_score"] = math.log(max(p, 1e-6))
                cp = _community(meta["type"], question)
                if cp is not None:
                    pc = _prob_of_outcome(meta["type"], cp, question.get("resolution"), question)
                    if pc:
                        row["vs_community"] = 100 * (row["log_score"] - math.log(max(pc, 1e-6)))
            score = ((question.get("my_forecasts") or {}).get("score_data") or {})
            if "spot_peer_score" in score:
                row["spot_peer"] = score["spot_peer_score"]
        rows.append(row)

    def summarise(group: list[dict]) -> dict:
        out = {"questions": len(group), "resolved_scored": sum(1 for r in group if "log_score" in r)}
        for key in ("log_score", "vs_community", "spot_peer", "cost_usd"):
            vals = [r[key] for r in group if r.get(key) is not None]
            if vals:
                mean = statistics.fmean(vals)
                half = 1.96 * statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
                out[key] = {"mean": round(mean, 3), "n": len(vals),
                            "ci95": [round(mean - half, 3), round(mean + half, 3)] if half is not None else None}
        return out

    report = {"open_not_read": open_skipped, "all": summarise(rows), "by_type": {}, "by_tournament": {}}
    for key, field in (("by_type", "type"), ("by_tournament", "tournament")):
        for name in sorted({r[field] for r in rows if r.get(field)}):
            report[key][name] = summarise([r for r in rows if r.get(field) == name])
    return report
