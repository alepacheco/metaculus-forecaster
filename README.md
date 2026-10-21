# forecaster

An automated forecaster for the Metaculus bot tournaments (FutureEval seasons and MiniBench).
It runs on its own: every 20 minutes it lists the open questions of the tournaments it is
entered in, skips those it has already answered, and for each new one gathers research,
asks an ensemble of language models for forecasts, combines them and posts the forecast with
a private comment explaining it. Nobody reviews or edits a forecast.

## How it forecasts

- **Research** from two sources: a news search API (AskNews) for the latest and background
  news, and a model with native web search that writes a short brief.
- **Ensemble**: five forecasts from models of several labs (through OpenRouter), each asked to
  work from an outside view (base rates) to an inside view (the specific evidence).
- **Aggregation**: binary questions take the median, capped to 2-98%; multiple-choice
  questions take each option's median with a 1% floor, renormalised; numeric, discrete and
  date questions average the forecasters' distributions, smooth them lightly and turn them
  into a valid 201-point (or discrete-size) CDF within Metaculus's per-bin cap.
- MiniBench questions, and every question once 80% of the season's budget is spent, use a
  cheaper ensemble without the web-search brief.

## Running it

```
docker build -t forecaster .
docker run -v $PWD/live:/live -v $PWD/secret:/secret:ro forecaster
```

`/secret/bot.env` holds `METACULUS_TOKEN`, `OPENROUTER_API_KEY` and `ASKNEWS_API_KEY`.
`/live/config.toml` holds only where to forecast and how much may be spent:

```toml
tournaments = ["fall-futureeval-2026", "minibench"]
season_budget_usd = 150
per_question_cap_usd = 2.0
budget_label = "fall-2026"
```

Commands: `python -m forecaster run | pass | dry-run | analyse | check` (see
`forecaster/__main__.py`). `dry-run` forecasts closed or resolved questions without posting;
with `--backtest` its research is limited to news published before each question opened and
web search is off. `analyse` scores the bot on resolved questions.

## What it writes

- `live/health.json` each pass: counts of questions seen, done, skipped and failed in the last
  24 hours, spend, credit left, budget warning, and the last error's type.
- `live/bot.log`: the same counts, and errors as their type, place, question id and type.
- `live/records/<question id>.json`: each question's full record, used by the bot to finish a
  question interrupted mid-way and by `analyse` once the question has closed.

## Exit codes

`2` a key, credit or API failure lasting three passes; `3` the season's budget is spent;
`4` every question failed three passes in a row; `5` a pass took over an hour.

## Licence

MIT. The CDF standardisation follows the Metaculus forecasting-tools library (MIT); see NOTICE.
