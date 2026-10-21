"""Runtime configuration: the tournaments to forecast and the budget caps.

Everything that decides how a forecast is made lives in code; the config file only says
where to forecast and how much may be spent, so entering a new season changes no code.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    tournaments: list[str]
    season_budget_usd: float
    per_question_cap_usd: float
    budget_label: str
    live_dir: Path
    secrets: dict[str, str] = field(default_factory=dict)

    @property
    def records_dir(self) -> Path:
        return self.live_dir / "records"


def load_secrets(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines. Values are never logged."""
    secrets: dict[str, str] = {}
    if not path.exists():
        return secrets
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        secrets[key.strip()] = value.strip().strip('"').strip("'")
    return secrets


def load_config(live_dir: Path | None = None, secret_file: Path | None = None) -> Config:
    live_dir = Path(live_dir or os.environ.get("FORECASTER_LIVE_DIR", "/live"))
    secret_file = Path(secret_file or os.environ.get("FORECASTER_SECRET_FILE", "/secret/bot.env"))
    raw = tomllib.loads((live_dir / "config.toml").read_text())
    tournaments = [str(t) for t in raw.get("tournaments", [])]
    if not tournaments:
        raise ValueError("config.toml lists no tournaments")
    season_budget = float(raw["season_budget_usd"])
    per_question = float(raw.get("per_question_cap_usd", 2.0))
    if season_budget < 0 or per_question <= 0:
        raise ValueError("budget caps must be positive")
    return Config(
        tournaments=tournaments,
        season_budget_usd=season_budget,
        per_question_cap_usd=per_question,
        budget_label=str(raw.get("budget_label", "season")),
        live_dir=live_dir,
        secrets=load_secrets(secret_file),
    )
