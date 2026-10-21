"""Each question's full record, kept by the bot for itself.

A record is written before anything is posted, so after a restart a half-done question is
finished from its record and never forecast again. Records are read only by the bot: to finish
a half-done question, and by `analyse` once the question has closed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class Records:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, question_id: int) -> Path:
        return self.directory / f"{int(question_id)}.json"

    def load(self, question_id: int) -> dict | None:
        path = self.path(question_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save(self, question_id: int, record: dict) -> None:
        path = self.path(question_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, default=str))
        os.replace(tmp, path)
        self._index(question_id, record)

    def _index(self, question_id: int, record: dict) -> None:
        """Which post and tournament each record belongs to, so analyse can check a question's
        status before opening its record."""
        index_path = self.directory / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        question = record.get("question", {})
        index[str(question_id)] = {"post_id": question.get("post_id"),
                                   "tournament": question.get("tournament"),
                                   "type": question.get("type")}
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=1))
        os.replace(tmp, index_path)

    def index(self) -> dict:
        index_path = self.directory / "index.json"
        return json.loads(index_path.read_text()) if index_path.exists() else {}
