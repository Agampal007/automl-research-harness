"""Append-only experiment journal. The journal is the only state: a run that is
killed at any point resumes by replaying it."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class State:
    incumbent: dict | None = None  # the last accepted record
    incumbent_code: str = ""
    experiments: list[dict] = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)
    iteration: int = 0
    since_improvement: int = 0
    stopped: str | None = None


class Journal:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "journal.jsonl"
        (run_dir / "candidates").mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> dict:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def save_code(self, name: str, code: str) -> str:
        rel = f"candidates/{name}.py"
        (self.run_dir / rel).write_text(code, encoding="utf-8")
        return rel

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line from a hard kill
        return out

    def replay(self) -> State:
        state = State()
        for rec in self.records():
            kind = rec.get("type")
            state.iteration = max(state.iteration, rec.get("iter", 0))
            if kind == "baseline" or (kind == "experiment" and rec["status"] == "accepted"):
                state.incumbent = rec
                state.incumbent_code = (self.run_dir / rec["code_file"]).read_text(encoding="utf-8")
                state.since_improvement = 0
            elif kind == "experiment":
                state.since_improvement += 1
            if kind == "experiment":
                state.experiments.append(rec)
            elif kind == "probe":
                state.probes.append(rec)
            elif kind == "stop":
                state.stopped = rec["reason"]
            elif kind == "resume":
                state.stopped = None
        return state
