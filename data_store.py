"""Local storage for teacher-calibrated answer pairs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from settings import DATA_DIR, TEACHER_DATA_PATH


DATA_PATH = TEACHER_DATA_PATH


def save_teacher_score(reference: str, student: str, score: float) -> None:
    """Append one teacher-labelled example for supervised GNN training."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "reference": reference.strip(),
        "student": student.strip(),
        "teacher_score": round(float(score), 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with DATA_PATH.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")


def load_teacher_scores() -> list[dict]:
    """Load valid labelled examples from every local JSONL dataset."""
    records = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if {"reference", "student", "teacher_score"} <= record.keys():
                    records.append(record)
            except json.JSONDecodeError:
                continue
    return records
