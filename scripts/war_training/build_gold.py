#!/usr/bin/env python3
"""Write de-identified WAR gold JSON under results/.agent/patient_regimen_review/gold/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from gold_seed import (  # noqa: E402
    CLINICAL_QUESTIONS,
    DDI_RANK_GOLD,
    GUIDELINE_ELIGIBILITY,
    MISMATCH_TABLE,
    PATIENT_STATE,
    TOOL_TRACE_LABELS,
)

OUT_DIR = ROOT / "results" / ".agent" / "patient_regimen_review" / "gold"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {
        "patient_state.json": PATIENT_STATE,
        "mismatch_table.json": MISMATCH_TABLE,
        "ddi_rank_gold.json": DDI_RANK_GOLD,
        "guideline_eligibility.json": GUIDELINE_ELIGIBILITY,
        "tool_trace_labels.json": TOOL_TRACE_LABELS,
        "clinical_questions.json": CLINICAL_QUESTIONS,
    }
    for name, payload in payloads.items():
        path = OUT_DIR / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
