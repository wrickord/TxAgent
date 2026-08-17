#!/usr/bin/env python3
"""Expand DDI rank examples from existing ToolUniverse dumps, optionally live.

Default path uses the 2026-08-13 dumps so WAR stays a holdout and we do not
re-download DrugBank XML. Pass --live-tu to query ToolUniverse xml tools.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gold_seed import DDI_RANK_GOLD, PATIENT_STATE  # noqa: E402

DUMP_DIR = ROOT / "results" / ".agent" / "patient_regimen_review" / "tool_dumps"
OUT_DIR = ROOT / "results" / ".agent" / "patient_regimen_review" / "gold"

REGIMEN_NAMES = {
    "minoxidil",
    "baclofen",
    "finasteride",
    "lansoprazole",
    "fexofenadine",
    "dupilumab",
    "tretinoin",
    "magnesium oxide",
    "cholecalciferol",
    "creatine",
}

PRIORITY_ORDER = {"discuss_now": 0, "timing": 1, "monitor": 2, "ignore": 3}


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def heuristic_priority(drug_a: str, drug_b: str, description: str) -> str:
    """Weak label for synthetic rows. WAR gold overrides this when names match."""
    a, b = _norm(drug_a), _norm(drug_b)
    pair = {a, b}
    desc = description.lower()
    this_key = tuple(sorted((a, b)))
    for row in DDI_RANK_GOLD:
        if tuple(sorted((_norm(row["drug_a"]), _norm(row["drug_b"])))) == this_key:
            return row["priority"]

    on_regimen = a in REGIMEN_NAMES and b in REGIMEN_NAMES
    if not on_regimen:
        return "ignore"
    if "absorption" in desc or "fruit juice" in desc or "antacid" in desc:
        return "timing"
    if "excretion" in desc or "serum concentration" in desc:
        return "monitor"
    if "hypotens" in desc or "orthostatic" in desc or "adverse effects can be increased" in desc:
        return "discuss_now"
    return "monitor"


def rows_from_dump(path: Path, query_name: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for bucket in ("partner_hits", "hypotension_orthostasis_sample"):
        for hit in data.get(bucket, []):
            partner = hit.get("name") or ""
            desc = hit.get("description") or ""
            if not partner or not desc:
                continue
            priority = heuristic_priority(query_name, partner, desc)
            rows.append(
                {
                    "drug_a": query_name,
                    "drug_b": partner,
                    "drugbank_id_b": hit.get("id"),
                    "description": desc,
                    "priority": priority,
                    "source_dump": path.name,
                    "split": "train_synthetic" if priority == "ignore" or _norm(partner) not in REGIMEN_NAMES else "holdout_war",
                    "case_id": PATIENT_STATE["case_id"],
                }
            )
    return rows


def live_tooluniverse_minoxidil_baclofen() -> dict[str, Any]:
    from tooluniverse import ToolUniverse

    tu = ToolUniverse()
    tu.load_tools(
        categories=["xml"],
        include_tools=["drugbank_get_drug_interactions_by_drug_name_or_id"],
    )
    return tu.run(
        {
            "name": "drugbank_get_drug_interactions_by_drug_name_or_id",
            "arguments": {
                "query": "minoxidil",
                "exact_match": False,
                "limit": 1,
                "nested_contains": "baclofen",
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-tu",
        action="store_true",
        help="Call ToolUniverse DrugBank XML (may download/parse full_database.xml).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_files = {
        "minoxidil": DUMP_DIR / "drugbank_inte__minoxidil.json",
        "baclofen": DUMP_DIR / "drugbank_inte__baclofen.json",
        "fexofenadine": DUMP_DIR / "drugbank_inte__fexofenadine.json",
        "lansoprazole": DUMP_DIR / "drugbank_inte__lansoprazole.json",
        "finasteride": DUMP_DIR / "drugbank_inte__finasteride.json",
        "dupilumab": DUMP_DIR / "drugbank_inte__dupilumab.json",
    }

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for query_name, path in dump_files.items():
        if not path.exists():
            print(f"skip missing dump {path}")
            continue
        for row in rows_from_dump(path, query_name):
            key = (
                tuple(sorted((_norm(row["drug_a"]), _norm(row["drug_b"])))),
                row["description"],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    # Always keep explicit gold rows, even if a dump was missing.
    for gold in DDI_RANK_GOLD:
        key = (
            tuple(sorted((_norm(gold["drug_a"]), _norm(gold["drug_b"])))),
            gold["description"],
        )
        if key in seen:
            continue
        rows.append({**gold, "source_dump": "gold_seed", "case_id": PATIENT_STATE["case_id"]})
        seen.add(key)

    rows.sort(key=lambda r: (PRIORITY_ORDER.get(r["priority"], 9), r["drug_a"], r["drug_b"]))

    out_path = OUT_DIR / "ddi_rank_expanded.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["priority"]] = counts.get(row["priority"], 0) + 1
    print(f"wrote {out_path} n={len(rows)} by_priority={counts}")

    if args.live_tu:
        print("calling ToolUniverse DrugBank XML for minoxidil nested_contains=baclofen")
        result = live_tooluniverse_minoxidil_baclofen()
        live_path = OUT_DIR / "tooluniverse_live_minoxidil_baclofen.json"
        live_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {live_path}")


if __name__ == "__main__":
    main()
