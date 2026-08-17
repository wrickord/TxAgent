#!/usr/bin/env python3
"""Tiny bag-of-words DDI ranker. Hold out minoxidil+baclofen.

Uses numpy only (comes with tooluniverse). Not a production model — a
reproducible baseline that the named hypotensive pair should outrank
class-PD hypotension hits whose partner is not on the regimen.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "results" / ".agent" / "patient_regimen_review" / "gold"

PRIORITY_SCORE = {"discuss_now": 3.0, "timing": 2.0, "monitor": 1.0, "ignore": 0.0}
TOKEN_RE = re.compile(r"[a-z0-9]+")
REGIMEN = {
    "minoxidil",
    "baclofen",
    "finasteride",
    "lansoprazole",
    "fexofenadine",
    "dupilumab",
    "tretinoin",
    "magnesium oxide",
}


def partner_on_list(row: dict) -> float:
    names = {row["drug_a"].lower(), row["drug_b"].lower()}
    return 1.0 if len(names & REGIMEN) == 2 else 0.0


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_rows() -> list[dict]:
    path = GOLD / "ddi_rank_expanded.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run build_gold.py and expand_ddi.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def patient_text() -> str:
    state = json.loads((GOLD / "patient_state.json").read_text(encoding="utf-8"))
    facts = state.get("key_facts", {})
    meds = " ".join(m["name"] for m in state.get("meds_chart", []))
    return " ".join(
        [
            meds,
            str(facts.get("sitting_bp", "")),
            str(facts.get("orthostatic_vitals", "")),
            "orthostasis vasodilator",
        ]
    )


def vectorize(docs: list[str]) -> tuple[np.ndarray, list[str]]:
    tokenized = [tokenize(d) for d in docs]
    counts: Counter[str] = Counter()
    for toks in tokenized:
        counts.update(set(toks))
    vocab = [w for w, _n in counts.most_common(400)]
    index = {w: i for i, w in enumerate(vocab)}
    x = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        for w in toks:
            j = index.get(w)
            if j is not None:
                x[i, j] += 1.0
        n = x[i].sum()
        if n:
            x[i] /= n
    return x, vocab


def main() -> None:
    rows = load_rows()
    ptxt = patient_text()
    docs = [f"{ptxt} {r['drug_a']} {r['drug_b']} {r['description']}" for r in rows]
    y = np.array([PRIORITY_SCORE[r["priority"]] for r in rows], dtype=np.float32)

    x, vocab = vectorize(docs)
    on_list = np.array([[partner_on_list(r)] for r in rows], dtype=np.float32)
    x = np.concatenate([x, on_list], axis=1)

    hold_pair = {"minoxidil", "baclofen"}
    hold_idx = {
        i
        for i, r in enumerate(rows)
        if {r["drug_a"].lower(), r["drug_b"].lower()} == hold_pair
    }
    train_idx = [i for i in range(len(rows)) if i not in hold_idx]
    x_train = x[train_idx]
    y_train = y[train_idx]
    xtx = x_train.T @ x_train + 1e-2 * np.eye(x_train.shape[1])
    w = np.linalg.solve(xtx, x_train.T @ y_train)
    pred = x @ w

    ranked = sorted(range(len(rows)), key=lambda i: float(pred[i]), reverse=True)
    print(f"n={len(rows)} train={len(train_idx)} holdout={len(hold_idx)} vocab={len(vocab)}")
    print("top 8 predicted:")
    for rank, i in enumerate(ranked[:8], start=1):
        r = rows[i]
        print(
            f"  {rank:2d} pred={pred[i]:6.3f} gold={r['priority']:11s} "
            f"{r['drug_a']} + {r['drug_b']}"
        )

    baclofen_i = next(iter(hold_idx))
    baclofen_rank = ranked.index(baclofen_i) + 1
    print(f"minoxidil+baclofen rank={baclofen_rank} / {len(rows)} pred={pred[baclofen_i]:.3f}")
    if baclofen_rank > 5:
        raise SystemExit("baseline failed: named hypotensive pair should rank in top 5")
    print("baseline ok")


if __name__ == "__main__":
    main()
