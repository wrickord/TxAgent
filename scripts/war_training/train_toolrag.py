#!/usr/bin/env python3
"""Contrastive MiniLM ToolRAG: prefer pairwise FDA/DrugBank over the analyzer that timed out.

Production TxAgent uses ToolRAG-T1-GTE-Qwen2-1.5B. That encoder is too large to
fine-tune comfortably on 8 GB, so this trains all-MiniLM-L6-v2 and evaluates
whether DrugInteractionAnalyzerAgent falls behind completed tools.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gold_seed import CLINICAL_QUESTIONS, TOOL_TRACE_LABELS  # noqa: E402

GOLD = ROOT / "results" / ".agent" / "patient_regimen_review" / "gold"
MODEL_DIR = ROOT / "results" / ".agent" / "patient_regimen_review" / "models" / "toolrag_minilm"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

POSITIVE_TOOLS = TOOL_TRACE_LABELS["positives"]
HARD_NEGATIVES = TOOL_TRACE_LABELS["negatives_timeout"] + [
    "LiteratureSynthesisAgent",
    "AdverseEventPredictionAgent",
    "FDA_get_indications_and_usage_by_drug_name",
]


def tool_catalog() -> list[dict]:
    """Load name+description for WAR-relevant tools plus distractors."""
    import json as json_lib
    from tooluniverse.default_config import default_tool_files

    wanted = set(POSITIVE_TOOLS) | set(HARD_NEGATIVES) | {
        "drugbank_vocab_search",
        "FDA_get_contraindications_by_drug_name",
        "OpenFDA_search_drug_labels",
        "PubMed_get_article",
        "mesh_get_subjects_by_pharmacological_action",
        "EuropePMC_search_articles",
        "ChEMBL_search_similar_molecules",
    }
    catalog: list[dict] = []
    seen: set[str] = set()
    for _cat, path in default_tool_files.items():
        try:
            tools = json_lib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json_lib.JSONDecodeError):
            continue
        if not isinstance(tools, list):
            continue
        for tool in tools:
            name = tool.get("name")
            if name not in wanted or name in seen:
                continue
            seen.add(name)
            catalog.append(
                {
                    "name": name,
                    "description": tool.get("description") or "",
                    "json": json_lib.dumps(
                        {
                            "name": name,
                            "description": tool.get("description"),
                            "parameter": tool.get("parameter"),
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    if "DrugInteractionAnalyzerAgent" not in seen:
        catalog.append(
            {
                "name": "DrugInteractionAnalyzerAgent",
                "description": "AI agent that analyzes drug-drug interactions and provides clinical recommendations",
                "json": json.dumps(
                    {
                        "name": "DrugInteractionAnalyzerAgent",
                        "description": "AI agent that analyzes drug-drug interactions and provides clinical recommendations",
                    }
                ),
            }
        )
    return catalog


def main() -> None:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("Need sentence-transformers + torch") from exc

    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        device = "cpu"
        print("WARNING: CUDA not available; training on CPU", file=sys.stderr)

    catalog = tool_catalog()
    by_name = {t["name"]: t for t in catalog}
    missing_pos = [n for n in POSITIVE_TOOLS if n not in by_name]
    if missing_pos:
        print(f"WARNING: missing tool specs {missing_pos}; using name-only placeholders")
        for n in missing_pos:
            by_name[n] = {"name": n, "description": n, "json": json.dumps({"name": n})}
            catalog.append(by_name[n])
    for n in HARD_NEGATIVES:
        if n not in by_name:
            by_name[n] = {"name": n, "description": n, "json": json.dumps({"name": n, "description": n})}
            catalog.append(by_name[n])

    queries = list(CLINICAL_QUESTIONS) + [
        "Which tools give pairwise FDA labels and local DrugBank XML interactions?",
        "The DrugInteractionAnalyzerAgent timed out; what should I call instead?",
        "Find labeled contraindications and adverse reactions for oral minoxidil tablets.",
        "Search FAERS for minoxidil plus baclofen combination reports.",
        "Get DailyMed interaction text for glycopyrronium cloths.",
    ]
    triples: list[tuple[str, str, str]] = []
    hard_neg_json = [
        by_name[n]["json"] for n in HARD_NEGATIVES if n in by_name
    ] or [json.dumps({"name": "DrugInteractionAnalyzerAgent"})]
    for q in queries:
        for pos_name in POSITIVE_TOOLS:
            if pos_name not in by_name:
                continue
            for neg in hard_neg_json:
                triples.append((q, by_name[pos_name]["json"], neg))
    print(f"contrastive triples={len(triples)} tools_in_catalog={len(catalog)}")

    class TripleDS(Dataset):
        def __init__(self, rows: list[tuple[str, str, str]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            return self.rows[idx]

    model = SentenceTransformer(ENCODER_NAME, device=device)
    loader = DataLoader(TripleDS(triples), batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    def embed(texts) -> torch.Tensor:
        feats = model.preprocess(list(texts))
        tensor_feats = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                tensor_feats[k] = v.to(device)
        out = model(tensor_feats)
        emb = out["sentence_embedding"] if isinstance(out, dict) else out
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    epochs = 3
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for q, pos, neg in loader:
            q_emb = embed(q)
            p_emb = embed(pos)
            n_emb = embed(neg)
            pos_score = (q_emb * p_emb).sum(dim=1)
            neg_score = (q_emb * n_emb).sum(dim=1)
            loss = torch.nn.functional.softplus(neg_score - pos_score).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"toolrag epoch {epoch+1}/{epochs} loss={total / max(n, 1):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    n_pairs = len(triples)

    eval_query = (
        "Is this regimen safe? Prefer pairwise FDA labels and DrugBank XML. "
        "Do not use the interaction analyzer agent that times out."
    )
    q_emb = model.encode([eval_query], normalize_embeddings=True)
    tool_emb = model.encode([t["json"] for t in catalog], normalize_embeddings=True)
    scores = (q_emb @ tool_emb.T)[0]
    ranked = sorted(range(len(catalog)), key=lambda i: float(scores[i]), reverse=True)
    print("ToolRAG ranks for regimen-safety query:")
    for rank, i in enumerate(ranked[:12], start=1):
        print(f"  {rank:2d} {scores[i]:6.3f} {catalog[i]['name']}")

    analyzer_i = next(i for i, t in enumerate(catalog) if t["name"] == "DrugInteractionAnalyzerAgent")
    analyzer_rank = ranked.index(analyzer_i) + 1
    pos_ranks = [
        ranked.index(next(i for i, t in enumerate(catalog) if t["name"] == n)) + 1
        for n in POSITIVE_TOOLS
        if n in {t["name"] for t in catalog}
    ]
    best_pos = min(pos_ranks) if pos_ranks else 999
    print(f"DrugInteractionAnalyzerAgent rank={analyzer_rank} best_positive_rank={best_pos}")
    metrics = {
        "encoder": ENCODER_NAME,
        "analyzer_rank": analyzer_rank,
        "best_positive_rank": best_pos,
        "top5": [catalog[i]["name"] for i in ranked[:5]],
        "n_pairs": n_pairs,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if analyzer_rank <= best_pos:
        raise SystemExit("ToolRAG failed: analyzer still outranks completed pairwise tools")
    print("toolrag ok")


if __name__ == "__main__":
    main()
