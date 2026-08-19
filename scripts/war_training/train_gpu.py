#!/usr/bin/env python3
"""Train MiniLM DDI ranker + mismatch classifier on GPU (RTX 2070 SUPER 8 GB).

Hold out WAR. Uses transformers AutoModel (not a 1-label MS MARCO head).
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gold_seed import DDI_PRIORITIES, MISMATCH_LABELS  # noqa: E402

GOLD = ROOT / "results" / ".agent" / "patient_regimen_review" / "gold"
MODEL_DIR = ROOT / "results" / ".agent" / "patient_regimen_review" / "models"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_json(name: str):
    path = GOLD / name
    if not path.exists():
        raise SystemExit(f"missing {path}; run build_gold.py and expand_synthetic.py")
    return json.loads(path.read_text(encoding="utf-8"))


def fit_classifier(
    *,
    name: str,
    rows: list[dict],
    text_a_key: str,
    text_b_key: str,
    label_key: str,
    labels: list[str],
    epochs: int = 3,
    batch_size: int = 16,
):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available; training on CPU", file=sys.stderr)

    label2id = {lab: i for i, lab in enumerate(labels)}
    train_rows = [r for r in rows if r.get("split") != "holdout_war" and r.get(label_key) in label2id]
    hold_rows = [r for r in rows if r.get("split") == "holdout_war" and r.get(label_key) in label2id]
    by_lab: dict[str, list] = {}
    for r in train_rows:
        by_lab.setdefault(r[label_key], []).append(r)
    max_per = max(400, max((len(v) for k, v in by_lab.items() if k != "ignore"), default=50) * 3)
    rng = random.Random(0)
    balanced: list = []
    upsample = "ignore" in by_lab
    for lab, group in by_lab.items():
        if upsample and lab == "ignore":
            group = rng.sample(group, min(len(group), 400))
        elif upsample and len(group) < 160:
            group = group + rng.choices(group, k=160 - len(group))
        elif upsample and len(group) > 400:
            group = rng.sample(group, 400)
        balanced.extend(group)
    train_rows = balanced
    rng.shuffle(train_rows)
    print(f"{name}: train={len(train_rows)} holdout={len(hold_rows)} labels={Counter(r[label_key] for r in train_rows)}")
    if len(train_rows) < 20:
        raise SystemExit(f"{name}: not enough train rows")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

    class PairDS(Dataset):
        def __init__(self, subset: list[dict]):
            self.subset = subset

        def __len__(self) -> int:
            return len(self.subset)

        def __getitem__(self, idx: int):
            row = self.subset[idx]
            enc = tokenizer(
                str(row[text_a_key]),
                str(row[text_b_key]),
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(label2id[row[label_key]], dtype=torch.long)
            return item

    class PairClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(ENCODER_NAME)
            hidden = self.encoder.config.hidden_size
            self.classifier = nn.Linear(hidden, len(labels))

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
                kwargs["token_type_ids"] = token_type_ids
            out = self.encoder(**kwargs)
            cls = out.last_hidden_state[:, 0]
            return self.classifier(cls)

    model = PairClassifier().to(device)
    loader = DataLoader(PairDS(train_rows), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    counts = [max(Counter(r[label_key] for r in train_rows).get(lab, 1), 1) for lab in labels]
    weights = torch.tensor([max(counts) / c for c in counts], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"{name} class_weights={dict(zip(labels, [float(w) for w in weights]))}")
    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in loader:
            labels_t = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            loss = loss_fn(logits, labels_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"{name} epoch {epoch+1}/{epochs} loss={total / max(n, 1):.4f}")

    out = MODEL_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "encoder": ENCODER_NAME}, out / "model.pt")
    tokenizer.save_pretrained(str(out))
    (out / "labels.json").write_text(json.dumps(labels) + "\n", encoding="utf-8")

    def predict_rows(subset: list[dict]) -> tuple[list[str], np.ndarray]:
        model.eval()
        logits_all = []
        with torch.no_grad():
            for row in subset:
                enc = tokenizer(
                    str(row[text_a_key]),
                    str(row[text_b_key]),
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits_all.append(model(**enc).cpu().numpy()[0])
        arr = np.stack(logits_all) if logits_all else np.zeros((0, len(labels)))
        ids = arr.argmax(axis=1).tolist() if len(arr) else []
        return [labels[i] for i in ids], arr

    hold_pred, hold_logits = predict_rows(hold_rows) if hold_rows else ([], None)
    n_ok = sum(p == r[label_key] for p, r in zip(hold_pred, hold_rows))
    acc = (n_ok / len(hold_rows)) if hold_rows else float("nan")
    print(f"{name} WAR holdout accuracy: {n_ok}/{len(hold_rows)} = {acc:.3f}")
    for gold, pred, row in zip([r[label_key] for r in hold_rows], hold_pred, hold_rows):
        mark = "ok" if gold == pred else "MISS"
        extra = row.get("drug_a") or row.get("field") or ""
        extra2 = row.get("drug_b") or ""
        print(f"  [{mark}] gold={gold:18s} pred={pred:18s} {extra} {extra2}".rstrip())
    metrics = {
        "train_n": len(train_rows),
        "holdout_n": len(hold_rows),
        "holdout_correct": n_ok,
        "holdout_accuracy": acc,
        "encoder": ENCODER_NAME,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return model, metrics, hold_rows, hold_logits, predict_rows


def train_eligibility() -> dict:
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    rows = load_json("eligibility_synthetic.json")
    keys = [
        "weight_ge_40kg",
        "high_dose_ppi_weeks_ge_8",
        "active_symptoms_impaction_or_frequent_dysphagia",
        "peak_eos_ge_15_both_sites",
        "erefS_shows_more_than_mild_disease",
    ]

    def vec(row: dict) -> list[float]:
        feats = row.get("features") or row.get("criteria") or {}
        return [1.0 if feats.get(k) else 0.0 for k in keys]

    train = [r for r in rows if r.get("split") != "holdout_war"]
    hold = [r for r in rows if r.get("split") == "holdout_war"]
    x = np.array([vec(r) for r in train], dtype=float)
    y = np.array([0 if r["label"] == "not_eligible_on_this_chart" else 1 for r in train])
    clf = LogisticRegression(max_iter=200)
    clf.fit(x, y)
    hold_pred = []
    for r in hold:
        p = int(clf.predict(np.array([vec(r)]))[0])
        lab = "eligible" if p == 1 else "not_eligible_on_this_chart"
        hold_pred.append(lab)
        print(f"eligibility WAR gold={r['label']} pred={lab}")
    ok = hold_pred[0] == hold[0]["label"] if hold else False
    out = MODEL_DIR / "eligibility_logreg"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({"holdout_ok": ok, "train_n": len(train)}, indent=2) + "\n",
        encoding="utf-8",
    )
    if not ok:
        raise SystemExit("eligibility classifier failed WAR Dupixent refusal")
    return {"holdout_ok": ok}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ddi_rows = load_json("ddi_rank_synthetic.json")
    for r in ddi_rows:
        r["ddi_text"] = f"{r['drug_a']} + {r['drug_b']}: {r['description']}"
    mismatch_rows = load_json("mismatch_synthetic.json")

    ddi_model, ddi_metrics, war, war_logits, _pred = fit_classifier(
        name="ddi_minilm",
        rows=ddi_rows,
        text_a_key="patient_text",
        text_b_key="ddi_text",
        label_key="priority",
        labels=DDI_PRIORITIES,
        epochs=4,
        batch_size=16,
    )
    discuss_id = DDI_PRIORITIES.index("discuss_now")
    ranked = sorted(range(len(war)), key=lambda i: float(war_logits[i][discuss_id]), reverse=True)
    print("WAR DDI discuss_now scores:")
    for i in ranked:
        r = war[i]
        print(f"  {war_logits[i][discuss_id]:6.3f} gold={r['priority']:11s} {r['drug_a']} + {r['drug_b']}")
    top = war[ranked[0]]
    rank_ok = {top["drug_a"].lower(), top["drug_b"].lower()} == {"minoxidil", "baclofen"}
    if not rank_ok:
        print("WARNING: top discuss_now score was not minoxidil+baclofen", file=sys.stderr)
    else:
        print("DDI rank ok: minoxidil+baclofen has the highest discuss_now score on WAR holdout")
    ddi_metrics["baclofen_rank_ok"] = rank_ok

    fit_classifier(
        name="mismatch_minilm",
        rows=mismatch_rows,
        text_a_key="self_report",
        text_b_key="chart",
        label_key="label",
        labels=MISMATCH_LABELS,
        epochs=4,
        batch_size=8,
    )
    elig = train_eligibility()
    summary = {"ddi": ddi_metrics, "eligibility": elig}
    (MODEL_DIR / "train_gpu_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote models under {MODEL_DIR}")


if __name__ == "__main__":
    main()
