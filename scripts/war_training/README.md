# WAR regimen / health-summary training

De-identified labels from the 2026-08-13 patient regimen review. `patient/` and `results/` stay gitignored.

Not medical advice. WAR rows are `holdout_war`; models train on synthetic ToolUniverse-grounded pairs.

## Setup

```
python3 -m pip install --user tooluniverse
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install --user sentence-transformers scikit-learn
```

Local GPU target: RTX 2070 SUPER 8 GB. Do not fine-tune TxAgent-T1-Llama-3.1-8B here.

## Commands

```
python3 scripts/war_training/smoke_tooluniverse.py
python3 scripts/war_training/build_gold.py
python3 scripts/war_training/expand_ddi.py
python3 scripts/war_training/expand_synthetic.py --with-fda
python3 scripts/war_training/train_ddi_baseline.py
python3 scripts/war_training/train_gpu.py
python3 scripts/war_training/train_toolrag.py
```

`--with-fda` attaches live `FDA_search_drug_labels` snippets. `--live-tu` on `expand_ddi.py` loads DrugBank XML (large).

Outputs: `results/.agent/patient_regimen_review/gold/` and `.../models/` (gitignored).

## Eval

1. Mismatch classifier flags celiac overstatement on WAR holdout.
2. DDI ranker scores minoxidil+baclofen above off-regimen hypotension hits.
3. Eligibility logreg refuses Dupixent on the March EGD feature vector.
4. ToolRAG MiniLM ranks DrugBank/FDA tools above `DrugInteractionAnalyzerAgent`.
