# WAR regimen / health-summary training gold

De-identified labels from the 2026-08-13 patient regimen review. `patient/` and `results/` stay gitignored. This folder is the committed schema and builders.

Not medical advice.

## Setup

```
python3 -m pip install --user tooluniverse
```

Verified locally as `tooluniverse==1.4.1`.

## Commands

```
python3 scripts/war_training/smoke_tooluniverse.py
python3 scripts/war_training/build_gold.py
python3 scripts/war_training/expand_ddi.py
python3 scripts/war_training/train_ddi_baseline.py
```

`--live-tu` on `expand_ddi.py` loads DrugBank XML via ToolUniverse (`agenticx/DrugBank`). That can download/parse `drugbank_full_database.xml`. Default mode uses the existing dumps under `results/.agent/patient_regimen_review/tool_dumps/`.

Outputs land in `results/.agent/patient_regimen_review/gold/` (gitignored).
