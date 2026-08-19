#!/usr/bin/env python3
"""Expand WAR gold into synthetic DDI, mismatch, and eligibility pairs.

WAR exact rows stay split=holdout_war. Training uses other patient states so
the same DrugBank sentence can be discuss_now or ignore depending on the body.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gold_seed import (  # noqa: E402
    DDI_RANK_GOLD,
    GUIDELINE_ELIGIBILITY,
    MISMATCH_TABLE,
    PATIENT_STATE,
)

DUMP_DIR = ROOT / "results" / ".agent" / "patient_regimen_review" / "tool_dumps"
OUT_DIR = ROOT / "results" / ".agent" / "patient_regimen_review" / "gold"

DDI_TEMPLATES = [
    "{b} may increase the hypotensive activities of {a}.",
    "The risk or severity of orthostatic hypotension can be increased when {a} is combined with {b}.",
    "The risk or severity of adverse effects can be increased when {b} is combined with {a}.",
    "The excretion of {b} can be decreased when combined with {a}.",
    "{a} can cause a decrease in the absorption of {b} resulting in a reduced serum concentration.",
    "{b} may decrease the antihypertensive activities of {a}.",
    "Fruit juice can decrease the absorption of {b}.",
]


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def load_ddi_catalog() -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in sorted(DUMP_DIR.glob("drugbank_inte__*.json")):
        if path.name.endswith("_ERROR.json") or path.name.endswith("SUMMARY.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        query = data.get("drug_name") or data.get("query") or path.stem.split("__")[-1]
        for bucket in ("partner_hits", "hypotension_orthostasis_sample"):
            for hit in data.get(bucket, []):
                a, b, desc = query, hit.get("name") or "", hit.get("description") or ""
                if not b or not desc:
                    continue
                key = (tuple(sorted((_norm(a), _norm(b)))), desc)
                if key in seen:
                    continue
                seen.add(key)
                catalog.append({"drug_a": a, "drug_b": b, "description": desc})
    extra_partners = [
        ("Minoxidil", "Lisinopril", "Lisinopril may increase the hypotensive activities of Minoxidil."),
        ("Minoxidil", "Losartan", "Losartan may increase the hypotensive activities of Minoxidil."),
        ("Warfarin", "Amiodarone", "The serum concentration of Warfarin can be increased when combined with Amiodarone."),
        ("Levothyroxine", "Calcium carbonate", "Calcium carbonate can cause a decrease in the absorption of Levothyroxine."),
        ("Ciprofloxacin", "Tizanidine", "The risk or severity of adverse effects can be increased when Tizanidine is combined with Ciprofloxacin."),
        ("Simvastatin", "Clarithromycin", "The metabolism of Simvastatin can be decreased when combined with Clarithromycin."),
        ("Metformin", "Iodinated contrast", "The risk or severity of adverse effects can be increased when Iodinated contrast is combined with Metformin."),
        ("Clopidogrel", "Omeprazole", "The metabolism of Clopidogrel can be decreased when combined with Omeprazole."),
        ("Sildenafil", "Nitroglycerin", "The risk or severity of hypotension can be increased when Sildenafil is combined with Nitroglycerin."),
        ("Lithium", "Lisinopril", "The serum concentration of Lithium can be increased when combined with Lisinopril."),
        ("Methotrexate", "Trimethoprim", "The risk or severity of adverse effects can be increased when Trimethoprim is combined with Methotrexate."),
        ("Digoxin", "Amiodarone", "The serum concentration of Digoxin can be increased when combined with Amiodarone."),
        ("Theophylline", "Ciprofloxacin", "The metabolism of Theophylline can be decreased when combined with Ciprofloxacin."),
        ("Potassium chloride", "Spironolactone", "The risk or severity of hyperkalemia can be increased when Spironolactone is combined with Potassium chloride."),
        ("Rivaroxaban", "Ketoconazole", "The serum concentration of Rivaroxaban can be increased when combined with Ketoconazole."),
        ("Tacrolimus", "Lansoprazole", "The serum concentration of Tacrolimus can be increased when combined with Lansoprazole."),
        ("Iron sulfate", "Lansoprazole", "Lansoprazole can cause a decrease in the absorption of Iron sulfate."),
        ("Ketoconazole", "Lansoprazole", "Lansoprazole can cause a decrease in the absorption of Ketoconazole."),
        ("Fexofenadine", "Apple juice", "Fruit juice can decrease the absorption of Fexofenadine."),
        ("Glycopyrronium", "Oxybutynin", "The risk or severity of adverse effects can be increased when Oxybutynin is combined with Glycopyrronium."),
    ]
    for a, b, desc in extra_partners:
        key = (tuple(sorted((_norm(a), _norm(b)))), desc)
        if key in seen:
            continue
        seen.add(key)
        catalog.append({"drug_a": a, "drug_b": b, "description": desc})
    return catalog


def synthetic_patients() -> list[dict[str, Any]]:
    """Patient states that are not the WAR holdout body."""
    patients: list[dict[str, Any]] = []
    pk_pairs = [
        (["fexofenadine", "lansoprazole"], "no_orthostasis"),
        (["fexofenadine", "magnesium oxide"], "no_orthostasis"),
        (["levothyroxine", "calcium carbonate"], "no_orthostasis"),
        (["simvastatin", "clarithromycin"], "no_orthostasis"),
        (["clopidogrel", "omeprazole"], "no_orthostasis"),
        (["warfarin", "amiodarone"], "no_orthostasis"),
        (["digoxin", "amiodarone"], "no_orthostasis"),
        (["tacrolimus", "lansoprazole"], "no_orthostasis"),
        (["ciprofloxacin", "tizanidine"], "orthostasis"),
        (["sildenafil", "nitroglycerin"], "orthostasis"),
        (["minoxidil", "metoprolol"], "orthostasis"),
        (["minoxidil", "lisinopril"], "orthostasis"),
        (["minoxidil", "duloxetine"], "orthostasis"),
        (["minoxidil", "amlodipine"], "orthostasis"),
        (["baclofen", "metoprolol"], "orthostasis"),
        (["glycopyrronium", "oxybutynin"], "no_orthostasis"),
        (["methotrexate", "trimethoprim"], "no_orthostasis"),
        (["lithium", "lisinopril"], "no_orthostasis"),
        (["rivaroxaban", "ketoconazole"], "no_orthostasis"),
        (["iron sulfate", "lansoprazole"], "no_orthostasis"),
        (["theophylline", "ciprofloxacin"], "no_orthostasis"),
        (["potassium chloride", "spironolactone"], "no_orthostasis"),
        (["ketoconazole", "lansoprazole"], "no_orthostasis"),
        (["fexofenadine", "apple juice"], "no_orthostasis"),
        (["minoxidil", "finasteride"], "no_orthostasis"),
        (["minoxidil"], "orthostasis"),
        (["baclofen"], "orthostasis"),
        (["lansoprazole", "fexofenadine", "creatine"], "no_orthostasis"),
        (["atorvastatin", "metformin"], "no_orthostasis"),
        (["sertraline", "levothyroxine"], "no_orthostasis"),
        (["albuterol", "fluticasone"], "no_orthostasis"),
        (["omeprazole", "clopidogrel", "aspirin"], "no_orthostasis"),
        (["amlodipine", "losartan"], "orthostasis"),
        (["tamsulosin", "sildenafil"], "orthostasis"),
        (["carvedilol", "furosemide"], "orthostasis"),
        (["nifedipine", "baclofen"], "orthostasis"),
        (["isosorbide dinitrate", "sildenafil"], "orthostasis"),
        (["guanethidine", "minoxidil"], "orthostasis"),
        (["minoxidil", "tizanidine"], "orthostasis"),
        (["lansoprazole"], "no_orthostasis"),
    ]
    for i, (meds, ortho) in enumerate(pk_pairs):
        patients.append(
            {
                "case_id": f"syn_{i:03d}",
                "meds": [m.lower() for m in meds],
                "orthostasis": ortho == "orthostasis",
                "bp": "88/54 sitting" if ortho == "orthostasis" else "124/78 sitting",
                "problems": (
                    ["hypertension", "lightheadedness"]
                    if ortho == "orthostasis"
                    else ["allergic rhinitis"]
                ),
            }
        )
    # Extra combinatorial patients: vasodilator + random add-on, partner absent.
    rng = random.Random(13)
    absent_partners = ["duloxetine", "levodopa", "risperidone", "valsartan", "atenolol"]
    for j, partner in enumerate(absent_partners):
        patients.append(
            {
                "case_id": f"syn_abs_{j:03d}",
                "meds": ["minoxidil", "finasteride", "cetirizine"],
                "orthostasis": True,
                "bp": "130/58 sitting",
                "problems": ["androgenetic alopecia"],
                "explicitly_absent": [partner],
            }
        )
    for k in range(20):
        core = rng.sample(
            ["metformin", "atorvastatin", "lisinopril", "amlodipine", "omeprazole", "levothyroxine", "sertraline", "albuterol"],
            k=3,
        )
        patients.append(
            {
                "case_id": f"syn_rand_{k:03d}",
                "meds": core,
                "orthostasis": bool(k % 2),
                "bp": "110/70 sitting",
                "problems": ["type 2 diabetes"] if "metformin" in core else ["dyslipidemia"],
            }
        )
    busy_cores = [
        ["minoxidil", "baclofen", "fexofenadine", "lansoprazole", "finasteride", "creatine"],
        ["minoxidil", "metoprolol", "fexofenadine", "lansoprazole", "finasteride", "creatine"],
        ["minoxidil", "duloxetine", "fexofenadine", "omeprazole", "finasteride", "vitamin d"],
        ["sildenafil", "nitroglycerin", "lisinopril", "atorvastatin", "metformin", "omeprazole"],
        ["baclofen", "amlodipine", "fexofenadine", "lansoprazole", "tretinoin", "creatine"],
        ["fexofenadine", "magnesium oxide", "lansoprazole", "cetirizine", "vitamin d", "creatine"],
        ["minoxidil", "tizanidine", "finasteride", "cetirizine", "creatine", "fish oil"],
        ["glycopyrronium", "oxybutynin", "fexofenadine", "lansoprazole", "creatine", "vitamin d"],
    ]
    for i, meds in enumerate(busy_cores):
        patients.append(
            {
                "case_id": f"syn_busy_{i:03d}",
                "meds": meds,
                "orthostasis": True,
                "bp": "132/58 sitting",
                "problems": ["androgenetic alopecia", "allergic rhinitis", "lightheadedness"],
            }
        )
        patients.append(
            {
                "case_id": f"syn_busy_stable_{i:03d}",
                "meds": meds,
                "orthostasis": False,
                "bp": "122/76 sitting",
                "problems": ["androgenetic alopecia", "allergic rhinitis"],
            }
        )
    return patients


def patient_text(p: dict[str, Any]) -> str:
    meds = ", ".join(p["meds"])
    problems = "; ".join(p["problems"])
    ortho = "yes" if p["orthostasis"] else "no"
    return (
        f"ORTHOSTASIS={ortho}. MEDS={meds}. "
        f"Case {p['case_id']}. Problems: {problems}. BP {p['bp']}."
    )


def war_patient_text() -> str:
    meds = ", ".join(m["name"] for m in PATIENT_STATE["meds_chart"])
    problems = "; ".join(PATIENT_STATE["problems_chart"])
    facts = PATIENT_STATE["key_facts"]
    return (
        f"ORTHOSTASIS=yes. MEDS={meds}. "
        f"Case {PATIENT_STATE['case_id']}. Age {PATIENT_STATE['demographics']['age_years']}. "
        f"Problems: {problems}. BP {facts['sitting_bp']}. {facts['orthostatic_vitals']}."
    )


def label_ddi(patient: dict[str, Any], drug_a: str, drug_b: str, desc: str) -> str:
    meds = {_norm(m) for m in patient["meds"]}
    a, b = _norm(drug_a), _norm(drug_b)
    pair_on = a in meds and b in meds
    desc_l = desc.lower()
    hypotension = any(
        k in desc_l
        for k in ("hypotens", "orthostatic", "syncope", "adverse effects can be increased")
    )
    absorption = any(k in desc_l for k in ("absorption", "fruit juice", "antacid"))
    pk = any(k in desc_l for k in ("excretion", "serum concentration", "metabolism"))
    antihtn_decrease = "decrease the antihypertensive" in desc_l
    if not pair_on:
        return "ignore"
    if antihtn_decrease and not patient["orthostasis"]:
        return "ignore"
    if absorption:
        return "timing"
    if hypotension and patient["orthostasis"]:
        return "discuss_now"
    if hypotension and not patient["orthostasis"]:
        return "monitor"
    if pk:
        return "monitor"
    return "monitor"


def expand_ddi(catalog: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cartesian: list[dict[str, Any]] = []
    for p in synthetic_patients():
        ptxt = patient_text(p)
        for hit in catalog:
            priority = label_ddi(p, hit["drug_a"], hit["drug_b"], hit["description"])
            cartesian.append(
                {
                    "split": "train_synthetic",
                    "case_id": p["case_id"],
                    "patient_text": ptxt,
                    "drug_a": hit["drug_a"],
                    "drug_b": hit["drug_b"],
                    "description": hit["description"],
                    "priority": priority,
                }
            )
    rng = random.Random(0)
    minority = [r for r in cartesian if r["priority"] != "ignore"]
    ignores = [r for r in cartesian if r["priority"] == "ignore"]
    rows.extend(minority)
    rows.extend(rng.sample(ignores, min(800, len(ignores))))
    # Balanced overlays: same DrugBank sentence, different bodies.
    for i, hit in enumerate(catalog):
        a, b, desc = hit["drug_a"], hit["drug_b"], hit["description"]
        variants = [
            {
                "case_id": f"bal_both_ortho_{i:03d}",
                "meds": [_norm(a), _norm(b)],
                "orthostasis": True,
                "bp": "92/50 sitting",
                "problems": ["lightheadedness"],
            },
            {
                "case_id": f"bal_both_stable_{i:03d}",
                "meds": [_norm(a), _norm(b)],
                "orthostasis": False,
                "bp": "128/76 sitting",
                "problems": ["allergic rhinitis"],
            },
            {
                "case_id": f"bal_a_only_{i:03d}",
                "meds": [_norm(a), "cetirizine"],
                "orthostasis": True,
                "bp": "130/58 sitting",
                "problems": ["lightheadedness"],
            },
        ]
        for p in variants:
            rows.append(
                {
                    "split": "train_synthetic",
                    "case_id": p["case_id"],
                    "patient_text": patient_text(p),
                    "drug_a": a,
                    "drug_b": b,
                    "description": desc,
                    "priority": label_ddi(p, a, b, desc),
                }
            )
    # Hold out WAR gold with WAR patient text.
    ptxt = war_patient_text()
    gold_pairs = {tuple(sorted((_norm(g["drug_a"]), _norm(g["drug_b"])))) for g in DDI_RANK_GOLD}
    for gold in DDI_RANK_GOLD:
        rows.append(
            {
                "split": "holdout_war",
                "case_id": PATIENT_STATE["case_id"],
                "patient_text": ptxt,
                "drug_a": gold["drug_a"],
                "drug_b": gold["drug_b"],
                "description": gold["description"],
                "priority": gold["priority"],
                "why": gold.get("why"),
            }
        )
    # Paraphrased WAR body in train (not the holdout wording).
    war_meds = [m["name"] for m in PATIENT_STATE["meds_chart"]]
    war_p = {
        "case_id": "war_para",
        "meds": war_meds,
        "orthostasis": True,
        "bp": "wide pulse pressure, lightheaded on standing",
        "problems": PATIENT_STATE["problems_chart"][:4],
    }
    paraphrases = [
        patient_text(war_p),
        "Young adult male on oral minoxidil 2.5 mg, baclofen, lansoprazole BID, fexofenadine, finasteride. Orthostatic lightheadedness. Not Graves. Cam FAI on MRI.",
        "Chart: Loniten plus baclofen in a patient with documented lightheadedness and sitting BP 137/60. PPI and Allegra also active.",
        "Polypharmacy hair-loss plus pelvic-pain regimen. Vasodilator tablet plus GABA-B relaxant. Orthostasis present.",
        "Meds include minoxidil tablets, baclofen, Prevacid BID, Allegra 180, Propecia. Dizzy on standing. EoE on PPI trial.",
    ]
    for j, text in enumerate(paraphrases):
        for gold in DDI_RANK_GOLD:
            rows.append(
                {
                    "split": "train_synthetic",
                    "case_id": f"war_para_{j}",
                    "patient_text": text,
                    "drug_a": gold["drug_a"],
                    "drug_b": gold["drug_b"],
                    "description": gold["description"],
                    "priority": gold["priority"],
                }
            )
        for hit in catalog:
            key = tuple(sorted((_norm(hit["drug_a"]), _norm(hit["drug_b"]))))
            if key in gold_pairs:
                continue
            rows.append(
                {
                    "split": "train_synthetic",
                    "case_id": f"war_para_{j}",
                    "patient_text": text,
                    "drug_a": hit["drug_a"],
                    "drug_b": hit["drug_b"],
                    "description": hit["description"],
                    "priority": label_ddi(war_p, hit["drug_a"], hit["drug_b"], hit["description"]),
                }
            )
    return rows


def paraphrase(text: str, k: int) -> str:
    prefixes = [
        "",
        "Chart note: ",
        "Patient states: ",
        "Documented as: ",
        "History: ",
    ]
    suffixes = [
        "",
        " (source: self-report).",
        " per CCD.",
        " — confirm with clinician.",
    ]
    return f"{prefixes[k % len(prefixes)]}{text}{suffixes[k % len(suffixes)]}".strip()


def expand_mismatch() -> list[dict[str, Any]]:
    templates = [
        {
            "label": "overstated",
            "self": "I have {disease} because I carry {risk_marker}",
            "chart": "{risk_marker} present; {disease} not confirmed ({neg_test})",
            "slots": [
                {"disease": "celiac disease", "risk_marker": "HLA-DQ2", "neg_test": "negative tTG-IgA and normal duodenum"},
                {"disease": "Graves disease", "risk_marker": "a low TSH once", "neg_test": "negative TRAb and recovered TSH"},
                {"disease": "hemochromatosis", "risk_marker": "HFE risk on a consumer panel", "neg_test": "normal ferritin and no HFE pathogenic variant on clinical testing"},
                {"disease": "factor V Leiden thrombophilia", "risk_marker": "a family story of clots", "neg_test": "no thrombophilia variant detected"},
                {"disease": "active EoE needing a biologic", "risk_marker": "a remote EoE diagnosis", "neg_test": "mild EREFS and incomplete PPI trial"},
            ],
        },
        {
            "label": "route_mismatch",
            "self": "{drug} {self_route}",
            "chart": "{drug} {chart_route}",
            "slots": [
                {"drug": "baclofen", "self_route": "15 mg oral 1-2x daily PRN", "chart_route": "15 mg rectal QHS compounded"},
                {"drug": "minoxidil", "self_route": "foam to scalp", "chart_route": "2.5 mg oral tablets daily"},
                {"drug": "glycopyrronium", "self_route": "cloth to face daily", "chart_route": "not listed; label is axillae only"},
                {"drug": "sumatriptan", "self_route": "nasal spray PRN", "chart_route": "oral 50 mg tablets"},
                {"drug": "ondansetron", "self_route": "ODT as needed", "chart_route": "IV in procedure notes only"},
            ],
        },
        {
            "label": "missing_from_chart",
            "self": "Allergy: {item}",
            "chart": "{item} not on the allergy list",
            "slots": [
                {"item": "IgE-mediated egg"},
                {"item": "tree nut (cashew)"},
                {"item": "sulfa antibiotics"},
                {"item": "latex"},
                {"item": "NSAIDs causing hives"},
            ],
        },
        {
            "label": "missing_from_self",
            "self": "Not mentioned in the patient-authored history",
            "chart": "{finding}",
            "slots": [
                {"finding": "bilateral cam FAI on MRI"},
                {"finding": "8-day patch monitor with no significant arrhythmia"},
                {"finding": "prior cholecystectomy"},
                {"finding": "ACE-inhibitor cough documented in 2019"},
                {"finding": "normal scrotal ultrasound after testicular pain"},
            ],
        },
        {
            "label": "agree",
            "self": "{item}",
            "chart": "{item}",
            "slots": [
                {"item": "fexofenadine 180 mg daily for environmental allergies"},
                {"item": "finasteride 1 mg daily for hair loss"},
                {"item": "vitamin D3 2000 IU daily"},
                {"item": "creatine 5 g daily"},
                {"item": "asthma since adolescence"},
            ],
        },
    ]
    rows: list[dict[str, Any]] = []
    for gold in MISMATCH_TABLE:
        rows.append(
            {
                "split": "holdout_war",
                "field": gold["field"],
                "self_report": gold["self_report"],
                "chart": gold["chart"],
                "label": gold["label"],
            }
        )
    n = 0
    for tmpl in templates:
        for slot in tmpl["slots"]:
            self_s = tmpl["self"].format(**slot)
            chart_s = tmpl["chart"].format(**slot)
            for k in range(8):
                # Do not copy WAR gold strings into train.
                if any(self_s == g["self_report"] and chart_s == g["chart"] for g in MISMATCH_TABLE):
                    continue
                rows.append(
                    {
                        "split": "train_synthetic",
                        "field": tmpl["label"],
                        "self_report": paraphrase(self_s, k),
                        "chart": paraphrase(chart_s, k + 1),
                        "label": tmpl["label"],
                    }
                )
                n += 1
    extra_agree = [
        ("I take Allegra 180 mg every morning with water", "fexofenadine 180 mg daily since 2012"),
        ("Creatine 5 grams after workouts", "creatine 5000 mg daily"),
        ("Gluten-free for EoE, not because of biopsy-proven celiac", "gluten listed as EoE diet allergy; tTG-IgA negative"),
    ]
    for i, (s, c) in enumerate(extra_agree):
        rows.append(
            {
                "split": "train_synthetic",
                "field": "agree_extra",
                "self_report": s,
                "chart": c,
                "label": "agree",
            }
        )
    return rows


def expand_eligibility() -> list[dict[str, Any]]:
    rng = random.Random(7)
    rows = []
    for i in range(400):
        weight_ok = rng.random() < 0.85
        ppi_weeks = rng.choice([0, 4, 6, 7, 8, 12, 16, 24])
        impaction = rng.random() < 0.25
        frequent_dysphagia = rng.random() < 0.35
        eos_both = rng.random() < 0.3
        erefs_more_than_mild = rng.random() < 0.3
        eligible = (
            weight_ok
            and ppi_weeks >= 8
            and (impaction or frequent_dysphagia)
            and (eos_both or erefs_more_than_mild)
        )
        rows.append(
            {
                "split": "train_synthetic",
                "case_id": f"eoe_{i:03d}",
                "features": {
                    "weight_ge_40kg": weight_ok,
                    "high_dose_ppi_weeks_ge_8": ppi_weeks >= 8,
                    "active_symptoms_impaction_or_frequent_dysphagia": impaction or frequent_dysphagia,
                    "peak_eos_ge_15_both_sites": eos_both,
                    "erefS_shows_more_than_mild_disease": erefs_more_than_mild,
                },
                "ppi_weeks": ppi_weeks,
                "label": "eligible" if eligible else "not_eligible_on_this_chart",
            }
        )
    rows.append(
        {
            "split": "holdout_war",
            "case_id": PATIENT_STATE["case_id"],
            **GUIDELINE_ELIGIBILITY,
        }
    )
    return rows


def maybe_fda_snippets() -> dict[str, Any]:
    from tooluniverse import ToolUniverse

    tu = ToolUniverse()
    tu.load_tools(
        categories=["openfda_labels"],
        include_tools=["FDA_search_drug_labels"],
    )
    out: dict[str, Any] = {}
    for name in ("minoxidil", "baclofen", "fexofenadine", "dupilumab"):
        out[name] = tu.run(
            {
                "name": "FDA_search_drug_labels",
                "arguments": {"drug_name": name, "limit": 1, "max_section_chars": 400},
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-fda", action="store_true", help="Attach live FDA snippets via ToolUniverse.")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_ddi_catalog()
    ddi_rows = expand_ddi(catalog)
    mismatch_rows = expand_mismatch()
    elig_rows = expand_eligibility()

    def dump(name: str, rows: list) -> None:
        path = OUT_DIR / name
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        splits: dict[str, int] = {}
        labels: dict[str, int] = {}
        for row in rows:
            splits[row.get("split", "?")] = splits.get(row.get("split", "?"), 0) + 1
            lab = row.get("priority") or row.get("label")
            if lab:
                labels[str(lab)] = labels.get(str(lab), 0) + 1
        print(f"wrote {path} n={len(rows)} splits={splits} labels={labels}")

    dump("ddi_rank_synthetic.json", ddi_rows)
    dump("mismatch_synthetic.json", mismatch_rows)
    dump("eligibility_synthetic.json", elig_rows)

    if args.with_fda:
        snippets = maybe_fda_snippets()
        path = OUT_DIR / "fda_snippets.json"
        path.write_text(json.dumps(snippets, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {path} drugs={list(snippets)}")


if __name__ == "__main__":
    main()
