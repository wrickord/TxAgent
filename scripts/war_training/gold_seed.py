"""De-identified WAR case gold for regimen / health-summary models.

No name, address, MRN, member IDs, DOB, ancestry, or family tree.
Chart preferred over self-report when they disagree.
Not medical advice; labels are research targets from the 2026-08-13 review.
"""

from __future__ import annotations

CASE_ID = "WAR"
CHART_DATE = "2026-08-13"

PATIENT_STATE = {
    "case_id": CASE_ID,
    "chart_date": CHART_DATE,
    "demographics": {
        "age_years": 24,
        "sex": "male",
        "height_cm": 180.3,
        "weight_lb": 185,
        "bmi": 25.8,
    },
    "problems_chart": [
        "migraine",
        "thyroid enlargement / diffuse goiter",
        "neck pain",
        "eosinophilic esophagitis due to food (since 2014)",
        "asthma (since 2013)",
        "bilateral cam FAI (R>L)",
        "mild L5-S1 disc narrowing with posterior midline annular fissure",
    ],
    "problems_self_report_only": [
        "hypertonic pelvic floor",
        "pudendal neuralgia",
        "sciatica",
        "toe numbness",
        "fatigue",
        "hyperhidrosis",
        "mild hair loss",
        "acne",
    ],
    "allergies_chart": ["gluten (EoE diet)", "dairy"],
    "allergies_self_report_only": ["IgE egg"],
    "meds_chart": [
        {
            "name": "fexofenadine",
            "brand": "Allegra",
            "dose": "180 mg daily",
            "since": "2012",
            "indication": "allergies",
        },
        {
            "name": "cholecalciferol",
            "dose": "2000 IU daily",
            "since": "2020",
            "indication": "support",
        },
        {
            "name": "creatine",
            "dose": "5 g daily",
            "since": "2020",
            "indication": "support",
        },
        {
            "name": "omega-3 fish oil",
            "dose": "1000 mg capsule (120-180 mg EPA/DHA)",
            "since": "2020",
            "indication": "support",
        },
        {
            "name": "minoxidil",
            "brand": "Loniten",
            "dose": "2.5 mg oral daily",
            "since": "2025-02-01",
            "indication": "hair (off-label vs tablet label)",
            "route": "oral",
        },
        {
            "name": "finasteride",
            "brand": "Propecia",
            "dose": "1 mg daily",
            "since": "2025-02-01",
            "indication": "androgenetic alopecia",
        },
        {
            "name": "sodium fluoride paste",
            "brand": "PreviDent 5000",
            "dose": "1.1% nightly",
            "since": "2025-10-20",
            "indication": "dental",
        },
        {
            "name": "lansoprazole",
            "brand": "Prevacid",
            "dose": "30 mg BID",
            "since": "2026-06-24",
            "indication": "EoE PPI trial",
        },
        {
            "name": "baclofen",
            "dose": "15 mg rectal QHS compounded; oral TID lines also listed",
            "since": "2026-07-23",
            "indication": "pelvic pain (off-label)",
            "route": "rectal_qhs_plus_ambiguous_oral",
        },
    ],
    "meds_self_report_not_on_chart": [
        "glycopyrronium cloth (Qbrexza) face daily",
        "tretinoin 0.05% daily",
        "sodium sulfacetamide 10% daily",
        "men's multivitamin no iron",
        "magnesium glycinate 300 mg daily",
    ],
    "key_facts": {
        "sitting_bp": "137/60 (2026-07-23); also 123/78 (2026-06-01)",
        "orthostatic_vitals": "reported lightheadedness; lying/standing pair not documented",
        "zio": "8-day monitor on Loniten: no significant arrhythmia",
        "thyroid": "TSH 0.270 then 0.523; FT4/T3 normal; TPO/TRAb/TgAb negative; diffuse goiter; tiny TI-RADS 4 no F/U",
        "eoe_egd_2026_03": {
            "symptoms": "feels good; no impactions; very rare mild dysphagia",
            "erefS": "Ed0 R0 Ex1 F1 S0",
            "eos_hpf": "distal 16 / proximal 8, indeterminate",
            "ppi_at_scope": "lansoprazole 30 mg QD",
        },
        "celiac": "HLA-DQ2+DQ7 at-risk heterodimer; tTG-IgA negative; duodenum normal; not a celiac diagnosis",
        "hip_pelvis": "bilateral cam R>L; pubic symphysis normal; scrotal US normal",
        "dupixent": "not eligible on this chart (TREET-style: need active disease after >=8 weeks high-dose PPI)",
    },
}

MISMATCH_TABLE = [
    {
        "field": "height",
        "self_report": "6 ft",
        "chart": "5 ft 11 in (180.3 cm)",
        "label": "overstated",
    },
    {
        "field": "celiac_diagnosis",
        "self_report": "celiac disease with HLA-DQ2",
        "chart": "HLA-DQ2+DQ7 at-risk only; serology and duodenum negative",
        "label": "overstated",
    },
    {
        "field": "baclofen_regimen",
        "self_report": "15 mg oral 1-2x daily PRN",
        "chart": "compounded 15 mg rectal QHS; oral 10 mg TID and 15 mg TID also listed",
        "label": "route_mismatch",
    },
    {
        "field": "egg_allergy",
        "self_report": "IgE egg",
        "chart": "not on allergy list",
        "label": "missing_from_chart",
    },
    {
        "field": "qbrexza_site",
        "self_report": "face daily",
        "chart": "not on medication list; label is axillae only",
        "label": "missing_from_chart",
    },
    {
        "field": "dupixent_readiness",
        "self_report": "considering Dupixent for EoE",
        "chart": "March EGD mild/indeterminate; BID PPI started 2026-06-24; not TREET-ready",
        "label": "overstated",
    },
    {
        "field": "minoxidil_route",
        "self_report": "2.5 mg daily (route unspecified in profile header)",
        "chart": "Loniten oral tablets since 2025-02-01",
        "label": "agree",
    },
    {
        "field": "thyroid_graves",
        "self_report": "low TSH, goiter, fatigue (implied active endocrine problem)",
        "chart": "recovered TSH, negative antibodies, diffuse goiter, not Graves",
        "label": "overstated",
    },
    {
        "field": "cam_fai",
        "self_report": "not mentioned; pelvic pain attributed only to hypertonic floor / pudendal neuralgia",
        "chart": "bilateral cam FAI (R>L) on XR and MRI; pubic symphysis normal",
        "label": "missing_from_self",
    },
    {
        "field": "zio_monitor",
        "self_report": "not mentioned",
        "chart": "8-day Zio on Loniten: no significant arrhythmia",
        "label": "missing_from_self",
    },
]

MISMATCH_LABELS = [
    "agree",
    "overstated",
    "missing_from_chart",
    "missing_from_self",
    "route_mismatch",
]

DDI_PRIORITIES = ["ignore", "timing", "monitor", "discuss_now"]

CLINICAL_QUESTIONS = [
    "Is the current medication regimen safe given orthostatic lightheadedness?",
    "What drug-drug interactions matter in this patient, not just the full DrugBank list?",
    "Is oral minoxidil a poor match with baclofen and a wide pulse pressure?",
    "Is dupilumab a good fit for eosinophilic esophagitis on this chart?",
    "How should HLA-DQ2 be interpreted versus a celiac diagnosis?",
    "What tools should be used for pairwise FDA and DrugBank interactions if the analyzer agent times out?",
]

# Patient-conditioned DDI priority for THIS case.
# ignore = DrugBank hit that does not apply given current partners/state
# timing = separate administration
# monitor = real but lower urgency
# discuss_now = take to clinician given this body
DDI_RANK_GOLD = [
    {
        "drug_a": "minoxidil",
        "drug_b": "baclofen",
        "description": "The risk or severity of adverse effects can be increased when Baclofen is combined with Minoxidil.",
        "priority": "discuss_now",
        "why": "Named DrugBank pair plus documented orthostasis and 137/60 on oral minoxidil.",
        "split": "holdout_war",
    },
    {
        "drug_a": "lansoprazole",
        "drug_b": "fexofenadine",
        "description": "The excretion of Fexofenadine can be decreased when combined with Lansoprazole.",
        "priority": "monitor",
        "why": "PK excretion; not a contraindication.",
        "split": "holdout_war",
    },
    {
        "drug_a": "fexofenadine",
        "drug_b": "magnesium oxide",
        "description": "Magnesium oxide can cause a decrease in the absorption of Fexofenadine resulting in a reduced serum concentration and potentially a decrease in efficacy.",
        "priority": "timing",
        "why": "Applies if the multi (Mg oxide) is still used; separate from Allegra.",
        "split": "holdout_war",
    },
    {
        "drug_a": "finasteride",
        "drug_b": "minoxidil",
        "description": "Finasteride may decrease the antihypertensive activities of Minoxidil.",
        "priority": "ignore",
        "why": "Theoretical and opposite direction to the hypotensive concern at LDOM 2.5 mg.",
        "split": "holdout_war",
    },
    {
        "drug_a": "tretinoin",
        "drug_b": "minoxidil",
        "description": "Tretinoin may decrease the antihypertensive activities of Minoxidil.",
        "priority": "ignore",
        "why": "Topical acne tretinoin is not the systemic retinoid interaction.",
        "split": "holdout_war",
    },
    {
        "drug_a": "minoxidil",
        "drug_b": "duloxetine",
        "description": "The risk or severity of orthostatic hypotension and syncope can be increased when Minoxidil is combined with Duloxetine.",
        "priority": "ignore",
        "why": "Class PD hypotension; partner is not on this regimen.",
        "split": "holdout_war",
    },
    {
        "drug_a": "minoxidil",
        "drug_b": "levodopa",
        "description": "The risk or severity of hypotension and orthostatic hypotension can be increased when Minoxidil is combined with Levodopa.",
        "priority": "ignore",
        "why": "Partner not on regimen.",
        "split": "holdout_war",
    },
    {
        "drug_a": "dupilumab",
        "drug_b": "minoxidil",
        "description": "No DrugBank partner hit among current regimen drugs (387 listed DDIs, 0 with current partners).",
        "priority": "ignore",
        "why": "No pair hit; Dupixent decision is guideline eligibility, not DDI.",
        "split": "holdout_war",
    },
]

GUIDELINE_ELIGIBILITY = {
    "case_id": CASE_ID,
    "drug": "dupilumab",
    "indication": "eosinophilic_esophagitis",
    "criteria": {
        "weight_ge_40kg": True,
        "high_dose_ppi_weeks_ge_8": False,
        "active_symptoms_impaction_or_frequent_dysphagia": False,
        "peak_eos_ge_15_both_sites": False,
        "erefS_shows_more_than_mild_disease": False,
    },
    "label": "not_eligible_on_this_chart",
    "evidence_grade": "T1/T2",
}

TOOL_TRACE_LABELS = {
    "positives": [
        "FDA_get_drug_interactions_by_drug_name",
        "FDA_get_adverse_reactions_by_drug_name",
        "FDA_search_drug_labels",
        "OpenFDA_search_drug_labels",
        "DailyMed_parse_drug_interactions",
        "drugbank_get_drug_interactions_by_drug_name_or_id",
        "FAERS_search_reports_by_drug_combination",
        "PubMed_search_articles",
    ],
    "negatives_timeout": [
        "DrugInteractionAnalyzerAgent",
    ],
    "negatives_wrong_label_form": [
        "intrathecal baclofen (Gablofen) when oral/rectal was needed",
        "oral tretinoin APL label when topical acne tretinoin was needed",
    ],
}
