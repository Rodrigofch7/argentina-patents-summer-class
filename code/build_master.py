"""
Argentina Labor Market RCT — Build Master Dataset
World Bank / University of Chicago

Builds the individual-level master dataset by merging all sources on id_SSE.
Keeps ALL columns from every source (drop unneeded columns later in analysis).

Adds an analysis_sample flag for Phase 1 + Phase 2 individuals (~103,153),
since Phase 3 (931,197 individuals) was rolled out too late to have linked
outcome data and is dropped from the main analysis.

Run:
    uv run build_master.py
"""

import pyreadstat
import pandas as pd
from pathlib import Path

BASE   = Path.home() / "argentina-rct/data"
OUTDIR = Path.home() / "argentina-rct/output"
OUTDIR.mkdir(exist_ok=True)

TREAT_MAP = {0: "Control", 1: "T1_CV", 2: "T2_Full", 3: "T3_InDemand"}

# ── 1. RANDOMIZATION SPINE (keep ALL columns) ─────────────────
print("Loading randomization file...")
rand, _ = pyreadstat.read_dta(str(BASE / "worldbank/individ_randomization.dta"))
rand = rand.rename(columns={"id": "id_SSE"})
rand["treat_label"] = rand["treat"].map(TREAT_MAP)
print(f"  Spine: {len(rand):,} individuals, {rand.shape[1]} columns")
print(f"  Phases: {rand['treat_phase'].value_counts().sort_index().to_dict()}")

# ── 2. CROSSWALK (keep ALL columns, suffix duplicates) ────────
print("Merging crosswalk (id_correo, invited, registered, merge flags)...")
cw, _ = pyreadstat.read_dta(str(BASE / "worldbank/id_SSE_email_crosswalk.dta"))
# Avoid duplicate-column collisions: suffix overlapping names except the key
overlap = [c for c in cw.columns if c in rand.columns and c != "id_SSE"]
cw = cw.rename(columns={c: f"{c}_cw" for c in overlap})
rand = rand.merge(cw, on="id_SSE", how="left")
print(f"  After crosswalk: {rand.shape[1]} columns")

# ── 3. ADMIN OUTCOMES (flag = 1 if present) ───────────────────
print("Merging admin outcomes...")
admin = [
    ("receiving_benefit",        BASE / "ministry/receiving_benefits_variable.dta"),
    ("enrolled_in_course",       BASE / "ministry/soc_security_data_file1.dta"),
    ("applied_to_portal_job",    BASE / "ministry/soc_security_data_file2.dta"),
    ("allowed_companies_contact",BASE / "ministry/soc_security_data_file3.dta"),
    ("updated_cv",               BASE / "ministry/soc_security_data_file4.dta"),
]
for col, path in admin:
    df, _ = pyreadstat.read_dta(str(path))
    df["id_SSE"] = df["id_SSE"].astype(rand["id_SSE"].dtype)
    df[col] = 1
    rand = rand.merge(df[["id_SSE", col]].drop_duplicates("id_SSE"),
                      on="id_SSE", how="left")
    rand[col] = rand[col].fillna(0).astype(int)
    print(f"  {col}: {rand[col].sum():,} ({rand[col].mean():.2%})")

# ── 4. EMPLOYMENT (latest month per person) ───────────────────
print("Merging employment status (latest month)...")
emp, _ = pyreadstat.read_dta(str(BASE / "ministry/soc_security_emp_status.dta"))
emp_latest = (
    emp.sort_values("month_str", ascending=False)
       .drop_duplicates("id_SSE")
       [["id_SSE", "active_employment", "month_str"]]
       .rename(columns={"active_employment": "emp_latest",
                        "month_str": "emp_latest_month"})
)
rand = rand.merge(emp_latest, on="id_SSE", how="left")
print(f"  Individuals with employment data: {rand['emp_latest'].notna().sum():,}")

# ── 5. SURVEY (keep all P-variables) ──────────────────────────
print("Merging survey...")
survey = pd.read_excel(BASE / "opinaia/2025.08.06 Base.xlsx", sheet_name="Sheet1")
survey = survey.rename(columns={"Id_SSE": "id_SSE"})
survey["id_SSE"] = pd.to_numeric(survey["id_SSE"], errors="coerce")
# Suffix any survey columns that collide with existing names
overlap_s = [c for c in survey.columns if c in rand.columns and c != "id_SSE"]
survey = survey.rename(columns={c: f"{c}_svy" for c in overlap_s})
rand = rand.merge(survey, on="id_SSE", how="left")
print(f"  Individuals with survey data: {rand['P0'].notna().sum() if 'P0' in rand.columns else 0:,}")

# ── 6. ANALYSIS SAMPLE FLAG (Phase 1 + 2, drop Phase 3) ───────
# Phase 3 (931,197 individuals) was rolled out too late to have linked
# outcome data; the main analysis uses Phase 1 + Phase 2 (~103,153).
rand["analysis_sample"] = rand["treat_phase"].isin([1.0, 2.0]).astype(int)
print(f"\n  Analysis sample (Phase 1+2): {rand['analysis_sample'].sum():,} individuals")
print(f"  Dropped (Phase 3):           {(rand['analysis_sample']==0).sum():,} individuals")

# ── 7. SAVE FULL MASTER (all columns, all individuals) ────────
out_full = OUTDIR / "master.csv"
rand.to_csv(out_full, index=False)
print(f"\nFull master saved: {out_full}")
print(f"  Shape: {rand.shape[0]:,} rows x {rand.shape[1]} columns")

# ── 8. SAVE ANALYSIS SUBSET (Phase 1+2 only) ──────────────────
analysis = rand[rand["analysis_sample"] == 1].copy()
out_analysis = OUTDIR / "master_analysis.csv"
analysis.to_csv(out_analysis, index=False)
print(f"\nAnalysis subset saved: {out_analysis}")
print(f"  Shape: {analysis.shape[0]:,} rows x {analysis.shape[1]} columns")

# ── 9. SANITY CHECK ───────────────────────────────────────────
print("\n-- Treatment balance in analysis sample ------------------")
print(analysis["treat"].value_counts().sort_index().to_string())

print("\n-- Outcome rates by treatment arm (analysis sample) ------")
outcome_cols = [c for c in [
    "enrolled_in_course", "applied_to_portal_job",
    "updated_cv", "emp_latest"
] if c in analysis.columns]
print(analysis.groupby("treat_label")[outcome_cols].mean().T.to_string())