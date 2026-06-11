import pyreadstat
import pandas as pd
from pathlib import Path
from config import DATA_DIR as BASE, OUT_DIR as OUTDIR, TREAT_MAP

OUTDIR = Path.home() / "argentina-rct/output"
OUTDIR.mkdir(exist_ok=True)

TREAT_MAP = {0: "Control", 1: "T1_CV", 2: "T2_Full", 3: "T3_InDemand"}

# ── 1. RANDOMIZATION SPINE ───────────────────────────────────
print("Loading randomization file...")
rand, _ = pyreadstat.read_dta(str(BASE / "worldbank/individ_randomization.dta"))
rand["treat_label"] = rand["treat"].map(TREAT_MAP)
rand = rand.rename(columns={"id": "id_SSE"})
print(f"  Spine: {len(rand):,} individuals")

# ── 2. CROSSWALK (get id_correo) ─────────────────────────────
print("Merging crosswalk...")
cw, _ = pyreadstat.read_dta(str(BASE / "worldbank/id_SSE_email_crosswalk.dta"))
rand = rand.merge(cw[["id_SSE", "id_correo", "invited", "registered"]],
                  on="id_SSE", how="left")
print(f"  Invited:    {rand['invited'].sum():,.0f}")
print(f"  Registered: {rand['registered'].sum():,.0f}")

# ── 3. ADMIN OUTCOMES ────────────────────────────────────────
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
    # these files just contain id_SSE for people who did the action
    # so merge and flag as 1 if present
    df["id_SSE"] = df["id_SSE"].astype(rand["id_SSE"].dtype)
    df[col] = 1
    rand = rand.merge(df[["id_SSE", col]], on="id_SSE", how="left")
    rand[col] = rand[col].fillna(0).astype(int)
    print(f"  {col}: {rand[col].sum():,} ({rand[col].mean():.2%})")

# ── 4. EMPLOYMENT (latest month per person) ──────────────────
print("Merging employment status...")
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
print(f"  Employment rate (those with data): {rand['emp_latest'].mean():.2%}")

# ── 5. SURVEY ────────────────────────────────────────────────
print("Merging survey...")
survey = pd.read_excel(BASE / "opinaia/2025.08.06 Base.xlsx", sheet_name="Sheet1")
survey = survey.rename(columns={"Id_SSE": "id_SSE"})
survey["id_SSE"] = pd.to_numeric(survey["id_SSE"], errors="coerce")
survey_cols = [c for c in [
    "id_SSE", "P0", "P1", "P2", "P3", "P4",
    "P6", "P8", "P9", "P10", "P11", "P12"
] if c in survey.columns]
rand = rand.merge(survey[survey_cols], on="id_SSE", how="left")
print(f"  Individuals with survey data: {rand['P0'].notna().sum():,}")

# ── 6. SAVE ──────────────────────────────────────────────────
out = OUTDIR / "master.csv"
rand.to_csv(out, index=False)
print(f"\nMaster dataset saved: {out}")
print(f"Final shape: {rand.shape}")
print(f"\nColumns: {list(rand.columns)}")

# ── 7. QUICK SANITY CHECK ────────────────────────────────────
print("\n── Outcome rates by treatment arm ───────────────────────")
outcome_cols = [
    "enrolled_in_course", "applied_to_portal_job",
    "updated_cv", "emp_latest"
]
print(rand.groupby("treat_label")[outcome_cols].mean().T.to_string())
