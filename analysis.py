import pyreadstat
import pandas as pd
from pathlib import Path

BASE = Path.home() / "argentina-rct/data"

# ── 1. RANDOMIZATION FILE ────────────────────────────────────
print("=" * 50)
print("RANDOMIZATION FILE")
print("=" * 50)
rand, _ = pyreadstat.read_dta(str(BASE / "worldbank/individ_randomization.dta"))
print(f"Shape: {rand.shape}")
print(f"\nTreatment distribution:")
print(rand["treat"].value_counts().sort_index())
print(f"\nGender (mujer=1 is female):")
print(rand["mujer"].value_counts())
print(f"\nAge (above median):")
print(rand["edad_above_med"].value_counts())
print(f"\nEducation:")
print(f"  Primary only:    {rand['educ_primary'].mean():.2%}")
print(f"  Secondary:       {rand['educ_secondary'].mean():.2%}")
print(f"\nWork experience:  {rand['work_experience'].mean():.2%}")
print(f"Has children:     {rand['has_children'].mean():.2%}")
print(f"Fomentar (vs VAT):{rand['fomentar'].mean():.2%}")
print(f"\nTop provinces:")
print(rand["provincia"].value_counts().head(8))

# ── 2. CROSSWALK ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("CROSSWALK SSE <-> EMAIL")
print("=" * 50)
cw, _ = pyreadstat.read_dta(str(BASE / "worldbank/id_SSE_email_crosswalk.dta"))
print(f"Shape: {cw.shape}")
print(f"Columns: {list(cw.columns)}")

# ── 3. ADMIN OUTCOMES ────────────────────────────────────────
print("\n" + "=" * 50)
print("ADMIN OUTCOMES")
print("=" * 50)
outcomes = {
    "enrolled_in_course":        BASE / "ministry/soc_security_data_file1.dta",
    "applied_to_portal_job":     BASE / "ministry/soc_security_data_file2.dta",
    "allowed_companies_contact": BASE / "ministry/soc_security_data_file3.dta",
    "updated_cv":                BASE / "ministry/soc_security_data_file4.dta",
    "receiving_benefit":         BASE / "ministry/receiving_benefits_variable.dta",
}
for col, path in outcomes.items():
    df, _ = pyreadstat.read_dta(str(path))
    print(f"  {col}: {len(df):,} individuals ({len(df)/len(rand):.1%} of sample)")

# ── 4. EMPLOYMENT STATUS ─────────────────────────────────────
print("\n" + "=" * 50)
print("EMPLOYMENT STATUS")
print("=" * 50)
emp, _ = pyreadstat.read_dta(str(BASE / "ministry/soc_security_emp_status.dta"))
print(f"Shape: {emp.shape}")
print(f"Months covered: {emp['month_str'].min()} to {emp['month_str'].max()}")
print(f"Unique individuals: {emp['id_SSE'].nunique():,}")
print(f"Overall employment rate: {emp['active_employment'].mean():.2%}")
print(f"\nMonthly employment rate (last 6 months):")
print(emp.groupby("month_str")["active_employment"].mean().tail(6).to_string())

# ── 5. SURVEY ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("MIDLINE SURVEY")
print("=" * 50)
survey = pd.read_excel(BASE / "opinaia/2025.08.06 Base.xlsx", sheet_name="Sheet1")
print(f"Shape: {survey.shape}")
print(f"\nTreatment distribution in survey:")
print(survey["Grupo"].value_counts())
print(f"\nKey outcomes (means):")
for col, label in [
    ("P0", "Days worked last month"),
    ("P2", "Hours job searching/week"),
    ("P3", "Jobs applied last month"),
    ("P8", "Reservation wage (ARS)"),
    ("P9", "Job-finding confidence (1-10)"),
]:
    if col in survey.columns:
        print(f"  {label}: {pd.to_numeric(survey[col], errors='coerce').mean():.2f}")

# ── 6. PORTAL DATA ───────────────────────────────────────────
print("\n" + "=" * 50)
print("SKILLLAB PORTAL DATA")
print("=" * 50)
xl = pd.ExcelFile(BASE / "skilllab/28Oct2025_Raw_Data_Report_Argentina.xlsx")
for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    print(f"  {sheet}: {len(df):,} rows")

