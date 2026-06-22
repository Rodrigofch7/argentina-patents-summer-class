"""
Argentina Labor Market RCT — Verify EDA Summary Table Numbers
World Bank / University of Chicago

Checks every figure in table_eda.tex against the actual master dataset:
  - N, mean, SD for each variable
  - the registered/take-up discrepancy (0.001 vs 1.27% reported elsewhere)
  - confirms which file the SkillLab portal vars (sl_n_skills, sl_n_cvs,
    sl_satisfaction, employed_jun2025/jul2025) actually live in, since
    they may not be in master.csv at all

Does NOT modify any data. Read-only checks, prints a report.

Run:
    uv run verify_eda_table.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path.home() / "argentina-rct/output"
MASTER_FULL = OUT_DIR / "master.csv"
MASTER_ANALYSIS = OUT_DIR / "master_analysis.csv"

pd.set_option("display.width", 160)


def report_var(df, col, label, binary=False):
    if col not in df.columns:
        print(f"  [MISSING] {col} ({label}) — not in this file")
        return
    s = pd.to_numeric(df[col], errors="coerce")
    n = s.notna().sum()
    mean = s.mean()
    sd = s.std()
    print(f"  {col:28s} N={n:>8,}  mean={mean:.4f}  sd={sd:.4f}")
    if binary and n > 0:
        implied_sd = np.sqrt(mean * (1 - mean))
        print(f"      (binary check: sqrt(p(1-p)) = {implied_sd:.4f} "
              f"{'OK' if abs(implied_sd - sd) < 0.01 else '*** MISMATCH ***'})")


print("=" * 70)
print("LOADING FILES")
print("=" * 70)
print(f"Full master exists:     {MASTER_FULL.exists()}  -> {MASTER_FULL}")
print(f"Analysis subset exists: {MASTER_ANALYSIS.exists()}  -> {MASTER_ANALYSIS}")

if MASTER_ANALYSIS.exists():
    df = pd.read_csv(MASTER_ANALYSIS, low_memory=False)
    print(f"\nLoaded master_analysis.csv: {df.shape[0]:,} rows x {df.shape[1]} cols")
else:
    raise SystemExit("master_analysis.csv not found — run build_master.py first")

print("\n" + "=" * 70)
print("PANEL A — Treatment Assignment (T1/T2/T3 shares)")
print("=" * 70)
if "treat" in df.columns:
    t = pd.to_numeric(df["treat"], errors="coerce")
    print(t.value_counts(normalize=False).sort_index().to_string())
    print(t.value_counts(normalize=True).sort_index().round(4).to_string())
else:
    print("  [MISSING] treat column not found")

print("\n" + "=" * 70)
print("PANEL B — Demographics")
print("=" * 70)
report_var(df, "edad", "Age", binary=False)
report_var(df, "mujer", "Female", binary=True)
report_var(df, "educ_primary", "Primary educ.", binary=True)
report_var(df, "educ_secondary", "Secondary educ.", binary=True)
report_var(df, "has_children", "Has children", binary=True)
report_var(df, "work_experience", "Work experience", binary=True)

print("\n" + "=" * 70)
print("PANEL C — Take-up / Administrative Outcomes (full sample)")
print("=" * 70)
report_var(df, "invited", "Invited", binary=True)
report_var(df, "registered", "Registered (take-up)", binary=True)
report_var(df, "updated_cv", "Updated CV", binary=True)
report_var(df, "enrolled_in_course", "Enrolled in course", binary=True)
report_var(df, "applied_to_portal_job", "Applied to job", binary=True)
report_var(df, "allowed_companies_contact", "Allowed contact", binary=True)
report_var(df, "receiving_benefit", "Receiving benefit", binary=True)

# ── THE KEY DISCREPANCY CHECK ──────────────────────────────────
print("\n" + "=" * 70)
print("*** TAKE-UP DISCREPANCY CHECK ***")
print("=" * 70)
print("regressions.py reported: Take-up = 1,314/103,153 = 1.27%")
print("table_eda.tex showed:    registered mean = 0.001 (= 0.1%)")
print()
if "registered" in df.columns:
    reg = pd.to_numeric(df["registered"], errors="coerce")
    print(f"  registered: sum={reg.sum():.0f}  mean={reg.mean():.4f}  "
          f"({reg.mean():.2%})  N={reg.notna().sum():,}")
if "used_skilllab" in df.columns:
    uss = pd.to_numeric(df["used_skilllab"], errors="coerce")
    print(f"  used_skilllab: sum={uss.sum():.0f}  mean={uss.mean():.4f}  "
          f"({uss.mean():.2%})  N={uss.notna().sum():,}")
print("\n  -> If these two don't match, 'registered' and 'used_skilllab'")
print("     may be different columns/definitions. Check regressions.py's")
print("     line: df['used_skilllab'] = pd.to_numeric(df['registered'],...)")
print("     against whatever script produced the EDA table — they may be")
print("     reading take-up from different source files (e.g. the")
print("     id_SSE_email_crosswalk.dta 'registered' column vs. master's).")

print("\n" + "=" * 70)
print("PANEL D — Employment by month (need raw emp_status panel, not master)")
print("=" * 70)
emp_path = Path.home() / "argentina-rct/data/ministry/soc_security_emp_status.dta"
try:
    import pyreadstat
    emp, _ = pyreadstat.read_dta(str(emp_path))
    pm = emp.groupby(["id_SSE", "month_str"])["active_employment"].max().reset_index()
    for month in ["2025-06", "2025-07"]:
        sub = pm[pm["month_str"] == month]
        if len(sub) > 0:
            print(f"  Employed, {month}: N={len(sub):,}  "
                  f"mean={sub['active_employment'].mean():.4f}  "
                  f"sd={sub['active_employment'].std():.4f}")
        else:
            print(f"  [No rows found for month_str == '{month}'] "
                  f"-- check exact month_str format with: "
                  f"pm['month_str'].unique()")
    print(f"\n  Distinct individuals in emp_status panel: {emp['id_SSE'].nunique():,}")
except Exception as e:
    print(f"  Could not load emp_status panel: {e}")

print("\n" + "=" * 70)
print("PANEL E — SkillLab portal engagement (registered users only)")
print("=" * 70)
print("  NOTE: sl_n_skills / sl_n_cvs / sl_satisfaction are NOT in master.csv.")
print("  They must come from the raw SkillLab portal export. Checking...")
skilllab_path = Path.home() / "argentina-rct/data/skilllab/28Oct2025_Raw_Data_Report_Argentina.xlsx"
try:
    xl = pd.ExcelFile(skilllab_path)
    print(f"  Sheets available: {xl.sheet_names}")
    print("  -> Open each sheet and look for columns matching skills/CVs/satisfaction.")
    print("     Example: xl.parse('Skills').columns, xl.parse('CVs').columns, etc.")
except Exception as e:
    print(f"  Could not open SkillLab file: {e}")

print("\n" + "=" * 70)
print("DONE. Compare every printed N/mean/SD above against table_eda.tex.")
print("=" * 70)
