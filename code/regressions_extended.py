"""
Argentina Labor Market RCT — Extended Regressions
World Bank / University of Chicago

Adds to the main analysis:
  (a) Two previously-omitted ITT-able admin outcomes:
        - allowed_companies_contact  (job-search effort)
        - receiving_benefit          (benefit receipt; interpret with care re: timing)
  (b) Derived employment outcomes from the emp_status panel:
        - months_employed   (total months active in the 19-month window, 0-19)
        - n_employers       (distinct employers among employed; job switching)
        - emp_latest        (employed in latest month — the original)

Sample: Phase 1 + 2 (master_analysis.csv, ~103,153 individuals).
Specs:  province FE / + controls / + municipality FE  (same as main).

NOTE on what is NOT here:
  Portal-based outcomes (Theory of Change beliefs, Career Plan completion,
  NPS) exist ONLY for platform users and have no control group, so they
  cannot be used for ITT. They are analyzed separately (users-only).

Run:
    uv run regressions_extended.py
"""

import pandas as pd
import numpy as np
import pyreadstat
import statsmodels.formula.api as smf
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────
BASE    = Path.home() / "argentina-rct/data"
MASTER  = Path.home() / "argentina-rct/output/master_analysis.csv"
OUT_DIR = Path.home() / "argentina-rct/output/regressions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

def extract(model, outcome, label, spec):
    rows = []
    for term in ["T1", "T2", "T3"]:
        if term in model.params:
            p = model.pvalues[term]
            rows.append({
                "outcome": outcome, "label": label, "spec": spec, "term": term,
                "coef":   round(model.params[term], 4),
                "se":     round(model.bse[term], 4),
                "pvalue": round(p, 4),
                "nobs":   int(model.nobs),
                "r2":     round(model.rsquared, 4),
                "stars":  stars(p),
            })
    return rows

# ── 1. LOAD ANALYSIS SAMPLE ───────────────────────────────────
print("Loading analysis sample...")
df = pd.read_csv(
    MASTER,
    usecols=lambda c: c in [
        "id_SSE", "treat", "provincia", "muni_id",
        "mujer", "edad_above_med", "educ_secondary",
        "work_experience", "has_children", "fomentar",
        # (a) previously-omitted admin outcomes (already in master)
        "allowed_companies_contact", "receiving_benefit",
    ],
    low_memory=False
)
df["id_SSE"] = pd.to_numeric(df["id_SSE"], errors="coerce")
df["treat"] = pd.to_numeric(df["treat"], errors="coerce")
df["T1"] = (df["treat"] == 1).astype(np.int8)
df["T2"] = (df["treat"] == 2).astype(np.int8)
df["T3"] = (df["treat"] == 3).astype(np.int8)
df["prov_code"] = df["provincia"].astype("category").cat.codes.astype(np.int16)
df["muni_id"]   = pd.to_numeric(df["muni_id"], errors="coerce")
print(f"  Analysis sample: {len(df):,} individuals")

# ── 2. BUILD DERIVED EMPLOYMENT OUTCOMES from panel ───────────
print("Building derived employment outcomes from emp_status panel...")
emp, _ = pyreadstat.read_dta(str(BASE / "ministry/soc_security_emp_status.dta"))

# Collapse to person-month (employed if active with any employer)
pm = emp.groupby(["id_SSE","month_str"])["active_employment"].max().reset_index()

# months_employed: total active months in the window (0-19)
months_employed = pm.groupby("id_SSE")["active_employment"].sum().rename("months_employed")

# n_employers: distinct employers among months the person was active (job switching)
n_emp = (emp[emp["active_employment"]==1]
         .groupby("id_SSE")["Empleador"].nunique()
         .rename("n_employers"))

# emp_latest: employed in most recent month
emp_latest = (pm.sort_values("month_str")
                .groupby("id_SSE")["active_employment"].last()
                .rename("emp_latest"))

derived = pd.concat([months_employed, n_emp, emp_latest], axis=1).reset_index()
df = df.merge(derived, on="id_SSE", how="left")

print(f"  Individuals with employment panel data: {df['months_employed'].notna().sum():,}")

# ── 3. SETTINGS ───────────────────────────────────────────────
controls = [c for c in [
    "mujer", "edad_above_med", "educ_secondary",
    "work_experience", "has_children", "fomentar"
] if c in df.columns]
ctrl_str = " + ".join(controls)

# ── 4. OUTCOMES ───────────────────────────────────────────────
outcomes = {
    # (a) previously-omitted ITT-able admin outcomes
    "allowed_companies_contact": "Allowed companies to contact (admin)",
    "receiving_benefit":         "Receiving social benefit (admin)",
    # (b) derived employment outcomes
    "emp_latest":      "Employed in latest month (derived)",
    "months_employed": "Total months employed 0-19 (derived)",
    "n_employers":     "Distinct employers / job switching (derived)",
}

# ── 5. RUN ────────────────────────────────────────────────────
all_results = []

for var, label in outcomes.items():
    if var not in df.columns:
        print(f"  Skipping {var} — not present")
        continue
    sub = df.dropna(subset=[var, "treat"]).copy()
    sub[var] = pd.to_numeric(sub[var], errors="coerce")
    sub = sub.dropna(subset=[var])
    if len(sub) == 0:
        print(f"  Skipping {label} — no obs")
        continue

    print(f"\n── {label} (n={len(sub):,}) ──")

    # province FE
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + C(prov_code)",
                    data=sub).fit(cov_type="HC3")
        all_results += extract(m, var, label, "1_province_FE")
        print(f"  Province FE  | "
              f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}")
        del m
    except Exception as e:
        print(f"  Province FE failed: {e}")

    # + controls
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(prov_code)",
                    data=sub).fit(cov_type="HC3")
        all_results += extract(m, var, label, "2_controls_province_FE")
        print(f"  + Controls   | "
              f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}")
        del m
    except Exception as e:
        print(f"  + Controls failed: {e}")

    # + municipality FE
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(muni_id)",
                    data=sub.dropna(subset=["muni_id"])).fit(cov_type="HC3")
        all_results += extract(m, var, label, "3_controls_muni_FE")
        print(f"  + Muni FE    | "
              f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}")
        del m
    except Exception as e:
        print(f"  + Muni FE failed: {e}")

    del sub

# ── 6. SAVE ───────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "extended_regressions.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'extended_regressions.csv'}")

# ── 7. SUMMARY ────────────────────────────────────────────────
print("\n── Summary: controls + province FE ──────────────────────")
s = results_df[results_df["spec"] == "2_controls_province_FE"].copy()
s["coef_stars"] = s["coef"].astype(str) + s["stars"]
print(s.pivot_table(index="label", columns="term",
                    values="coef_stars", aggfunc="first").to_string())
print("\n*** p<0.01  ** p<0.05  * p<0.1")
