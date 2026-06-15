"""
Argentina Labor Market RCT — Regressions
World Bank / University of Chicago

Specifications:
    1. ITT basic:          Y ~ T1 + T2 + T3 + province_FE
    2. ITT controls:       Y ~ T1 + T2 + T3 + controls + province_FE
    3. ITT month FE:       Y ~ T1 + T2 + T3 + controls + province_FE + month_FE
    4. ITT municipality FE:Y ~ T1 + T2 + T3 + controls + municipality_FE + month_FE
    5. IV 2SLS (TOT):      instruments T1/T2/T3 for actual SkillLab use

Why month FE?
    The portal was rolled out in two phases:
      - Phase 1: January 2025 (pilot)
      - Phase 2: June 2025 (actual RCT arms T1/T2/T3)
    Employment outcomes are measured through July 2025, meaning Phase 2
    participants had only 1-2 months of exposure. Without controlling for
    month, we might confuse seasonal/economic trends with treatment effects.
    Month FE removes any economy-wide shocks common to all individuals
    in the same month, isolating the within-month treatment effect.

Why municipality FE?
    Randomization was stratified by municipality (muni_id). Adding municipality
    FE controls for local labor market conditions (e.g. one city having a large
    employer open/close), which reduces residual variance and improves precision.
    Municipality FE is stricter than province FE — it absorbs more local variation.
    Note: municipality FE is only feasible for the employment outcome (n=6,026)
    and survey outcomes (n~1,000). For the full 1M sample it would be too slow.

Run:
    uv run regressions.py
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────
MASTER  = Path.home() / "argentina-rct/output/master.csv"
OUT_DIR = Path.home() / "argentina-rct/output/regressions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD — only columns we need ───────────────────────────────
print("Loading master dataset...")
needed_cols = [
    "treat", "provincia", "municipio", "muni_id",
    "mujer", "edad_above_med", "educ_secondary",
    "work_experience", "has_children", "fomentar",
    "registered",
    # admin outcomes
    "enrolled_in_course", "applied_to_portal_job",
    "updated_cv", "emp_latest", "emp_latest_month",
    # survey outcomes
    "P0", "P3", "P8", "P9", "P4", "P6",
]

df = pd.read_csv(MASTER, usecols=lambda c: c in needed_cols, low_memory=False)

# Treatment dummies (Control = reference group)
df["treat"] = pd.to_numeric(df["treat"], errors="coerce")
df["T1"] = (df["treat"] == 1).astype(np.int8)
df["T2"] = (df["treat"] == 2).astype(np.int8)
df["T3"] = (df["treat"] == 3).astype(np.int8)

# Fix mixed-type survey columns
for col in ["P4", "P6", "P9", "P0", "P3", "P8"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Take-up variable: 1 if person actually used SkillLab
# Used as the endogenous variable in the IV regression
if "registered" in df.columns:
    df["used_skilllab"] = pd.to_numeric(
        df["registered"], errors="coerce").fillna(0).astype(np.int8)

# Month FE variable: extracted from emp_latest_month (format: YYYY-MM)
# This controls for economy-wide shocks in each calendar month
if "emp_latest_month" in df.columns:
    df["emp_month"] = df["emp_latest_month"].astype(str).str[:7]
    df["emp_month"] = df["emp_month"].where(df["emp_month"].str.match(r"\d{4}-\d{2}"), np.nan)

# Municipality: use muni_id as integer for memory efficiency
# Municipality FE absorbs local labor market conditions (stricter than province FE)
if "muni_id" in df.columns:
    df["muni_id"] = pd.to_numeric(df["muni_id"], errors="coerce")

# Province: encode as integer category (memory efficient)
if "provincia" in df.columns:
    df["prov_code"] = df["provincia"].astype("category").cat.codes.astype(np.int16)
    del df["provincia"]

print(f"  Shape: {df.shape}")
print(f"  RAM: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"  Take-up: {df['used_skilllab'].sum():,} ({df['used_skilllab'].mean():.2%})")
print(f"  Months in employment data: {df['emp_month'].dropna().unique() if 'emp_month' in df.columns else 'N/A'}")

# ── CONTROLS ──────────────────────────────────────────────────
# Individual-level controls selected per PAP Section 5.1
controls = [c for c in [
    "mujer",          # gender
    "edad_above_med", # age above median
    "educ_secondary", # secondary education
    "work_experience",# prior work experience
    "has_children",   # has minor children
    "fomentar",       # Fomentar vs VAT program
] if c in df.columns]
ctrl_str = " + ".join(controls)

# ── OUTCOMES ──────────────────────────────────────────────────
# Admin outcomes: binary (0/1), full sample ~1M
# Note: emp_latest only has 6,026 obs (small subset with employment records)
admin_outcomes = {
    "enrolled_in_course":    "Enrolled in course (admin)",
    "applied_to_portal_job": "Applied to job via portal (admin)",
    "updated_cv":            "Updated CV (admin)",
    "emp_latest":            "Formally employed latest month (admin)",
}
# Survey outcomes: continuous/binary, only ~1,000 obs (midline survey)
survey_outcomes = {
    "P9": "Job-finding confidence 1-10 (survey)",
    "P0": "Days worked last month (survey)",
    "P3": "Jobs applied last month (survey)",
    "P8": "Reservation wage ARS (survey)",
    "P4": "Discovered new skill (survey)",
    "P6": "Considered new career (survey)",
}
all_outcomes = {**admin_outcomes, **survey_outcomes}

# ── HELPERS ───────────────────────────────────────────────────
def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

def extract_ols(model, outcome, label, spec):
    rows = []
    for term in ["T1", "T2", "T3"]:
        if term in model.params:
            p = model.pvalues[term]
            rows.append({
                "outcome": outcome, "label": label, "spec": spec, "term": term,
                "coef":    round(model.params[term], 6),
                "se":      round(model.bse[term], 6),
                "pvalue":  round(p, 4),
                "ci_low":  round(model.conf_int().loc[term, 0], 6),
                "ci_high": round(model.conf_int().loc[term, 1], 6),
                "nobs":    int(model.nobs),
                "r2":      round(model.rsquared, 4),
                "stars":   stars(p),
            })
    return rows

def extract_iv(model, outcome, label):
    term = "used_skilllab"
    if term not in model.params.index:
        return []
    p = float(model.pvalues[term])
    return [{
        "outcome": outcome, "label": label,
        "spec":    "IV 2SLS (TOT)", "term": "used_skilllab",
        "coef":    round(float(model.params[term]), 6),
        "se":      round(float(model.std_errors[term]), 6),
        "pvalue":  round(p, 4),
        "ci_low":  round(float(model.params[term] - 1.96 * model.std_errors[term]), 6),
        "ci_high": round(float(model.params[term] + 1.96 * model.std_errors[term]), 6),
        "nobs":    int(model.nobs),
        "r2":      np.nan,
        "stars":   stars(p),
    }]

def run_iv(sub, var, label, strata_col):
    """
    IV 2SLS (TOT): instruments T1/T2/T3 for actual SkillLab use.
    Identifies the LATE (Local Average Treatment Effect) for compliers —
    people who used SkillLab because they were randomly invited.
    Uses numeric strata variable (not dummies) to avoid RAM explosion.
    """
    if "used_skilllab" not in sub.columns:
        return []
    try:
        sub_iv = sub.dropna(subset=["used_skilllab"]).reset_index(drop=True)

        # Exog matrix: intercept + numeric strata (avoids huge dummy matrix)
        exog_cols = {"const": np.ones(len(sub_iv))}
        if strata_col and strata_col in sub_iv.columns:
            exog_cols[strata_col] = sub_iv[strata_col].astype(float).values
        exog_df = pd.DataFrame(exog_cols, index=sub_iv.index)

        iv_model = IV2SLS(
            dependent=sub_iv[var].astype(float),
            exog=exog_df,
            endog=sub_iv["used_skilllab"].astype(float),
            instruments=sub_iv[["T1", "T2", "T3"]].astype(float),
        ).fit(cov_type="robust")

        rows = extract_iv(iv_model, var, label)
        coef = float(iv_model.params["used_skilllab"])
        pval = float(iv_model.pvalues["used_skilllab"])
        print(f"  IV (TOT)        | used_skilllab={coef:.4f}{stars(pval)}  p={pval:.3f}")
        del iv_model
        return rows
    except Exception as e:
        print(f"  IV failed: {e}")
        return []

# ── RUN REGRESSIONS ───────────────────────────────────────────
all_results = []

for var, label in all_outcomes.items():
    if var not in df.columns:
        print(f"  Skipping {var} — not in master")
        continue

    sub = df.dropna(subset=["treat"]).copy()
    sub[var] = pd.to_numeric(sub[var], errors="coerce")
    sub = sub.dropna(subset=[var])

    if len(sub) == 0:
        print(f"  Skipping {var} — no valid observations")
        continue

    # Decide on strata for this outcome
    # For large samples (admin outcomes except emp): use province to save RAM
    # For small samples (emp, survey): use municipality for more precision
    is_small_sample = len(sub) < 10000
    strata_col = "muni_id" if is_small_sample else "prov_code"

    # Check if month FE is feasible
    has_month = "emp_month" in sub.columns and sub["emp_month"].notna().sum() > 0

    print(f"\n── {label} (n={len(sub):,}) ──────────────────────────")
    print(f"   Strata: {strata_col} | Month FE: {has_month}")

    # SPEC 1: ITT basic + province FE
    try:
        m = smf.ols(
            f"{var} ~ T1 + T2 + T3 + C(prov_code)", data=sub
        ).fit(cov_type="HC3")
        all_results += extract_ols(m, var, label, "1_ITT_province_FE")
        print(f"  ITT province FE | "
              f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}")
        del m
    except Exception as e:
        print(f"  ITT province FE failed: {e}")

    # SPEC 2: ITT + controls + province FE
    if ctrl_str:
        try:
            m = smf.ols(
                f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(prov_code)", data=sub
            ).fit(cov_type="HC3")
            all_results += extract_ols(m, var, label, "2_ITT_controls_province_FE")
            print(f"  ITT + controls  | "
                  f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
                  f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
                  f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}")
            del m
        except Exception as e:
            print(f"  ITT controls failed: {e}")

    # SPEC 3: ITT + controls + province FE + month FE
    # Month FE removes calendar-time shocks (seasonal patterns, economic cycles)
    # Only feasible when emp_month is available (employment outcome)
    if has_month and sub["emp_month"].notna().sum() > 10:
        try:
            m = smf.ols(
                f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(prov_code) + C(emp_month)",
                data=sub.dropna(subset=["emp_month"])
            ).fit(cov_type="HC3")
            all_results += extract_ols(m, var, label, "3_ITT_month_FE")
            print(f"  ITT + month FE  | "
                  f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
                  f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
                  f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}")
            del m
        except Exception as e:
            print(f"  ITT month FE failed: {e}")

    # SPEC 4: ITT + controls + municipality FE + month FE
    # Municipality FE is stricter — absorbs local labor market conditions
    # Only run on small samples to avoid RAM issues
    if is_small_sample and "muni_id" in sub.columns:
        month_fe = "+ C(emp_month)" if has_month and sub["emp_month"].notna().sum() > 10 else ""
        sub_muni = sub.dropna(subset=["muni_id", "emp_month"] if has_month else ["muni_id"])
        if len(sub_muni) > 0:
            try:
                m = smf.ols(
                    f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(muni_id) {month_fe}",
                    data=sub_muni
                ).fit(cov_type="HC3")
                all_results += extract_ols(m, var, label, "4_ITT_muni_FE_month_FE")
                print(f"  ITT + muni FE   | "
                      f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
                      f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
                      f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}")
                del m
            except Exception as e:
                print(f"  ITT muni FE failed: {e}")

    # SPEC 5: IV 2SLS (TOT)
    all_results += run_iv(sub, var, label, strata_col)

    del sub

# ── FIRST STAGE ───────────────────────────────────────────────
print("\n── First Stage: Does assignment predict take-up? ────────")
print("   (F-stat > 10 = strong instrument)")
first_stage_rows = []
if "used_skilllab" in df.columns:
    sub_fs = df.dropna(subset=["used_skilllab", "treat"]).copy()
    try:
        fs = smf.ols(
            "used_skilllab ~ T1 + T2 + T3 + C(prov_code)",
            data=sub_fs
        ).fit(cov_type="HC3")
        for term in ["T1", "T2", "T3"]:
            p = fs.pvalues.get(term, np.nan)
            print(f"  {term}: coef={fs.params.get(term,np.nan):.4f}{stars(p)}  p={p:.3f}")
            first_stage_rows.append({
                "term":   term,
                "coef":   round(fs.params.get(term, np.nan), 6),
                "se":     round(fs.bse.get(term, np.nan), 6),
                "pvalue": round(p, 4),
                "stars":  stars(p),
                "fstat":  round(fs.fvalue, 2),
                "nobs":   int(fs.nobs),
            })
        print(f"  F-stat: {fs.fvalue:.2f}")
    except Exception as e:
        print(f"  First stage failed: {e}")

# ── SAVE ──────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "all_regressions.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'all_regressions.csv'}")

if first_stage_rows:
    pd.DataFrame(first_stage_rows).to_csv(OUT_DIR / "first_stage.csv", index=False)
    print(f"Saved: {OUT_DIR / 'first_stage.csv'}")

# ── SUMMARY TABLES ────────────────────────────────────────────
for spec in results_df["spec"].unique():
    print(f"\n── Summary: {spec} ──────────────────────────────────")
    sub = results_df[results_df["spec"] == spec].copy()
    sub["coef_stars"] = sub["coef"].round(4).astype(str) + sub["stars"]
    pivot = sub.pivot_table(
        index="label", columns="term",
        values="coef_stars", aggfunc="first"
    )
    print(pivot.to_string())

results_df.pivot_table(
    index=["label", "spec"], columns="term",
    values=["coef", "stars", "pvalue"], aggfunc="first"
).to_csv(OUT_DIR / "summary_all_specs.csv")
print(f"\nSaved: {OUT_DIR / 'summary_all_specs.csv'}")
print("\n*** p<0.01  ** p<0.05  * p<0.1")