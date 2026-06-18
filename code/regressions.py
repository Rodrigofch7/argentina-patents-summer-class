"""
Argentina Labor Market RCT — Main Regressions
World Bank / University of Chicago

Sample: Phase 1 + Phase 2 individuals only (103,153), from master_analysis.csv.
        Phase 3 (931,197) was rolled out too late to have linked outcomes
        and is excluded.

Cross-sectional ITT specification (one row per individual):
    1. ITT province FE:           Y ~ T1 + T2 + T3 + province_FE
    2. ITT controls + province FE:Y ~ T1 + T2 + T3 + controls + province_FE
    3. ITT controls + muni FE:    Y ~ T1 + T2 + T3 + controls + municipality_FE
    4. IV 2SLS (TOT):             instruments T1/T2/T3 for actual SkillLab use

Controls: gender, age above median, secondary education, work experience,
          children, program (Fomentar vs VAT).

Run:
    uv run regressions.py
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────
MASTER  = Path.home() / "argentina-rct/output/master_analysis.csv"
OUT_DIR = Path.home() / "argentina-rct/output/regressions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD — only columns we need ───────────────────────────────
print("Loading analysis dataset (Phase 1 + 2)...")
needed = [
    "treat", "treat_phase", "provincia", "muni_id",
    "mujer", "edad_above_med", "educ_secondary",
    "work_experience", "has_children", "fomentar",
    "registered",
    "enrolled_in_course", "applied_to_portal_job",
    "updated_cv", "emp_latest",
]
df = pd.read_csv(MASTER, usecols=lambda c: c in needed, low_memory=False)

df["treat"] = pd.to_numeric(df["treat"], errors="coerce")
df["T1"] = (df["treat"] == 1).astype(np.int8)
df["T2"] = (df["treat"] == 2).astype(np.int8)
df["T3"] = (df["treat"] == 3).astype(np.int8)

# Take-up: actual SkillLab usage (endogenous var for IV)
if "registered" in df.columns:
    df["used_skilllab"] = pd.to_numeric(
        df["registered"], errors="coerce").fillna(0).astype(np.int8)

# Memory-efficient strata codes
df["prov_code"] = df["provincia"].astype("category").cat.codes.astype(np.int16)
df["muni_id"]   = pd.to_numeric(df["muni_id"], errors="coerce")

print(f"  Shape: {df.shape}")
print(f"  Individuals: {len(df):,}")
print(f"  Take-up: {df['used_skilllab'].sum():,} ({df['used_skilllab'].mean():.2%})")

# ── SETTINGS ──────────────────────────────────────────────────
controls = [c for c in [
    "mujer", "edad_above_med", "educ_secondary",
    "work_experience", "has_children", "fomentar"
] if c in df.columns]
ctrl_str = " + ".join(controls)

def fe(col):
    return f" + C({col})" if col else ""

# ── OUTCOMES (admin, binary) ──────────────────────────────────
outcomes = {
    "enrolled_in_course":    "Enrolled in course",
    "applied_to_portal_job": "Applied to job via portal",
    "updated_cv":            "Updated CV",
    "emp_latest":            "Formally employed (latest month)",
}

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
        "spec":    "4_IV_2SLS_TOT", "term": "used_skilllab",
        "coef":    round(float(model.params[term]), 6),
        "se":      round(float(model.std_errors[term]), 6),
        "pvalue":  round(p, 4),
        "ci_low":  round(float(model.params[term] - 1.96*model.std_errors[term]), 6),
        "ci_high": round(float(model.params[term] + 1.96*model.std_errors[term]), 6),
        "nobs":    int(model.nobs),
        "r2":      np.nan,
        "stars":   stars(p),
    }]

# ── RUN ───────────────────────────────────────────────────────
all_results = []

for var, label in outcomes.items():
    if var not in df.columns:
        print(f"  Skipping {var} — not in data")
        continue

    sub = df.dropna(subset=["treat"]).copy()
    sub[var] = pd.to_numeric(sub[var], errors="coerce")
    sub = sub.dropna(subset=[var])

    if len(sub) == 0:
        print(f"  Skipping {var} — no valid observations")
        continue

    is_small = len(sub) < 10000  # emp_latest is small (~6k)
    print(f"\n── {label} (n={len(sub):,}) ──────────────────────────")

    # SPEC 1: ITT + province FE
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + C(prov_code)",
                    data=sub).fit(cov_type="HC3")
        all_results += extract_ols(m, var, label, "1_ITT_province_FE")
        print(f"  Province FE     | "
              f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}  N={int(m.nobs):,}")
        del m
    except Exception as e:
        print(f"  Province FE failed: {e}")

    # SPEC 2: ITT + controls + province FE
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(prov_code)",
                    data=sub).fit(cov_type="HC3")
        all_results += extract_ols(m, var, label, "2_ITT_controls_province_FE")
        print(f"  + Controls      | "
              f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}  N={int(m.nobs):,}")
        del m
    except Exception as e:
        print(f"  + Controls failed: {e}")

    # SPEC 3: ITT + controls + municipality FE
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(muni_id)",
                    data=sub.dropna(subset=["muni_id"])).fit(cov_type="HC3")
        all_results += extract_ols(m, var, label, "3_ITT_controls_muni_FE")
        print(f"  + Municipality  | "
              f"T1={m.params.get('T1',np.nan):.4f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.4f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.4f}{stars(m.pvalues.get('T3',1))}  N={int(m.nobs):,}")
        del m
    except Exception as e:
        print(f"  + Municipality failed: {e}")

    # SPEC 4: IV 2SLS (TOT)
    if "used_skilllab" in sub.columns:
        try:
            sub_iv = sub.dropna(subset=["used_skilllab"]).reset_index(drop=True)
            exog_cols = {"const": np.ones(len(sub_iv))}
            exog_cols["prov_code"] = sub_iv["prov_code"].astype(float).values
            exog_df = pd.DataFrame(exog_cols, index=sub_iv.index)

            iv = IV2SLS(
                dependent=sub_iv[var].astype(float),
                exog=exog_df,
                endog=sub_iv["used_skilllab"].astype(float),
                instruments=sub_iv[["T1","T2","T3"]].astype(float),
            ).fit(cov_type="robust")
            all_results += extract_iv(iv, var, label)
            coef = float(iv.params["used_skilllab"])
            pval = float(iv.pvalues["used_skilllab"])
            print(f"  IV (TOT)        | used_skilllab={coef:.4f}{stars(pval)}  p={pval:.3f}")
            del iv
        except Exception as e:
            print(f"  IV failed: {e}")

    del sub

# ── FIRST STAGE ───────────────────────────────────────────────
print("\n── First Stage: assignment -> take-up ───────────────────")
first_stage_rows = []
sub_fs = df.dropna(subset=["used_skilllab", "treat"]).copy()
try:
    fs = smf.ols("used_skilllab ~ T1 + T2 + T3 + C(prov_code)",
                 data=sub_fs).fit(cov_type="HC3")
    for term in ["T1","T2","T3"]:
        p = fs.pvalues.get(term, np.nan)
        print(f"  {term}: {fs.params.get(term,np.nan):.4f}{stars(p)}  p={p:.3f}")
        first_stage_rows.append({
            "term": term, "coef": round(fs.params.get(term, np.nan), 6),
            "se": round(fs.bse.get(term, np.nan), 6), "pvalue": round(p, 4),
            "stars": stars(p), "fstat": round(fs.fvalue, 2), "nobs": int(fs.nobs),
        })
    print(f"  F-stat: {fs.fvalue:.2f}  (need >10 for strong instrument)")
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
    print(f"\n── {spec} ──")
    s = results_df[results_df["spec"] == spec].copy()
    s["coef_stars"] = s["coef"].round(4).astype(str) + s["stars"]
    print(s.pivot_table(index="label", columns="term",
                        values="coef_stars", aggfunc="first").to_string())

results_df.to_csv(OUT_DIR / "summary_all_specs.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'summary_all_specs.csv'}")
print("\n*** p<0.01  ** p<0.05  * p<0.1")