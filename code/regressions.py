"""
Argentina Labor Market RCT — Regressions
World Bank / University of Chicago

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

# ── LOAD — only columns we actually need ──────────────────────
print("Loading master dataset...")
needed_cols = [
    "treat", "provincia",
    "mujer", "edad_above_med", "educ_secondary",
    "work_experience", "has_children", "fomentar",
    "registered",
    # admin outcomes
    "enrolled_in_course", "applied_to_portal_job",
    "updated_cv", "emp_latest",
    # survey outcomes
    "P0", "P3", "P8", "P9", "P4", "P6",
]

df = pd.read_csv(MASTER, usecols=lambda c: c in needed_cols, low_memory=False)

df["treat"] = pd.to_numeric(df["treat"], errors="coerce")
df["T1"] = (df["treat"] == 1).astype(np.int8)
df["T2"] = (df["treat"] == 2).astype(np.int8)
df["T3"] = (df["treat"] == 3).astype(np.int8)

# Fix mixed-type survey columns
for col in ["P4", "P6", "P9", "P0", "P3", "P8"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Take-up
if "registered" in df.columns:
    df["used_skilllab"] = pd.to_numeric(
        df["registered"], errors="coerce").fillna(0).astype(np.int8)

# Encode provincia as integer category (memory efficient)
if "provincia" in df.columns:
    df["prov_code"] = df["provincia"].astype("category").cat.codes.astype(np.int16)
    del df["provincia"]

print(f"  Shape: {df.shape}")
print(f"  RAM usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"  Take-up: {df['used_skilllab'].sum():,} ({df['used_skilllab'].mean():.2%})")

# ── SETTINGS ──────────────────────────────────────────────────
controls = [c for c in [
    "mujer", "edad_above_med", "educ_secondary",
    "work_experience", "has_children", "fomentar"
] if c in df.columns]
ctrl_str = " + ".join(controls)
strata   = "prov_code" if "prov_code" in df.columns else None

def fe(col):
    return f" + C({col})" if col else ""

# ── OUTCOMES ──────────────────────────────────────────────────
admin_outcomes = {
    "enrolled_in_course":    "Enrolled in course (admin)",
    "applied_to_portal_job": "Applied to job via portal (admin)",
    "updated_cv":            "Updated CV (admin)",
    "emp_latest":            "Formally employed latest month (admin)",
}
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

# ── RUN ───────────────────────────────────────────────────────
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

    print(f"\n── {label} (n={len(sub):,}) ──")

    # ITT basic
    try:
        m1 = smf.ols(f"{var} ~ T1 + T2 + T3{fe(strata)}", data=sub).fit(cov_type="HC3")
        all_results += extract_ols(m1, var, label, "ITT basic")
        print(f"  ITT basic    | "
              f"T1={m1.params.get('T1',np.nan):.4f}{stars(m1.pvalues.get('T1',1))} "
              f"T2={m1.params.get('T2',np.nan):.4f}{stars(m1.pvalues.get('T2',1))} "
              f"T3={m1.params.get('T3',np.nan):.4f}{stars(m1.pvalues.get('T3',1))}")
        del m1
    except Exception as e:
        print(f"  ITT basic failed: {e}")

    # ITT with controls
    if ctrl_str:
        try:
            m2 = smf.ols(
                f"{var} ~ T1 + T2 + T3 + {ctrl_str}{fe(strata)}", data=sub
            ).fit(cov_type="HC3")
            all_results += extract_ols(m2, var, label, "ITT controls")
            print(f"  ITT controls | "
                  f"T1={m2.params.get('T1',np.nan):.4f}{stars(m2.pvalues.get('T1',1))} "
                  f"T2={m2.params.get('T2',np.nan):.4f}{stars(m2.pvalues.get('T2',1))} "
                  f"T3={m2.params.get('T3',np.nan):.4f}{stars(m2.pvalues.get('T3',1))}")
            del m2
        except Exception as e:
            print(f"  ITT controls failed: {e}")

    # IV 2SLS (TOT) — no strata dummies, use prov_code as numeric control instead
    if "used_skilllab" in sub.columns:
        try:
            sub_iv = sub.dropna(subset=["used_skilllab"]).reset_index(drop=True)

            # Exog: intercept + numeric prov_code (avoids huge dummy matrix)
            exog_cols = {"const": np.ones(len(sub_iv))}
            if strata and strata in sub_iv.columns:
                exog_cols[strata] = sub_iv[strata].astype(float).values
            exog_df = pd.DataFrame(exog_cols, index=sub_iv.index)

            iv_model = IV2SLS(
                dependent=sub_iv[var].astype(float),
                exog=exog_df,
                endog=sub_iv["used_skilllab"].astype(float),
                instruments=sub_iv[["T1", "T2", "T3"]].astype(float),
            ).fit(cov_type="robust")

            all_results += extract_iv(iv_model, var, label)
            coef = float(iv_model.params["used_skilllab"])
            pval = float(iv_model.pvalues["used_skilllab"])
            print(f"  IV (TOT)     | used_skilllab={coef:.4f}{stars(pval)}  p={pval:.3f}")
            del iv_model

        except Exception as e:
            print(f"  IV failed: {e}")

    del sub

# ── FIRST STAGE ───────────────────────────────────────────────
print("\n── First Stage ──────────────────────────────────────────")
first_stage_rows = []
if "used_skilllab" in df.columns:
    sub_fs = df.dropna(subset=["used_skilllab", "treat"]).copy()
    try:
        fs = smf.ols(
            f"used_skilllab ~ T1 + T2 + T3{fe(strata)}", data=sub_fs
        ).fit(cov_type="HC3")
        for term in ["T1", "T2", "T3"]:
            p = fs.pvalues.get(term, np.nan)
            print(f"  {term}: coef={fs.params.get(term,np.nan):.4f}{stars(p)}  p={p:.3f}")
            first_stage_rows.append({
                "term": term, "coef": round(fs.params.get(term, np.nan), 6),
                "se": round(fs.bse.get(term, np.nan), 6),
                "pvalue": round(p, 4), "stars": stars(p),
                "fstat": round(fs.fvalue, 2), "nobs": int(fs.nobs),
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
print("\n── Summary: ITT basic ───────────────────────────────────")
itt = results_df[results_df["spec"] == "ITT basic"].copy()
itt["coef_stars"] = itt["coef"].round(4).astype(str) + itt["stars"]
pivot = itt.pivot_table(index="label", columns="term", values="coef_stars", aggfunc="first")
print(pivot.to_string())
pivot.to_csv(OUT_DIR / "summary_itt.csv")
print(f"Saved: {OUT_DIR / 'summary_itt.csv'}")

print("\n── Summary: IV (TOT) ────────────────────────────────────")
iv_res = results_df[results_df["spec"] == "IV 2SLS (TOT)"].copy()
if len(iv_res) > 0:
    iv_res["coef_stars"] = iv_res["coef"].round(4).astype(str) + iv_res["stars"]
    print(iv_res[["label", "coef_stars", "se", "pvalue", "nobs"]].to_string(index=False))
    iv_res.to_csv(OUT_DIR / "summary_iv.csv", index=False)
    print(f"Saved: {OUT_DIR / 'summary_iv.csv'}")
else:
    print("  No IV results.")

print("\n*** p<0.01  ** p<0.05  * p<0.1")