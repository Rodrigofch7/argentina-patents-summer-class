"""
Argentina Labor Market RCT — Panel Regression (Employment)
World Bank / University of Chicago

Uses full panel: 6,026 individuals × 19 months = ~114,000 person-month obs.

Specifications:
    1. Pooled OLS + month FE + province FE
    2. Pooled OLS + controls + month FE + province FE
    3. Treatment × Post interaction (DiD-style, no individual FE to save RAM)
       Only looks at treatment effect AFTER June 2025 rollout.

Run:
    uv run panel_regression.py
"""

import pandas as pd
import numpy as np
import pyreadstat
import statsmodels.formula.api as smf
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────
BASE    = Path.home() / "argentina-rct/data"
OUT_DIR = Path.home() / "argentina-rct/output/regressions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

def extract(model, spec, terms):
    rows = []
    for t in terms:
        if t in model.params:
            p = model.pvalues[t]
            rows.append({
                "spec":   spec, "term": t,
                "coef":   round(model.params[t], 6),
                "se":     round(model.bse[t], 6),
                "pvalue": round(p, 4),
                "nobs":   int(model.nobs),
                "r2":     round(model.rsquared, 4),
                "stars":  stars(p),
            })
    return rows

# ── LOAD & MERGE ──────────────────────────────────────────────
print("Loading data...")
emp, _ = pyreadstat.read_dta(str(BASE / "ministry/soc_security_emp_status.dta"))

# One obs per person-month (employed if active with ANY employer)
panel = (
    emp.groupby(["id_SSE", "month_str"])["active_employment"]
       .max().reset_index()
)
del emp

# Merge treatment
rand, _ = pyreadstat.read_dta(
    str(BASE / "worldbank/individ_randomization.dta"),
    usecols=["id", "treat", "provincia",
             "mujer", "edad_above_med", "educ_secondary",
             "work_experience", "has_children", "fomentar"]
)
rand = rand.rename(columns={"id": "id_SSE"})
rand["T1"] = (rand["treat"] == 1).astype(np.int8)
rand["T2"] = (rand["treat"] == 2).astype(np.int8)
rand["T3"] = (rand["treat"] == 3).astype(np.int8)
rand["prov_code"] = rand["provincia"].astype("category").cat.codes.astype(np.int16)
rand = rand.drop(columns=["provincia"])

panel = panel.merge(rand, on="id_SSE", how="left")
panel = panel.dropna(subset=["treat", "active_employment"])
del rand

# Post-treatment indicator: June 2025 onwards (Phase 2 rollout)
panel["post"] = (panel["month_str"] >= "2025-06").astype(np.int8)

print(f"  Panel: {len(panel):,} obs, {panel['id_SSE'].nunique():,} individuals")
print(f"  Pre:  {(panel['post']==0).sum():,} obs")
print(f"  Post: {(panel['post']==1).sum():,} obs")

# ── DESCRIPTIVE ───────────────────────────────────────────────
print("\n── Employment rate by treatment and period ──────────────")
desc = (
    panel.groupby(["treat", "post"])["active_employment"]
    .agg(["mean", "count"])
)
desc.index = desc.index.map(lambda x: (
    {0:"Control",1:"T1",2:"T2",3:"T3"}[int(x[0])],
    {0:"Pre",1:"Post"}[int(x[1])]
))
print(desc.round(4).to_string())

ctrl = " + ".join([c for c in [
    "mujer", "edad_above_med", "educ_secondary",
    "work_experience", "has_children", "fomentar"
] if c in panel.columns])

all_results = []

# ── SPEC 1: Pooled OLS ────────────────────────────────────────
print("\nSpec 1: Pooled OLS + month FE + province FE")
try:
    m = smf.ols(
        "active_employment ~ T1 + T2 + T3 + C(month_str) + C(prov_code)",
        data=panel
    ).fit(cov_type="HC3")
    all_results += extract(m, "1_pooled", ["T1","T2","T3"])
    for t in ["T1","T2","T3"]:
        p = m.pvalues.get(t, np.nan)
        print(f"  {t}: {m.params.get(t,np.nan):.4f}{stars(p)}  p={p:.3f}  N={int(m.nobs):,}")
    del m
except Exception as e:
    print(f"  Failed: {e}")

# ── SPEC 2: + Controls ────────────────────────────────────────
print("\nSpec 2: Pooled OLS + controls + month FE + province FE")
try:
    m = smf.ols(
        f"active_employment ~ T1 + T2 + T3 + {ctrl} + C(month_str) + C(prov_code)",
        data=panel
    ).fit(cov_type="HC3")
    all_results += extract(m, "2_pooled_controls", ["T1","T2","T3"])
    for t in ["T1","T2","T3"]:
        p = m.pvalues.get(t, np.nan)
        print(f"  {t}: {m.params.get(t,np.nan):.4f}{stars(p)}  p={p:.3f}  N={int(m.nobs):,}")
    del m
except Exception as e:
    print(f"  Failed: {e}")

# ── SPEC 3: DiD — Treatment × Post ───────────────────────────
# Asks: did employment increase MORE for treated vs control AFTER June 2025?
# T1:post, T2:post, T3:post are the causal DiD estimates.
# Uses province FE + month FE instead of individual FE to save RAM.
print("\nSpec 3: DiD — Treatment × Post + month FE + province FE + controls")
try:
    m = smf.ols(
        f"active_employment ~ T1*post + T2*post + T3*post + {ctrl} + C(month_str) + C(prov_code)",
        data=panel
    ).fit(cov_type="HC3")

    # Main effects
    main_terms = ["T1", "T2", "T3", "post"]
    # Interaction terms (DiD estimates — these are what we care about)
    int_terms = [k for k in m.params.index
                 if ":" in k and any(t in k for t in ["T1","T2","T3"])
                 and "C(" not in k]

    print("  Main effects:")
    for t in ["T1","T2","T3"]:
        p = m.pvalues.get(t, np.nan)
        print(f"    {t}: {m.params.get(t,np.nan):.4f}{stars(p)}  p={p:.3f}")

    print("  DiD estimates (treatment effect AFTER rollout):")
    for t in int_terms:
        p = m.pvalues.get(t, np.nan)
        print(f"    {t}: {m.params.get(t,np.nan):.4f}{stars(p)}  p={p:.3f}")

    print(f"  N={int(m.nobs):,}  R2={m.rsquared:.4f}")
    all_results += extract(m, "3_DiD_post", main_terms + int_terms)
    del m
except Exception as e:
    print(f"  Failed: {e}")

# ── SAVE ──────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "panel_regressions.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'panel_regressions.csv'}")
print("\n*** p<0.01  ** p<0.05  * p<0.1")