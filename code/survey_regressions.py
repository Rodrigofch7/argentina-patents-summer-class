"""
Argentina Labor Market RCT — Survey Outcome Regressions
World Bank / University of Chicago

Cleans and analyzes the midline survey outcomes (n ~ 1,000 respondents).
These are SECONDARY outcomes measuring job-search behavior, skill discovery,
and beliefs — the mechanisms the PAP cares about.

IMPORTANT: survey P-variables are stored as Spanish text ("Sí"/"No") or with
text suffixes (e.g. "10 - Muy posible"). They MUST be cleaned before analysis;
naively calling pd.to_numeric() silently deletes them.

Outcomes:
    P0  Days worked last month            (numeric 0-31)
    P1  Portal visits last 2 weeks        (numeric)
    P2  Hours job searching per week      (numeric)
    P3  Jobs applied to last month        (numeric)
    P4  Discovered a new skill?           (Sí/No -> 1/0)
    P6  Considered a new career?          (Sí/No -> 1/0)
    P8  Reservation wage (ARS)            (numeric)
    P9  Job-finding confidence 1-10       (text suffixes stripped)
    P10 Received a job offer?             (Sí/No -> 1/0)

Specifications (same controls + FE as admin regressions):
    1. ITT province FE
    2. ITT + controls + province FE
    3. ITT + controls + municipality FE

Run:
    uv run survey_regressions.py
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────
BASE    = Path.home() / "argentina-rct/data"
MASTER  = Path.home() / "argentina-rct/output/master_analysis.csv"
OUT_DIR = Path.home() / "argentina-rct/output/regressions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CLEANING FUNCTIONS ────────────────────────────────────────
def clean_yesno(series):
    """Convert Spanish Sí/No to 1/0."""
    m = {
        "sí": 1, "si": 1, "si, la usé": 1, "sí, la usé": 1,
        "no": 0, "no la usé": 0,
    }
    return (series.astype(str).str.strip().str.lower()
                  .map(m))

def clean_likert(series):
    """
    Strip text suffixes from Likert answers.
    '10 - Muy posible' -> 10, '1 - Imposible' -> 1, '5' -> 5
    """
    return pd.to_numeric(
        series.astype(str).str.extract(r"^\s*(\d+)")[0],
        errors="coerce"
    )

def clean_numeric(series):
    """Plain numeric coercion (for already-numeric vars)."""
    return pd.to_numeric(series, errors="coerce")

# ── LOAD SURVEY (raw, from Excel — master.csv may have coerced it) ──
print("Loading survey...")
survey = pd.read_excel(BASE / "opinaia/2025.08.06 Base.xlsx", sheet_name="Sheet1")
survey = survey.rename(columns={"Id_SSE": "id_SSE"})
survey["id_SSE"] = pd.to_numeric(survey["id_SSE"], errors="coerce")

# Clean each outcome with the correct method
survey["P0_clean"]  = clean_numeric(survey["P0"])    # days worked
survey["P1_clean"]  = clean_numeric(survey["P1"])    # portal visits
survey["P2_clean"]  = clean_numeric(survey["P2"])    # hours searching
survey["P3_clean"]  = clean_numeric(survey["P3"])    # jobs applied
survey["P4_clean"]  = clean_yesno(survey["P4"])      # discovered skill
survey["P6_clean"]  = clean_yesno(survey["P6"])      # considered career
survey["P8_clean"]  = clean_numeric(survey["P8"])    # reservation wage
survey["P9_clean"]  = clean_likert(survey["P9"])     # confidence 1-10
survey["P10_clean"] = clean_yesno(survey["P10"])     # received offer

clean_cols = ["id_SSE"] + [c for c in survey.columns if c.endswith("_clean")]

# Report cleaning success
print("\n── Cleaning check (non-null counts) ─────────────────────")
for c in clean_cols[1:]:
    print(f"  {c}: {survey[c].notna().sum()} non-null")

# ── MERGE WITH ANALYSIS SAMPLE (treatment + controls + strata) ──
print("\nMerging with analysis sample...")
admin = pd.read_csv(
    MASTER,
    usecols=lambda c: c in [
        "id_SSE", "treat", "provincia", "muni_id",
        "mujer", "edad_above_med", "educ_secondary",
        "work_experience", "has_children", "fomentar",
    ],
    low_memory=False
)
admin["id_SSE"] = pd.to_numeric(admin["id_SSE"], errors="coerce")
df = admin.merge(survey[clean_cols], on="id_SSE", how="inner")

df["treat"] = pd.to_numeric(df["treat"], errors="coerce")
df["T1"] = (df["treat"] == 1).astype(np.int8)
df["T2"] = (df["treat"] == 2).astype(np.int8)
df["T3"] = (df["treat"] == 3).astype(np.int8)
df["prov_code"] = df["provincia"].astype("category").cat.codes.astype(np.int16)
df["muni_id"]   = pd.to_numeric(df["muni_id"], errors="coerce")

print(f"  Survey respondents in analysis sample: {len(df):,}")

# ── SETTINGS ──────────────────────────────────────────────────
controls = [c for c in [
    "mujer", "edad_above_med", "educ_secondary",
    "work_experience", "has_children", "fomentar"
] if c in df.columns]
ctrl_str = " + ".join(controls)

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

# ── OUTCOMES ──────────────────────────────────────────────────
outcomes = {
    "P0_clean":  "Days worked last month",
    "P1_clean":  "Portal visits (2 weeks)",
    "P2_clean":  "Hours job searching/week",
    "P3_clean":  "Jobs applied last month",
    "P4_clean":  "Discovered a new skill (1=Yes)",
    "P6_clean":  "Considered a new career (1=Yes)",
    "P8_clean":  "Reservation wage (ARS)",
    "P9_clean":  "Job-finding confidence (1-10)",
    "P10_clean": "Received a job offer (1=Yes)",
}

# ── RUN ───────────────────────────────────────────────────────
all_results = []

for var, label in outcomes.items():
    sub = df.dropna(subset=[var, "treat"]).copy()
    if len(sub) == 0:
        print(f"  Skipping {label} — no obs")
        continue

    print(f"\n── {label} (n={len(sub):,}) ──")

    # SPEC 1: province FE
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + C(prov_code)",
                    data=sub).fit(cov_type="HC3")
        all_results += extract(m, var, label, "1_province_FE")
        print(f"  Province FE   | "
              f"T1={m.params.get('T1',np.nan):.3f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.3f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.3f}{stars(m.pvalues.get('T3',1))}")
        del m
    except Exception as e:
        print(f"  Province FE failed: {e}")

    # SPEC 2: + controls
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(prov_code)",
                    data=sub).fit(cov_type="HC3")
        all_results += extract(m, var, label, "2_controls_province_FE")
        print(f"  + Controls    | "
              f"T1={m.params.get('T1',np.nan):.3f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.3f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.3f}{stars(m.pvalues.get('T3',1))}")
        del m
    except Exception as e:
        print(f"  + Controls failed: {e}")

    # SPEC 3: + municipality FE
    try:
        m = smf.ols(f"{var} ~ T1 + T2 + T3 + {ctrl_str} + C(muni_id)",
                    data=sub.dropna(subset=["muni_id"])).fit(cov_type="HC3")
        all_results += extract(m, var, label, "3_controls_muni_FE")
        print(f"  + Muni FE     | "
              f"T1={m.params.get('T1',np.nan):.3f}{stars(m.pvalues.get('T1',1))} "
              f"T2={m.params.get('T2',np.nan):.3f}{stars(m.pvalues.get('T2',1))} "
              f"T3={m.params.get('T3',np.nan):.3f}{stars(m.pvalues.get('T3',1))}")
        del m
    except Exception as e:
        print(f"  + Muni FE failed: {e}")

# ── SAVE ──────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "survey_regressions.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'survey_regressions.csv'}")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n── Summary: controls + province FE ──────────────────────")
s = results_df[results_df["spec"] == "2_controls_province_FE"].copy()
s["coef_stars"] = s["coef"].astype(str) + s["stars"]
print(s.pivot_table(index="label", columns="term",
                    values="coef_stars", aggfunc="first").to_string())
print("\n*** p<0.01  ** p<0.05  * p<0.1")
