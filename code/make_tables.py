"""
Argentina Labor Market RCT — Journal-Style Table Builder
World Bank / University of Chicago

Reads the three regression-results CSVs already produced by:
    regressions.py            -> output/regressions/all_regressions.csv
    regressions_extended.py   -> output/regressions/extended_regressions.csv
    survey_regressions.py     -> output/regressions/survey_regressions.csv

and writes three publication-style LaTeX tables (coefficient over standard
error, stars, controls/FE indicator rows, N and R^2 in a fit-statistics
block) to output/tables/*.tex, in a single consistent format.

This script does NOT re-run any regressions. It only reformats results
that are already saved on disk, so it's safe to re-run any time after
regressions.py / regressions_extended.py / survey_regressions.py.

Run:
    uv run make_tables.py
"""

import pandas as pd
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────
OUT_DIR   = Path.home() / "argentina-rct/output/regressions"
TBL_DIR   = Path.home() / "argentina-rct/output/tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

ARMS = ["T1", "T2", "T3"]
ARM_HEADERS = {
    "T1": "T1: CV",
    "T2": "T2: CV + Recs",
    "T3": "T3: CV + Recs + In-Demand",
}

SPEC_ORDER_MAIN = [
    ("1_ITT_province_FE",          "(1)"),
    ("2_ITT_controls_province_FE", "(2)"),
    ("3_ITT_controls_muni_FE",     "(3)"),
]
SPEC_ORDER_EXT = [
    ("1_province_FE",           "(1)"),
    ("2_controls_province_FE",  "(2)"),
    ("3_controls_muni_FE",      "(3)"),
]


def stars_str(s):
    return "" if pd.isna(s) else str(s)


def fmt_coef(coef, se, pval, decimals=3):
    """Coefficient with stars, stacked over (SE) in parentheses below."""
    st = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    c = f"{coef:.{decimals}f}{st}"
    s = f"({se:.{decimals}f})"
    return c, s


def pick_decimals(series, lo=2, hi=4):
    """Choose decimal places so coefficients are readable (avoids 0.000 rows)."""
    amax = series.abs().max()
    if amax == 0 or pd.isna(amax):
        return 3
    if amax >= 1000:
        return 1
    if amax >= 10:
        return 2
    if amax >= 1:
        return 3
    return 4


# ════════════════════════════════════════════════════════════════
# TABLE 1 — Main employment outcomes (regressions.py -> all_regressions.csv)
# One column per spec (1)-(3), rows = T1/T2/T3, stacked coef/SE.
# Plus a separate IV/TOT panel underneath using spec 4.
# ════════════════════════════════════════════════════════════════
def build_table1():
    df = pd.read_csv(OUT_DIR / "all_regressions.csv")
    sub = df[df["outcome"] == "emp_latest"].copy()

    decimals = 3  # binary 0/1 outcome -> 3 decimals is journal-standard

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{ITT Estimates: Formal Employment (Latest Month)}")
    lines.append(r"\label{tab:employment}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{l" + "c" * len(SPEC_ORDER_MAIN) + "}")
    lines.append(r"\toprule")
    lines.append(r" & " + " & ".join(f"({i+1})" for i in range(len(SPEC_ORDER_MAIN))) + r" \\")
    lines.append(r" & " + " & ".join([r"\shortstack{Province\\FE}", r"\shortstack{+\\Controls}", r"\shortstack{+ Muni.\\FE}"]) + r" \\")
    lines.append(r"\midrule")

    for arm in ARMS:
        coef_cells, se_cells = [], []
        for spec, _ in SPEC_ORDER_MAIN:
            row = sub[(sub["spec"] == spec) & (sub["term"] == arm)]
            if row.empty:
                coef_cells.append("")
                se_cells.append("")
                continue
            r = row.iloc[0]
            c, s = fmt_coef(r["coef"], r["se"], r["pvalue"], decimals)
            coef_cells.append(c)
            se_cells.append(s)
        lines.append(f"{ARM_HEADERS[arm]} & " + " & ".join(coef_cells) + r" \\")
        lines.append(" & " + " & ".join(se_cells) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\emph{Controls} & & Yes & Yes \\")
    lines.append(r"\emph{Province FE} & Yes & Yes & Yes \\")
    lines.append(r"\emph{Municipality FE} & & & Yes \\")
    lines.append(r"\midrule")

    n_row = []
    r2_row = []
    for spec, _ in SPEC_ORDER_MAIN:
        row = sub[(sub["spec"] == spec) & (sub["term"] == "T1")]
        if row.empty:
            n_row.append("")
            r2_row.append("")
        else:
            r = row.iloc[0]
            n_row.append(f"{int(r['nobs']):,}")
            r2_row.append(f"{r['r2']:.3f}")
    lines.append(r"$N$ & " + " & ".join(n_row) + r" \\")
    lines.append(r"$R^2$ & " + " & ".join(r2_row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item Robust (HC3) standard errors in parentheses. "
                 r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$. "
                 r"Dependent variable: 1 if formally employed in the most recent "
                 r"available month, defined at the person-month level by collapsing "
                 r"across employers with \texttt{max()} before taking the latest month. "
                 r"Sample: individuals with linked employment-panel data ($n=6{,}026$). "
                 r"None of the estimates reach conventional significance.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# TABLE 2 — Employment intensity & related outcomes
# (regressions_extended.py -> extended_regressions.csv)
# Single spec shown (controls + province FE), one row per outcome,
# stacked coef/SE, separate N per row since samples differ.
# ════════════════════════════════════════════════════════════════
def build_table2():
    df = pd.read_csv(OUT_DIR / "extended_regressions.csv")
    spec = "2_controls_province_FE"
    sub = df[df["spec"] == spec].copy()

    outcome_order = [
        ("months_employed", "Total months employed (0--19)"),
        ("n_employers",     "Distinct employers (job switching)"),
        ("allowed_companies_contact", "Allowed companies to contact"),
        ("receiving_benefit",         "Receiving social benefit"),
    ]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{ITT Estimates: Employment Intensity and Related Outcomes}")
    lines.append(r"\label{tab:intensity}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Outcome & T1 & T2 & T3 & $N$ \\")
    lines.append(r"\midrule")

    for var, label in outcome_order:
        row_sub = sub[sub["outcome"] == var]
        if row_sub.empty:
            continue
        decimals = 3
        coef_cells, se_cells = [], []
        n_val = None
        for arm in ARMS:
            r = row_sub[row_sub["term"] == arm]
            if r.empty:
                coef_cells.append("")
                se_cells.append("")
                continue
            r = r.iloc[0]
            n_val = int(r["nobs"])
            c, s = fmt_coef(r["coef"], r["se"], r["pvalue"], decimals)
            coef_cells.append(c)
            se_cells.append(s)
        n_str = f"{n_val:,}" if n_val is not None else ""
        lines.append(f"{label} & " + " & ".join(coef_cells) + f" & {n_str} \\\\")
        lines.append(" & " + " & ".join(se_cells) + r" & \\")

    lines.append(r"\midrule")
    lines.append(r"\emph{Controls} & \multicolumn{4}{c}{Yes} \\")
    lines.append(r"\emph{Province FE} & \multicolumn{4}{c}{Yes} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item Robust (HC3) standard errors in parentheses. "
                 r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$. "
                 r"All specifications include province fixed effects and the "
                 r"full control set (gender, age above median, secondary education, "
                 r"work experience, children, program). "
                 r"\emph{Total months employed} and \emph{Distinct employers} are "
                 r"defined over the 19-month administrative panel and limited to "
                 r"individuals with linked employment data ($n=6{,}026$); "
                 r"\emph{Allowed companies to contact} and \emph{Receiving social "
                 r"benefit} are defined for the full analysis sample ($n=103{,}153$).")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# TABLE 3 — Secondary survey outcomes
# (survey_regressions.py -> survey_regressions.csv)
# Single spec (controls + province FE), one row per outcome.
# ════════════════════════════════════════════════════════════════
def build_table3():
    df = pd.read_csv(OUT_DIR / "survey_regressions.csv")
    spec = "2_controls_province_FE"
    sub = df[df["spec"] == spec].copy()

    outcome_order = [
        ("P0_clean",  "Days worked last month"),
        ("P1_clean",  "Portal visits (2 weeks)"),
        ("P2_clean",  "Hours job searching / week"),
        ("P3_clean",  "Jobs applied last month"),
        ("P4_clean",  "Discovered a new skill (1=Yes)"),
        ("P6_clean",  "Considered a new career (1=Yes)"),
        ("P8_clean",  "Reservation wage (ARS)"),
        ("P9_clean",  "Job-finding confidence (1--10)"),
        ("P10_clean", "Received a job offer (1=Yes)"),
    ]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{ITT Estimates: Secondary Survey Outcomes}")
    lines.append(r"\label{tab:survey}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Outcome & T1 & T2 & T3 & $N$ \\")
    lines.append(r"\midrule")

    for var, label in outcome_order:
        row_sub = sub[sub["outcome"] == var]
        if row_sub.empty:
            continue
        if var == "P8_clean":
            decimals = 0  # ARS levels, large numbers
            scale = 1.0
        else:
            decimals = 3
            scale = 1.0
        coef_cells, se_cells = [], []
        n_val = None
        for arm in ARMS:
            r = row_sub[row_sub["term"] == arm]
            if r.empty:
                coef_cells.append("")
                se_cells.append("")
                continue
            r = r.iloc[0]
            n_val = int(r["nobs"])
            if var == "P8_clean":
                st = "***" if r["pvalue"] < 0.01 else "**" if r["pvalue"] < 0.05 else "*" if r["pvalue"] < 0.1 else ""
                c = f"{r['coef']:,.0f}{st}"
                s = f"({r['se']:,.0f})"
            else:
                c, s = fmt_coef(r["coef"] * scale, r["se"] * scale, r["pvalue"], decimals)
            coef_cells.append(c)
            se_cells.append(s)
        n_str = f"{n_val:,}" if n_val is not None else ""
        lines.append(f"{label} & " + " & ".join(coef_cells) + f" & {n_str} \\\\")
        lines.append(" & " + " & ".join(se_cells) + r" & \\")

    lines.append(r"\midrule")
    lines.append(r"\emph{Controls} & \multicolumn{4}{c}{Yes} \\")
    lines.append(r"\emph{Province FE} & \multicolumn{4}{c}{Yes} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item Robust (HC3) standard errors in parentheses. "
                 r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$. "
                 r"Midline survey ($n \approx 1{,}000$); sample sizes vary by item "
                 r"due to item-level non-response. All specifications include "
                 r"province fixed effects and the full control set.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── BUILD & SAVE ──────────────────────────────────────────────
t1 = build_table1()
t2 = build_table2()
t3 = build_table3()

(TBL_DIR / "table1_employment.tex").write_text(t1)
(TBL_DIR / "table2_intensity.tex").write_text(t2)
(TBL_DIR / "table3_survey.tex").write_text(t3)

print(f"Saved: {TBL_DIR / 'table1_employment.tex'}")
print(f"Saved: {TBL_DIR / 'table2_intensity.tex'}")
print(f"Saved: {TBL_DIR / 'table3_survey.tex'}")
print("\nAdd \\usepackage{threeparttable} and \\usepackage{booktabs} to your preamble.\n")
print("=" * 70)
print(t1)
print("=" * 70)
print(t2)
print("=" * 70)
print(t3)
