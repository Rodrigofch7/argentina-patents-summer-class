import pyreadstat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from config import DATA_DIR as BASE, OUT_DIR as OUTDIR, TREAT_MAP

OUTDIR = Path.home() / "argentina-rct/output"
OUTDIR.mkdir(exist_ok=True)

# ── COLORS ───────────────────────────────────────────────────
COLORS = {
    "Control":      "#888888",
    "T1_CV":        "#4c9be8",
    "T2_Full":      "#2e6da4",
    "T3_InDemand":  "#1a3a5c",
}
TREAT_MAP = {0: "Control", 1: "T1_CV", 2: "T2_Full", 3: "T3_InDemand"}

# ── LOAD ─────────────────────────────────────────────────────
print("Loading data...")
rand, _ = pyreadstat.read_dta(str(BASE / "worldbank/individ_randomization.dta"))
rand["treat_label"] = rand["treat"].map(TREAT_MAP)

survey = pd.read_excel(BASE / "opinaia/2025.08.06 Base.xlsx", sheet_name="Sheet1")
survey = survey.rename(columns={"Grupo": "treat_label"})
survey["treat_label"] = survey["treat_label"].str.strip()
survey_map = {
    "Control":                          "Control",
    "T1 - SkillLab CV":                 "T1_CV",
    "T2 - SkillLab Full":               "T2_Full",
    "T3 - SkillLab Full + In-Demand":   "T3_InDemand",
}
survey["treat_label"] = survey["treat_label"].map(survey_map)

emp, _ = pyreadstat.read_dta(str(BASE / "ministry/soc_security_emp_status.dta"))

print("Done. Generating plots...")

# ─────────────────────────────────────────────────────────────
# FIG 1 — Sample composition
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Sample Composition", fontsize=14, fontweight="bold", y=1.01)

# Gender
ax = axes[0]
vals = rand["mujer"].value_counts().sort_index()
ax.bar(["Male", "Female"], vals.values,
       color=["#4c9be8", "#e87c4c"], edgecolor="white", width=0.5)
ax.set_title("Gender")
ax.set_ylabel("Count")
for i, v in enumerate(vals.values):
    ax.text(i, v + 5000, f"{v/len(rand):.0%}", ha="center", fontsize=10)

# Education
ax = axes[1]
ed_labels = ["Primary only", "Secondary+"]
ed_vals   = [rand["educ_primary"].mean(), rand["educ_secondary"].mean()]
ax.bar(ed_labels, ed_vals, color=["#4c9be8", "#2e6da4"], edgecolor="white", width=0.5)
ax.set_title("Education Level")
ax.set_ylabel("Share")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
for i, v in enumerate(ed_vals):
    ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=10)

# Top provinces
ax = axes[2]
provs = rand["provincia"].value_counts().head(6)
ax.barh(provs.index[::-1], provs.values[::-1], color="#2e6da4", edgecolor="white")
ax.set_title("Top 6 Provinces")
ax.set_xlabel("Count")

plt.tight_layout()
fig.savefig(OUTDIR / "fig1_sample_composition.png", dpi=150, bbox_inches="tight")
print("  Saved fig1_sample_composition.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIG 2 — Treatment balance on key baseline vars
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Baseline Balance Across Treatment Arms", fontsize=14, fontweight="bold")

vars_labels = [
    ("mujer",           "Female (%)"),
    ("work_experience", "Work Experience (%)"),
    ("has_children",    "Has Children (%)"),
    ("edad_above_med",  "Above Median Age (%)"),
]

for ax, (var, label) in zip(axes, vars_labels):
    means = rand.groupby("treat_label")[var].mean()
    means = means.reindex(["Control", "T1_CV", "T2_Full", "T3_InDemand"])
    bars = ax.bar(means.index, means.values,
                  color=[COLORS[t] for t in means.index],
                  edgecolor="white", width=0.6)
    ax.set_title(label, fontsize=10)
    ax.set_ylim(0, means.max() * 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.01, f"{v:.0%}", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(OUTDIR / "fig2_balance.png", dpi=150, bbox_inches="tight")
print("  Saved fig2_balance.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIG 3 — Survey outcomes by treatment arm
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Midline Survey Outcomes by Treatment Arm", fontsize=14, fontweight="bold")

survey_vars = [
    ("P9",  "Job-finding Confidence (1-10)"),
    ("P0",  "Days Worked Last Month"),
    ("P3",  "Jobs Applied Last Month"),
]

order = ["Control", "T1_CV", "T2_Full", "T3_InDemand"]

for ax, (var, label) in zip(axes, survey_vars):
    survey[var] = pd.to_numeric(survey[var], errors="coerce")
    means = survey.groupby("treat_label")[var].mean().reindex(order)
    sems  = survey.groupby("treat_label")[var].sem().reindex(order)
    bars  = ax.bar(means.index, means.values,
                   color=[COLORS[t] for t in order],
                   edgecolor="white", width=0.6,
                   yerr=sems.values, capsize=4, error_kw={"linewidth": 1.2})
    ax.set_title(label, fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + sems.max() * 0.5 + 0.1,
                f"{v:.2f}", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(OUTDIR / "fig3_survey_outcomes.png", dpi=150, bbox_inches="tight")
print("  Saved fig3_survey_outcomes.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIG 4 — Monthly employment rate over time
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Monthly Employment Rate (Admin Data)", fontsize=14, fontweight="bold")

monthly = (
    emp.groupby("month_str")["active_employment"]
       .mean()
       .reset_index()
)
monthly["month_dt"] = pd.to_datetime(monthly["month_str"], format="%Y-%m")
monthly = monthly.sort_values("month_dt")

ax.plot(monthly["month_dt"], monthly["active_employment"],
        color="#2e6da4", linewidth=2, marker="o", markersize=4)
ax.fill_between(monthly["month_dt"], monthly["active_employment"],
                alpha=0.15, color="#2e6da4")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("Month")
ax.set_ylabel("Employment Rate")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
fig.savefig(OUTDIR / "fig4_employment_over_time.png", dpi=150, bbox_inches="tight")
print("  Saved fig4_employment_over_time.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIG 5 — Portal engagement funnel
# ─────────────────────────────────────────────────────────────
xl = pd.ExcelFile(BASE / "skilllab/28Oct2025_Raw_Data_Report_Argentina.xlsx")
portal = {s: xl.parse(s) for s in xl.sheet_names}

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("SkillLab Portal Engagement Funnel", fontsize=14, fontweight="bold")

funnel_steps = [
    ("Invited",       500000),
    ("Signed up",     len(portal["Sign-ups"])),
    ("Added skills",  len(portal["Skills"].drop_duplicates(portal["Skills"].columns[0]))),
    ("Explored careers", len(portal["Careers"].drop_duplicates(portal["Careers"].columns[0]))),
    ("Took courses",  len(portal["Courses"].drop_duplicates(portal["Courses"].columns[0]))),
    ("Exported CV",   len(portal["CVs"])),
]

labels = [s[0] for s in funnel_steps]
values = [s[1] for s in funnel_steps]
bar_colors = ["#1a3a5c", "#2e6da4", "#4c9be8", "#74b3e8", "#a8cff0", "#d0e8f8"]

bars = ax.barh(labels[::-1], values[::-1], color=bar_colors, edgecolor="white")
ax.set_xlabel("Number of Users")
for bar, v in zip(bars, values[::-1]):
    ax.text(bar.get_width() + values[0] * 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{v:,}", va="center", fontsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
fig.savefig(OUTDIR / "fig5_portal_funnel.png", dpi=150, bbox_inches="tight")
print("  Saved fig5_portal_funnel.png")
plt.close()

print(f"\nAll plots saved to {OUTDIR}")
