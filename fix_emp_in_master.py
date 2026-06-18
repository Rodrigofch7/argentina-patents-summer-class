import pyreadstat, pandas as pd
from pathlib import Path

BASE = Path.home() / "argentina-rct/data"
OUT  = Path.home() / "argentina-rct/output"

# Load existing master
master = pd.read_csv(OUT / "master.csv", low_memory=False)
master["id_SSE"] = pd.to_numeric(master["id_SSE"], errors="coerce")

# Rebuild emp_latest the CORRECT way (collapse with max first)
emp, _ = pyreadstat.read_dta(str(BASE / "ministry/soc_security_emp_status.dta"))
pm = emp.groupby(["id_SSE","month_str"])["active_employment"].max().reset_index()
emp_latest = (
    pm.sort_values("month_str")
      .groupby("id_SSE")["active_employment"].last()
      .rename("emp_latest_fixed").reset_index()
)
emp_latest["id_SSE"] = pd.to_numeric(emp_latest["id_SSE"], errors="coerce")

# Replace the buggy column
master = master.drop(columns=["emp_latest"], errors="ignore")
master = master.merge(emp_latest, on="id_SSE", how="left")
master = master.rename(columns={"emp_latest_fixed": "emp_latest"})

master.to_csv(OUT / "master.csv", index=False)

# Rebuild analysis subset
phase_col = "treat_phase" if "treat_phase" in master.columns else "treat_phase_x"
analysis = master[master[phase_col].isin([1.0, 2.0])].copy()
analysis.to_csv(OUT / "master_analysis.csv", index=False)

print("Fixed emp_latest.")
print(f"  Corrected employment rate: {master['emp_latest'].mean():.4f}")
print(f"  master.csv and master_analysis.csv updated")
