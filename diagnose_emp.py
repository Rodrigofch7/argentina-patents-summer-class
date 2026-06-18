import pyreadstat, pandas as pd, numpy as np
from pathlib import Path

BASE = Path.home() / "argentina-rct/data"
emp, _ = pyreadstat.read_dta(str(BASE / "ministry/soc_security_emp_status.dta"))

print(f"Raw emp rows: {len(emp):,}")
print(f"Unique people: {emp['id_SSE'].nunique():,}")
print(f"Months: {emp['month_str'].min()} to {emp['month_str'].max()}")

# ---- METHOD A: build_master.py approach ----
# sort by month descending, drop_duplicates keeps FIRST row per person
# (does NOT collapse multiple employers first -> may grab a row with emp=0)
methodA = (
    emp.sort_values("month_str", ascending=False)
       .drop_duplicates("id_SSE")
       [["id_SSE", "active_employment"]]
       .rename(columns={"active_employment": "emp_A"})
)

# ---- METHOD B: regressions_extended.py approach ----
# collapse to person-month with MAX first, then take last month
pm = emp.groupby(["id_SSE","month_str"])["active_employment"].max().reset_index()
methodB = (
    pm.sort_values("month_str")
      .groupby("id_SSE")["active_employment"].last()
      .rename("emp_B").reset_index()
)

# Compare
cmp = methodA.merge(methodB, on="id_SSE")
print(f"\nMethod A mean (latest month employed): {cmp['emp_A'].mean():.4f}")
print(f"Method B mean (latest month employed): {cmp['emp_B'].mean():.4f}")
print(f"Rows where they DIFFER: {(cmp['emp_A'] != cmp['emp_B']).sum():,}")

# Show an example person who differs
diff = cmp[cmp['emp_A'] != cmp['emp_B']]
if len(diff) > 0:
    pid = diff.iloc[0]['id_SSE']
    print(f"\nExample person {pid} (their last 2 months in raw data):")
    print(emp[emp['id_SSE']==pid].sort_values('month_str')
          [['id_SSE','month_str','Empleador','active_employment']].tail(4).to_string())

# What's the actual last month each method uses?
print(f"\nLast month in data: {pm['month_str'].max()}")
