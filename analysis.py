import pyreadstat
from pathlib import Path

BASE = Path.home() / "argentina-rct/data"

df, _ = pyreadstat.read_dta(str(BASE / "worldbank/individ_randomization.dta"))
print(f"Total individuals: {len(df):,}")
print(f"\nTreatment distribution:")
print(df["treat"].value_counts().sort_index())
