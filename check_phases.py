import pyreadstat
from pathlib import Path

rand, _ = pyreadstat.read_dta(str(Path.home() / "argentina-rct/data/worldbank/individ_randomization.dta"))
rand = rand.rename(columns={"id": "id_SSE"})

print("treat_phase values:")
print(rand["treat_phase"].value_counts().sort_index())

print("\ntreat_phase vs treat cross-tab:")
print(rand.groupby(["treat_phase", "treat"]).size().unstack())
