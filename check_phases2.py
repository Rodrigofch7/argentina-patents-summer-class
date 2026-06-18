import pandas as pd
from pathlib import Path

master = pd.read_csv(Path.home() / "argentina-rct/output/master.csv", low_memory=False)
master["emp_latest"] = pd.to_numeric(master["emp_latest"], errors="coerce")

print("treat_phase value counts:")
print(master["treat_phase"].value_counts().sort_index())

print("\nEmployment records (non-null) by treat_phase:")
print(master.groupby("treat_phase")["emp_latest"].apply(lambda x: x.notna().sum()))

print("\nEmployment rate by treat_phase (among those with data):")
print(master.groupby("treat_phase")["emp_latest"].mean())

print("\nEmployment non-null by treat_phase AND treat:")
print(master.groupby(["treat_phase","treat"])["emp_latest"].apply(lambda x: x.notna().sum()).unstack())
