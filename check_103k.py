import pandas as pd
from pathlib import Path

master = pd.read_csv(Path.home() / "argentina-rct/output/master.csv", low_memory=False)

# treat_phase may be named treat_phase_x after the earlier merge
phase_col = "treat_phase" if "treat_phase" in master.columns else "treat_phase_x"
print(f"Using column: {phase_col}")

print("\nIndividuals by phase:")
print(master[phase_col].value_counts().sort_index())

# Phase 1 + 2 only (drop Phase 3)
analysis = master[master[phase_col].isin([1.0, 2.0])]
print(f"\nPhase 1 + 2 (drop Phase 3): {len(analysis):,} individuals")

# Balance check across treatment arms in the analysis sample
print("\nTreatment distribution in Phase 1+2 sample:")
print(analysis["treat"].value_counts().sort_index())
