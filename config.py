from pathlib import Path

# ── Each person sets their own DATA_DIR here ──────────────────
# Point this to wherever your data folder is.
# Default assumes data/ is inside the project folder (recommended).
DATA_DIR = Path(__file__).parent / "data"

# Output folder (auto-created)
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

TREAT_MAP = {0: "Control", 1: "T1_CV", 2: "T2_Full", 3: "T3_InDemand"}
