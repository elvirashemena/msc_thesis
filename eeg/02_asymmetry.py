"""Compute frontal alpha asymmetry (FAA) into features/faa_long.csv.

Expensive step, run once. FAA is the mean log alpha power of the right
cluster minus that of the left cluster, with the log taken per electrode
(clusters and relative or absolute power are set in eeg_compute). Epoch
counts are matched across sessions per subject.

Inputs
------
DATA_DIR
    Preprocessed recordings named {session}_{eeg_id}_raw_{condition}.fif.

Outputs
-------
features/faa_long.csv
    eeg_id, session, condition, faa and n_epochs.
"""

from pathlib import Path
import pandas as pd
import eeg_compute as ec

DATA_DIR = Path(r"C:\Users\jojos\OneDrive\Desktop\caint-depression-eeg-master\results\preprocessed\depression")
OUT = Path("features"); OUT.mkdir(exist_ok=True)
CONDITIONS = ("ec", "eo")
PICKS = ec.FAA_LEFT + ec.FAA_RIGHT

rows = []
for cond in CONDITIONS:
    files = ec.index_files(DATA_DIR, cond)
    for eeg_id, sub in files.groupby("eeg_id"):
        ep, _ = ec.paired_epochs(sub, PICKS)
        if ep is None:
            print(f"  ! {eeg_id} [{cond}]: incomplete -> dropped")
            continue
        for session in ("Baseline", "Followup"):
            rows.append({"eeg_id": eeg_id, "session": session, "condition": cond,
                         "faa": ec.faa(ep[session]), "n_epochs": len(ep[session])})

df = pd.DataFrame(rows).dropna(subset=["faa"])
df.to_csv(OUT / "faa_long.csv", index=False)
print(f"faa_long.csv: {len(df)} rows "
      f"(absolute log alpha = {not ec.FAA_USE_RELATIVE}).")
