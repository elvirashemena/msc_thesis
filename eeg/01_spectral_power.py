"""Extract global band power per recording into features/power_long.csv.

Expensive step, run once. Makes clean epochs for every preprocessed EO and EC
recording and computes the absolute and relative power per band, averaged
over all channels. Epochs are not matched across sessions because power is a
per-recording average.

Inputs
------
DATA_DIR
    Preprocessed recordings named {session}_{eeg_id}_raw_{condition}.fif.

Outputs
-------
features/power_long.csv
    eeg_id, session, condition, band, abs_power, rel_power and n_epochs.
features/power_skipped.csv
    Recordings with too little clean data.
"""

from pathlib import Path
import pandas as pd
import eeg_compute as ec

DATA_DIR = Path(r"C:\Users\jojos\OneDrive\Desktop\caint-depression-eeg-master\results\preprocessed\depression")
OUT = Path("features"); OUT.mkdir(exist_ok=True)
CONDITIONS = ("ec", "eo")

rows, skipped = [], []
for cond in CONDITIONS:
    for r in ec.index_files(DATA_DIR, cond).itertuples(index=False):
        raw = ec.load_raw(r.path, picks=None)
        ep = ec.make_epochs(raw, name=f"{r.session} {r.eeg_id} {cond}") if raw else None
        if ep is None:
            skipped.append({"file": Path(r.path).name})
            continue
        for bp in ec.band_powers_global(ep):
            rows.append({"eeg_id": r.eeg_id, "session": r.session, "condition": cond,
                         **bp, "n_epochs": len(ep)})

pd.DataFrame(rows).to_csv(OUT / "power_long.csv", index=False)
pd.DataFrame(skipped).to_csv(OUT / "power_skipped.csv", index=False)
print(f"power_long.csv: {len(rows)} rows, {len(skipped)} recordings skipped.")
