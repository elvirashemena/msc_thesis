"""Compute band-wise wPLI connectivity into features/connectivity_long.csv.

Expensive step, run once. For each region in eeg_compute.CONN_LEVELS and each
band, computes the mean wPLI over all electrode pairs. Epoch counts are
matched across sessions per subject because wPLI is biased upward with few
epochs, and the analysed seconds are stored to audit the length confound.
Theta is the primary band.

Inputs
------
DATA_DIR
    Preprocessed recordings named {session}_{eeg_id}_raw_{condition}.fif.

Outputs
-------
features/connectivity_long.csv
    eeg_id, session, condition, level, band, n_ch, wpli, sec and n_epochs.
"""

from pathlib import Path
import pandas as pd
import eeg_compute as ec

DATA_DIR = Path(r"C:\Users\jojos\OneDrive\Desktop\caint-depression-eeg-master\results\preprocessed\depression")
OUT = Path("features"); OUT.mkdir(exist_ok=True)
CONDITIONS = ("ec", "eo")


def report_region_sizes(sample_path):
    """Print how many electrodes each region resolves to in one recording."""
    print("  region electrode counts (from first recording):")
    for level, picks in ec.CONN_LEVELS.items():
        raw = ec.load_raw(sample_path, picks)
        n = 0 if raw is None else len(raw.ch_names)
        flag = "" if (picks is None or n >= ec.CONN_MIN_CH) else "  <-- TOO FEW, merge/drop"
        got = "all" if picks is None else n
        print(f"    {level:<15} {got} ch{flag}")
 
 
rows = []
for cond in CONDITIONS:
    files = ec.index_files(DATA_DIR, cond)
    report_region_sizes(files.iloc[0]["path"])
    for level, picks in ec.CONN_LEVELS.items():
        for eeg_id, sub in files.groupby("eeg_id"):
            ep, _ = ec.paired_epochs(sub, picks)
            if ep is None:
                print(f"  ! {eeg_id} [{cond}/{level}]: incomplete -> dropped")
                continue
            for session in ("Baseline", "Followup"):
                e = ep[session]
                wp = ec.band_wpli(e)
                for band, val in wp.items():
                    rows.append({"eeg_id": eeg_id, "session": session,
                                 "condition": cond, "level": level, "band": band,
                                 "n_ch": len(e.ch_names), "wpli": val,
                                 "sec": len(e) * ec.EPOCH_LEN_S, "n_epochs": len(e)})
 
df = pd.DataFrame(rows).dropna(subset=["wpli"])
df.to_csv(OUT / "connectivity_long.csv", index=False)
print(f"connectivity_long.csv: {len(df)} rows across {df['level'].nunique()} levels "
      f"x {df['band'].nunique()} bands (theta primary; beta/gamma EMG-suspect).")