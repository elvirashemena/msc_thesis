"""Convert raw questionnaire scores to one wide row per participant.

Reads long-format HAM-D17 and BDI-II scores (timepoint 1 = pre, 3 = post) and
computes, per scale, the pre and post scores, the raw change, the percent
change and the responder status (percent change <= -50).

Inputs
------
questionnaire_raw.xlsx
    Columns ID, Timepoint, HAMD17 and BDI.

Outputs
-------
questionnaire_converted.csv, questionnaire_converted.xlsx
    One row per ID with {scale}_pre, {scale}_post, delta_{scale},
    {scale}_percentage and responder_{scale}.

Notes
-----
- Percent change is (post - pre) / pre * 100, so negative values mean
  improvement.
- The CSV uses ';' as separator.
"""

import pandas as pd
import os

INPUT_PATH = r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\questionnaire"
INPUT_FILENAME = "questionnaire_raw.xlsx"
OUTPUT_PATH = r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\questionnaire"
OUTPUT_FILENAME_CSV = "questionnaire_converted.csv"
OUTPUT_FILENAME_XLSX = "questionnaire_converted.xlsx"

QUESTIONNAIRES = ["HAMD17", "BDI"]

TIMEPOINT_MAP = {1: "pre", 3: "post"}
RESPONDER_THRESHOLD = -50

filepath = os.path.join(INPUT_PATH, INPUT_FILENAME)
df_raw = pd.read_excel(filepath)
print(f"Loaded {len(df_raw)} rows, columns: {list(df_raw.columns)}")

df = df_raw.copy()
df["ID"] = "ID_" + df["ID"].astype(str)
df["Timepoint"] = df["Timepoint"].map(TIMEPOINT_MAP)

df = df.dropna(subset=["Timepoint"])

df_pre = df[df["Timepoint"] == "pre"].set_index("ID")
df_post = df[df["Timepoint"] == "post"].set_index("ID")

all_ids = sorted(set(df_pre.index) | set(df_post.index))

rows = []
for pid in all_ids:
    row = {"ID": pid}

    for q in QUESTIONNAIRES:
        pre = df_pre.loc[pid, q] if pid in df_pre.index else None
        post = df_post.loc[pid, q] if pid in df_post.index else None

        pre = float(pre) if pd.notna(pre) else None
        post = float(post) if pd.notna(post) else None

        row[f"{q}_pre"] = pre
        row[f"{q}_post"] = post

        if pre is not None and post is not None:
            row[f"delta_{q}"] = round(post - pre, 2)
            if pre != 0:
                row[f"{q}_percentage"] = round((post - pre) / pre * 100, 2)
            else:
                row[f"{q}_percentage"] = None
        else:
            row[f"delta_{q}"] = None
            row[f"{q}_percentage"] = None

        pct = row[f"{q}_percentage"]
        if pct is not None:
            row[f"responder_{q}"] = "responder" if pct <= RESPONDER_THRESHOLD else "non-responder"
        else:
            row[f"responder_{q}"] = None

    rows.append(row)

df_result = pd.DataFrame(rows)

csv_path = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME_CSV)
xlsx_path = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME_XLSX)

df_result.to_csv(csv_path, index=False, sep=";")
df_result.to_excel(xlsx_path, index=False)

print(f"\nSaved CSV:   {csv_path}")
print(f"Saved Excel: {xlsx_path}")
print(df_result.to_string(index=False))

print(f"Unique participants: {df_result['ID'].nunique()}")

for q in QUESTIONNAIRES:
    missing_pre = df_result[f"{q}_pre"].isna().sum()
    missing_post = df_result[f"{q}_post"].isna().sum()
    print(f"\n{q}:")
    print(f"  Missing pre: {missing_pre}, missing post: {missing_post}")

    col = f"responder_{q}"
    n_valid = df_result[col].notna().sum()
    n_resp = (df_result[col] == "responder").sum()
    pct = round(n_resp / n_valid * 100, 1) if n_valid > 0 else 0
    print(f"  Responders: {n_resp} / {n_valid} ({pct}%)")
