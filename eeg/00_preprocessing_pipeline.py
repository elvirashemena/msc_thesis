# %%
'''
Author: Karahan Yılmazer yilmazerkarahan@gmail.com 
'''
import os
from pathlib import Path
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import torch
from utils import (
    add_bad_span_annotations,
    apply_notch_filter,
    drop_channels,
    fit_ica,
    get_artifact_indices,
    inspect_raw,
    load_config,
    load_data_as_raw,
    load_ica,
    save_ica,
    save_preprocessed,
    summarize_bad_spans,
    summarize_recording_duration,
    split_and_save_eo_ec,
)

RUN = "SUBJECT_SESSION"
INSPECT = True

# %%
# Load parameters and data
config = load_config(f"configs/config_{RUN}.yaml")
raw = load_data_as_raw(Path(config["dataset"]["raw_file"]), config=config)

# Drop unwanted channels
raw = drop_channels(raw, config["channels"].get("drop", []))

# Add bad span annotations
add_bad_span_annotations(raw, config.get("bad_spans", []))

inspect_raw(raw) if INSPECT else None

#%%
# Summarize bad spans and durations (for excel tracking)
#summarize_bad_spans(raw)

# Summarize recording duration and retrieve marker times
#total, clean_total, marker_time, eo, ec = summarize_recording_duration(raw)


# %%
# ICA PREPROCESSING
config = load_config(f"configs/config_{RUN}.yaml")

# Define bad channels
raw.info["bads"] = config["channels"]["bads"]
raw_ica = raw.copy()

# Initial filtering and rereferencing for Extended Infomax
raw_ica.filter(1, 100)
#raw_ica = apply_notch_filter(raw_ica, config["filtering"].get("notch"))

# Crop the beginning and the end of the recording
raw_ica.crop(config["cropping"]["tmin"], config["cropping"]["tmax"])

# Set CAR
raw_ica = raw_ica.set_eeg_reference(ref_channels="average")

inspect_raw(raw_ica) if INSPECT else None

# %%
# ICA PROCESSING
config = load_config(f"configs/config_{RUN}.yaml")

if os.path.exists(
    f"{config['output']['save_dir']}/{config['dataset']['name']}_ica.{config['output']['save_format']}"
):
    ica = load_ica(config)
else:
    ica = fit_ica(raw_ica, config["ica"])
    save_ica(ica, config)

if INSPECT:
    plt.close("all")
    ica.plot_components(nrows=4, ncols="auto")
    ica.plot_sources(raw_ica)
    plt.show()

# %%
#bad_ics = get_artifact_indices(
#    raw_ica, ica, config["ica"]["rejection_thr"], plot=INSPECT
#)
#print(f"Bad ICs: {bad_ics}")

# %%
config = load_config(f"configs/config_{RUN}.yaml")

# Read the Raw again if you have overwritten it
raw_final = raw.copy()

# Final preprocessing
raw_final.filter(config["filtering"]["highpass"], config["filtering"]["lowpass"])
# raw_final = apply_notch_filter(raw_final, config["filtering"].get("notch"))
raw_final.crop(config["cropping"]["tmin"], config["cropping"]["tmax"])
ica.exclude = config["ica"]["exclude_components"]
raw_final = ica.apply(raw_final)
raw_final = raw_final.set_eeg_reference(ref_channels="average")

inspect_raw(raw_final) if INSPECT else None

# %%
# Store bad channels before interpolation (for reference)
original_bads = raw_final.info["bads"].copy()
print(f"Bad channels before interpolation: {original_bads}")

# Interpolate bad channels
raw_final.interpolate_bads(reset_bads=True)
print(f"Bad channels after interpolation: {raw_final.info['bads']} (reset by MNE)")
print(f"Originally bad channels were: {original_bads}")
inspect_raw(raw_final) if INSPECT else None

# %%
# Final inspection and save
save_preprocessed(raw_final, config)

# %%
preproc_path = save_preprocessed(raw_final, config)
split_and_save_eo_ec(preproc_path)

# %%
