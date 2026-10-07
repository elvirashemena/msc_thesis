from pathlib import Path

import matplotlib.pyplot as plt
import mne
import pyxdf
import yaml
from matplotlib.colors import LinearSegmentedColormap
from mne.preprocessing import ICA
from mne_icalabel import label_components
import numpy as np


def load_config(config_file):
    """Load preprocessing parameters from YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_data_as_raw(data_file, config=None):
    """
    Load XDF or EEG file and convert to MNE Raw object with montage.
    Supports multiple EEG file formats.

    Parameters:
    -----------
    data_file : str or Path
        Path to data file (.xdf, .eeg, .set, .vhdr, .bdf, .edf, etc.)
    config : dict or None
        Configuration dictionary containing montage settings

    Returns:
    --------
    mne.io.Raw : Raw object with data and optionally montage
    """

    data_file = Path(data_file)
    print(f"Loading {data_file}")

    if config and "montage" in config:
        montage_file = config["montage"].get("montage_file")
        head_size = config["montage"].get("head_size", 0.075)

    # Determine file type and load accordingly
    if data_file.suffix.lower() == ".xdf":
        # Load XDF data
        streams, header = pyxdf.load_xdf(data_file)
        data = streams[0]["time_series"].T
        fs = streams[0]["info"]["nominal_srate"][0]

        # For XDF files, we need a montage file
        if not montage_file:
            raise ValueError(
                "XDF files require a montage file to be specified in config"
            )

        # Montage and info
        montage = mne.channels.read_custom_montage(montage_file, head_size=head_size)
        channels = montage.ch_names[2:]  # Exclude REF and GND
        info = mne.create_info(ch_names=channels, sfreq=fs, ch_types="eeg")
        info.set_montage(montage)

        raw = mne.io.RawArray(data[: len(channels)], info)
        print(f"Loaded XDF data with shape: {data.shape} and sampling rate: {fs} Hz")

    elif data_file.suffix.lower() == ".vhdr":
        # Load BrainVision data
        raw = mne.io.read_raw_brainvision(data_file, preload=True)
        if config["channels"].get("drop", None):
            raw.drop_channels(config["channels"].get("drop", None))
        print(
            f"Loaded BrainVision data with {len(raw.ch_names)} channels and sampling rate: {raw.info['sfreq']} Hz"
        )

        montage = mne.channels.make_standard_montage("easycap-M1")
        raw.set_montage(montage)

    else:
        raise ValueError(
            f"Unsupported file format: {data_file.suffix}. "
            f"Supported formats: .xdf, .vhdr"
        )

    return raw


def apply_notch_filter(raw, notch_freqs):
    """
    Apply notch filter to Raw object if notch frequencies are specified.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object to apply notch filter to
    notch_freqs : float, list, or None
        Notch frequencies to filter. Can be single frequency, list of frequencies, or None

    Returns:
    --------
    mne.io.Raw : Raw object with notch filter applied
    """
    if notch_freqs is None:
        print("No notch filtering specified")
        return raw

    # Convert single frequency to list for consistent handling
    if isinstance(notch_freqs, (int, float)):
        notch_freqs = [notch_freqs]

    print(f"Applying notch filter at {notch_freqs} Hz")
    raw.notch_filter(notch_freqs)

    return raw

def inspect_raw(raw, modify_original=False):
    """
    Inspect raw data with plots.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object to inspect.
    modify_original : bool
        If False (default), uses a copy so interactive plot doesn't modify
        the original. If True, uses original raw (allows marking bads).
    """
    plt.close("all")

    raw_to_plot = raw if modify_original else raw.copy()
    raw_to_plot.plot(scalings="auto")
    raw.compute_psd().plot()  # PSD doesn't modify raw
    plt.show()


def drop_channels(raw, channels_to_drop):
    """
    Drop specified channels from Raw object.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object to drop channels from
    channels_to_drop : list
        List of channel names to drop

    Returns:
    --------
    mne.io.Raw : Raw object with channels dropped
    """
    if not channels_to_drop:
        return raw

    # Find channels that actually exist in the data
    existing_channels_to_drop = [ch for ch in channels_to_drop if ch in raw.ch_names]

    if existing_channels_to_drop:
        print(f"Dropping channels: {existing_channels_to_drop}")
        raw = raw.drop_channels(existing_channels_to_drop)
        print(f"Remaining channels: {len(raw.ch_names)}")
    else:
        print(
            f"None of the specified channels to drop {channels_to_drop} were found in the data"
        )

    return raw


def get_artifact_indices(raw, ica, threshold=0.9, plot=True, verbose=True):
    """
    Get indices of components that are not 'other' or 'brain' and have
    probability higher than threshold.

    Parameters:
    -----------
    ic_labels : dict
        Dictionary with 'y_pred_proba' and 'labels' keys
    threshold : float
        Probability threshold (default 0.9)

    Returns:
    --------
    list : indices of components meeting the criteria
    """
    ic_labels = label_components(raw, ica, method="iclabel")

    probabilities = ic_labels["y_pred_proba"]
    labels = ic_labels["labels"]

    # Find indices where label is not 'other' or 'brain' AND probability > threshold
    indices = []
    for i, (prob, label) in enumerate(zip(probabilities, labels)):
        if label not in ["other", "brain"] and prob > threshold:
            indices.append(i)

    if verbose:
        if indices:
            print("Details of identified components:")
            for idx in indices:
                print(
                    f"Index {idx}: {ic_labels['labels'][idx]}"
                    f" (probability: {ic_labels['y_pred_proba'][idx]:.4f})"
                )
        else:
            print("No components found matching the criteria.")

        print()

        print("All non-brain/other components (regardless of probability):")
        for i, (prob, label) in enumerate(
            zip(ic_labels["y_pred_proba"], ic_labels["labels"])
        ):
            if label not in ["other", "brain"]:
                print(f"Index {i}: {label} (probability: {prob:.4f})")
        print()

    if plot:
        for idx in indices:
            ica.plot_properties(raw, picks=[idx])

    return indices


def fit_ica(raw, ica_cfg):
    """
    Apply ICA pipeline: fit, plot, and remove artifacts.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object to apply ICA to
    ica_cfg : dict
        ICA configuration from params
    params : dict, optional
        Full parameters dictionary (for updating YAML if ICLabel is used)
    params_file : str
        Path to params YAML file (default: "params.yaml")

    Returns:
    --------
    mne.preprocessing.ICA : Fitted ICA object
    """
    # Create and fit ICA
    ica = ICA(
        n_components=ica_cfg["n_components"],
        random_state=ica_cfg["random_state"],
        method=ica_cfg["method"],
        max_iter=ica_cfg.get("max_iter", "auto"),
        fit_params=ica_cfg.get("fit_params", None),
    )
    ica.fit(raw, reject_by_annotation=ica_cfg.get("reject_by_annotation", True))

    # Plot components
    ica.plot_components()
    ica.plot_sources(raw)

    # Remove bad components
    bad_ics = ica_cfg["exclude_components"]
    if bad_ics is None:
        # Perform ICLabel
        artifact_indices = get_artifact_indices(raw, ica, ica_cfg["rejection_thr"])
        ica.exclude = artifact_indices
    else:
    # Use pre-defined bad IC labels
        ica.exclude = bad_ics

    return ica


def save_ica(ica, params, suffix="ica"):
    """
    Save fitted ICA object.

    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object to save
    params : dict
        Parameters dictionary containing output configuration
    suffix : str
        Suffix for filename (default: "ica")

    Returns:
    --------
    Path : Path to saved ICA file
    """
    out_cfg = params["output"]
    save_dir = Path(out_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        save_dir
        / f"{params['dataset']['name']}_{suffix}.{params['output']['save_format']}"
    )
    ica.save(out_path, overwrite=True)
    print(f"Saved ICA object to {out_path}")

    return out_path


def load_ica(params, suffix="ica"):
    """
    Load a previously saved ICA object.

    Parameters:
    -----------
    params : dict
        Parameters dictionary containing output configuration
    suffix : str
        Suffix for filename (default: "ica")

    Returns:
    --------
    mne.preprocessing.ICA : Loaded ICA object
    """
    out_cfg = params["output"]
    save_dir = Path(out_cfg["save_dir"])

    ica_path = (
        save_dir
        / f"{params['dataset']['name']}_{suffix}.{params['output']['save_format']}"
    )

    if not ica_path.exists():
        raise FileNotFoundError(f"ICA file not found: {ica_path}")

    ica = mne.preprocessing.read_ica(ica_path)
    print(f"Loaded ICA object from {ica_path}")

    return ica


def save_preprocessed(raw, params, suffix="raw"):
    """
    Save preprocessed data.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object to save
    params : dict
        Parameters dictionary containing output configuration
    suffix : str
        Suffix for filename (default: "raw")
    """
    out_cfg = params["output"]
    save_dir = Path(out_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        save_dir / f"{params['dataset']['name']}_{suffix}.{out_cfg['save_format']}"
    )
    raw.save(out_path, overwrite=True)
    print(f"Saved preprocessed data to {out_path}")

    return out_path


def get_colormap(name="parula", as_colors=False):
    """
    Get a colormap by name. First checks palettable library, then falls back to custom colormaps.

    Parameters:
    -----------
    name : str
        Name of the colormap to retrieve

    Returns:
    --------
    matplotlib colormap object (mpl.cm.Colormap or mpl.colors.Colormap if as_colors=True)
    """
    import inspect

    import matplotlib.colors as mcolors
    import numpy as np

    # First, try to find the colormap in palettable
    try:
        import palettable

        # Search through all palettable modules and submodules
        for module_name in dir(palettable):
            if module_name.startswith("_"):
                continue

            module = getattr(palettable, module_name)
            if not inspect.ismodule(module):
                continue

            # Check submodules (e.g., sequential, diverging, qualitative)
            for submodule_name in dir(module):
                if submodule_name.startswith("_"):
                    continue

                submodule = getattr(module, submodule_name)
                if not inspect.ismodule(submodule):
                    continue

                # Look for the colormap in this submodule
                for attr_name in dir(submodule):
                    if attr_name.lower() == name.lower() and hasattr(
                        getattr(submodule, attr_name), "mpl_colormap"
                    ):
                        colormap_obj = getattr(submodule, attr_name)
                        print(
                            f"Found '{name}' in palettable.{module_name}.{submodule_name}"
                        )
                        if as_colors and hasattr(colormap_obj, "mpl_colors"):
                            cmap = colormap_obj.mpl_colors
                        else:
                            cmap = colormap_obj.mpl_colormap
                        return cmap

    except ImportError:
        print("Palettable not available, using custom colormaps only")
    except Exception as e:
        print(f"Error searching palettable: {e}")

    # Fall back to custom colormaps if not found in palettable
    if name.lower() == "parula":
        cm_data = [
            [0.2422, 0.1504, 0.6603],
            [0.2444, 0.1534, 0.6728],
            [0.2464, 0.1569, 0.6847],
            [0.2484, 0.1607, 0.6961],
            [0.2503, 0.1648, 0.7071],
            [0.2522, 0.1689, 0.7179],
            [0.254, 0.1732, 0.7286],
            [0.2558, 0.1773, 0.7393],
            [0.2576, 0.1814, 0.7501],
            [0.2594, 0.1854, 0.761],
            [0.2611, 0.1893, 0.7719],
            [0.2628, 0.1932, 0.7828],
            [0.2645, 0.1972, 0.7937],
            [0.2661, 0.2011, 0.8043],
            [0.2676, 0.2052, 0.8148],
            [0.2691, 0.2094, 0.8249],
            [0.2704, 0.2138, 0.8346],
            [0.2717, 0.2184, 0.8439],
            [0.2729, 0.2231, 0.8528],
            [0.274, 0.228, 0.8612],
            [0.2749, 0.233, 0.8692],
            [0.2758, 0.2382, 0.8767],
            [0.2766, 0.2435, 0.884],
            [0.2774, 0.2489, 0.8908],
            [0.2781, 0.2543, 0.8973],
            [0.2788, 0.2598, 0.9035],
            [0.2794, 0.2653, 0.9094],
            [0.2798, 0.2708, 0.915],
            [0.2802, 0.2764, 0.9204],
            [0.2806, 0.2819, 0.9255],
            [0.2809, 0.2875, 0.9305],
            [0.2811, 0.293, 0.9352],
            [0.2813, 0.2985, 0.9397],
            [0.2814, 0.304, 0.9441],
            [0.2814, 0.3095, 0.9483],
            [0.2813, 0.315, 0.9524],
            [0.2811, 0.3204, 0.9563],
            [0.2809, 0.3259, 0.96],
            [0.2807, 0.3313, 0.9636],
            [0.2803, 0.3367, 0.967],
            [0.2798, 0.3421, 0.9702],
            [0.2791, 0.3475, 0.9733],
            [0.2784, 0.3529, 0.9763],
            [0.2776, 0.3583, 0.9791],
            [0.2766, 0.3638, 0.9817],
            [0.2754, 0.3693, 0.984],
            [0.2741, 0.3748, 0.9862],
            [0.2726, 0.3804, 0.9881],
            [0.271, 0.386, 0.9898],
            [0.2691, 0.3916, 0.9912],
            [0.267, 0.3973, 0.9924],
            [0.2647, 0.403, 0.9935],
            [0.2621, 0.4088, 0.9946],
            [0.2591, 0.4145, 0.9955],
            [0.2556, 0.4203, 0.9965],
            [0.2517, 0.4261, 0.9974],
            [0.2473, 0.4319, 0.9983],
            [0.2424, 0.4378, 0.9991],
            [0.2369, 0.4437, 0.9996],
            [0.2311, 0.4497, 0.9995],
            [0.225, 0.4559, 0.9985],
            [0.2189, 0.462, 0.9968],
            [0.2128, 0.4682, 0.9948],
            [0.2066, 0.4743, 0.9926],
            [0.2006, 0.4803, 0.9906],
            [0.195, 0.4861, 0.9887],
            [0.1903, 0.4919, 0.9867],
            [0.1869, 0.4975, 0.9844],
            [0.1847, 0.503, 0.9819],
            [0.1831, 0.5084, 0.9793],
            [0.1818, 0.5138, 0.9766],
            [0.1806, 0.5191, 0.9738],
            [0.1795, 0.5244, 0.9709],
            [0.1785, 0.5296, 0.9677],
            [0.1778, 0.5349, 0.9641],
            [0.1773, 0.5401, 0.9602],
            [0.1768, 0.5452, 0.956],
            [0.1764, 0.5504, 0.9516],
            [0.1755, 0.5554, 0.9473],
            [0.174, 0.5605, 0.9432],
            [0.1716, 0.5655, 0.9393],
            [0.1686, 0.5705, 0.9357],
            [0.1649, 0.5755, 0.9323],
            [0.161, 0.5805, 0.9289],
            [0.1573, 0.5854, 0.9254],
            [0.154, 0.5902, 0.9218],
            [0.1513, 0.595, 0.9182],
            [0.1492, 0.5997, 0.9147],
            [0.1475, 0.6043, 0.9113],
            [0.1461, 0.6089, 0.908],
            [0.1446, 0.6135, 0.905],
            [0.1429, 0.618, 0.9022],
            [0.1408, 0.6226, 0.8998],
            [0.1383, 0.6272, 0.8975],
            [0.1354, 0.6317, 0.8953],
            [0.1321, 0.6363, 0.8932],
            [0.1288, 0.6408, 0.891],
            [0.1253, 0.6453, 0.8887],
            [0.1219, 0.6497, 0.8862],
            [0.1185, 0.6541, 0.8834],
            [0.1152, 0.6584, 0.8804],
            [0.1119, 0.6627, 0.877],
            [0.1085, 0.6669, 0.8734],
            [0.1048, 0.671, 0.8695],
            [0.1009, 0.675, 0.8653],
            [0.0964, 0.6789, 0.8609],
            [0.0914, 0.6828, 0.8562],
            [0.0855, 0.6865, 0.8513],
            [0.0789, 0.6902, 0.8462],
            [0.0713, 0.6938, 0.8409],
            [0.0628, 0.6972, 0.8355],
            [0.0535, 0.7006, 0.8299],
            [0.0433, 0.7039, 0.8242],
            [0.0328, 0.7071, 0.8183],
            [0.0234, 0.7103, 0.8124],
            [0.0155, 0.7133, 0.8064],
            [0.0091, 0.7163, 0.8003],
            [0.0046, 0.7192, 0.7941],
            [0.0019, 0.722, 0.7878],
            [0.0009, 0.7248, 0.7815],
            [0.0018, 0.7275, 0.7752],
            [0.0046, 0.7301, 0.7688],
            [0.0094, 0.7327, 0.7623],
            [0.0162, 0.7352, 0.7558],
            [0.0253, 0.7376, 0.7492],
            [0.0369, 0.74, 0.7426],
            [0.0504, 0.7423, 0.7359],
            [0.0638, 0.7446, 0.7292],
            [0.077, 0.7468, 0.7224],
            [0.0899, 0.7489, 0.7156],
            [0.1023, 0.751, 0.7088],
            [0.1141, 0.7531, 0.7019],
            [0.1252, 0.7552, 0.695],
            [0.1354, 0.7572, 0.6881],
            [0.1448, 0.7593, 0.6812],
            [0.1532, 0.7614, 0.6741],
            [0.1609, 0.7635, 0.6671],
            [0.1678, 0.7656, 0.6599],
            [0.1741, 0.7678, 0.6527],
            [0.1799, 0.7699, 0.6454],
            [0.1853, 0.7721, 0.6379],
            [0.1905, 0.7743, 0.6303],
            [0.1954, 0.7765, 0.6225],
            [0.2003, 0.7787, 0.6146],
            [0.2061, 0.7808, 0.6065],
            [0.2118, 0.7828, 0.5983],
            [0.2178, 0.7849, 0.5899],
            [0.2244, 0.7869, 0.5813],
            [0.2318, 0.7887, 0.5725],
            [0.2401, 0.7905, 0.5636],
            [0.2491, 0.7922, 0.5546],
            [0.2589, 0.7937, 0.5454],
            [0.2695, 0.7951, 0.536],
            [0.2809, 0.7964, 0.5266],
            [0.2929, 0.7975, 0.517],
            [0.3052, 0.7985, 0.5074],
            [0.3176, 0.7994, 0.4975],
            [0.3301, 0.8002, 0.4876],
            [0.3424, 0.8009, 0.4774],
            [0.3548, 0.8016, 0.4669],
            [0.3671, 0.8021, 0.4563],
            [0.3795, 0.8026, 0.4454],
            [0.3921, 0.8029, 0.4344],
            [0.405, 0.8031, 0.4233],
            [0.4184, 0.803, 0.4122],
            [0.4322, 0.8028, 0.4013],
            [0.4463, 0.8024, 0.3904],
            [0.4608, 0.8018, 0.3797],
            [0.4753, 0.8011, 0.3691],
            [0.4899, 0.8002, 0.3586],
            [0.5044, 0.7993, 0.348],
            [0.5187, 0.7982, 0.3374],
            [0.5329, 0.797, 0.3267],
            [0.547, 0.7957, 0.3159],
            [0.5609, 0.7943, 0.305],
            [0.5748, 0.7929, 0.2941],
            [0.5886, 0.7913, 0.2833],
            [0.6024, 0.7896, 0.2726],
            [0.6161, 0.7878, 0.2622],
            [0.6297, 0.7859, 0.2521],
            [0.6433, 0.7839, 0.2423],
            [0.6567, 0.7818, 0.2329],
            [0.6701, 0.7796, 0.2239],
            [0.6833, 0.7773, 0.2155],
            [0.6963, 0.775, 0.2075],
            [0.7091, 0.7727, 0.1998],
            [0.7218, 0.7703, 0.1924],
            [0.7344, 0.7679, 0.1852],
            [0.7468, 0.7654, 0.1782],
            [0.759, 0.7629, 0.1717],
            [0.771, 0.7604, 0.1658],
            [0.7829, 0.7579, 0.1608],
            [0.7945, 0.7554, 0.157],
            [0.806, 0.7529, 0.1546],
            [0.8172, 0.7505, 0.1535],
            [0.8281, 0.7481, 0.1536],
            [0.8389, 0.7457, 0.1546],
            [0.8495, 0.7435, 0.1564],
            [0.86, 0.7413, 0.1587],
            [0.8703, 0.7392, 0.1615],
            [0.8804, 0.7372, 0.165],
            [0.8903, 0.7353, 0.1695],
            [0.9, 0.7336, 0.1749],
            [0.9093, 0.7321, 0.1815],
            [0.9184, 0.7308, 0.189],
            [0.9272, 0.7298, 0.1973],
            [0.9357, 0.729, 0.2061],
            [0.944, 0.7285, 0.2151],
            [0.9523, 0.7284, 0.2237],
            [0.9606, 0.7285, 0.2312],
            [0.9689, 0.7292, 0.2373],
            [0.977, 0.7304, 0.2418],
            [0.9842, 0.733, 0.2446],
            [0.99, 0.7365, 0.2429],
            [0.9946, 0.7407, 0.2394],
            [0.9966, 0.7458, 0.2351],
            [0.9971, 0.7513, 0.2309],
            [0.9972, 0.7569, 0.2267],
            [0.9971, 0.7626, 0.2224],
            [0.9969, 0.7683, 0.2181],
            [0.9966, 0.774, 0.2138],
            [0.9962, 0.7798, 0.2095],
            [0.9957, 0.7856, 0.2053],
            [0.9949, 0.7915, 0.2012],
            [0.9938, 0.7974, 0.1974],
            [0.9923, 0.8034, 0.1939],
            [0.9906, 0.8095, 0.1906],
            [0.9885, 0.8156, 0.1875],
            [0.9861, 0.8218, 0.1846],
            [0.9835, 0.828, 0.1817],
            [0.9807, 0.8342, 0.1787],
            [0.9778, 0.8404, 0.1757],
            [0.9748, 0.8467, 0.1726],
            [0.972, 0.8529, 0.1695],
            [0.9694, 0.8591, 0.1665],
            [0.9671, 0.8654, 0.1636],
            [0.9651, 0.8716, 0.1608],
            [0.9634, 0.8778, 0.1582],
            [0.9619, 0.884, 0.1557],
            [0.9608, 0.8902, 0.1532],
            [0.9601, 0.8963, 0.1507],
            [0.9596, 0.9023, 0.148],
            [0.9595, 0.9084, 0.145],
            [0.9597, 0.9143, 0.1418],
            [0.9601, 0.9203, 0.1382],
            [0.9608, 0.9262, 0.1344],
            [0.9618, 0.932, 0.1304],
            [0.9629, 0.9379, 0.1261],
            [0.9642, 0.9437, 0.1216],
            [0.9657, 0.9494, 0.1168],
            [0.9674, 0.9552, 0.1116],
            [0.9692, 0.9609, 0.1061],
            [0.9711, 0.9667, 0.1001],
            [0.973, 0.9724, 0.0938],
            [0.9749, 0.9782, 0.0872],
            [0.9769, 0.9839, 0.0805],
        ]

        cmap = LinearSegmentedColormap.from_list("parula", cm_data)
        if as_colors:
            N = getattr(cmap, "N", 256)
            colors = cmap(np.linspace(0, 1, N))
            cmap = mcolors.ListedColormap(colors)
        return cmap

    # If colormap not found in palettable or custom colormaps, try matplotlib
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(name)
        print(f"Found '{name}' in matplotlib")
        if as_colors:
            N = getattr(cmap, "N", 256)
            colors = cmap(np.linspace(0, 1, N))
            cmap = mcolors.ListedColormap(colors)
        return cmap
    except ValueError:
        print(
            f"Colormap '{name}' not found in palettable, custom colormaps, or matplotlib"
        )
        print("Available custom colormaps: parula")
        print("Falling back to 'viridis'")
        cmap = plt.get_cmap("viridis")
        if as_colors:
            N = getattr(cmap, "N", 256)
            colors = cmap(np.linspace(0, 1, N))
            cmap = mcolors.ListedColormap(colors)
        return cmap


def list_available_colormaps():
    """
    List all available colormaps from palettable and custom colormaps.

    Returns:
    --------
    dict : Dictionary with categories as keys and colormap names as values
    """
    import inspect

    available_colormaps = {
        "custom": ["parula"],
        "matplotlib": [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "turbo",
            "coolwarm",
            "seismic",
        ],
    }

    # Get palettable colormaps
    try:
        import palettable

        for module_name in dir(palettable):
            if module_name.startswith("_"):
                continue

            module = getattr(palettable, module_name)
            if not inspect.ismodule(module):
                continue

            # Check submodules
            for submodule_name in dir(module):
                if submodule_name.startswith("_"):
                    continue

                submodule = getattr(module, submodule_name)
                if not inspect.ismodule(submodule):
                    continue

                category_name = f"{module_name}_{submodule_name}"
                available_colormaps[category_name] = []

                # Get all colormap objects
                for attr_name in dir(submodule):
                    if attr_name.startswith("_"):
                        continue

                    attr = getattr(submodule, attr_name)
                    if hasattr(attr, "mpl_colormap"):
                        available_colormaps[category_name].append(attr_name)

                # Remove empty categories
                if not available_colormaps[category_name]:
                    del available_colormaps[category_name]

    except ImportError:
        print("Palettable not available")

    return available_colormaps

# Summary for excel table
def add_bad_span_annotations(raw, bad_spans):
    """Add bad span annotations using onset/offset times."""
    if not bad_spans:
        return raw
    
    ann = mne.Annotations(
        onset=[float(s["onset"]) for s in bad_spans],
        duration=[float(s["offset"]) - float(s["onset"]) for s in bad_spans],
        description=[f"bad_{d}" if not (d := s.get("description", "bad_span")).lower().startswith("bad") else d for s in bad_spans],
        orig_time=raw.info["meas_date"]
    )
    raw.set_annotations(raw.annotations + ann if raw.annotations else ann)
    return raw

def summarize_recording_duration(raw,
                                 eyes_closed_event="Stimulus/s1050",
                                 bad_prefix="bad"):
    """
    Return summary of durations (in seconds):
      - total recording duration
      - duration after rejecting bad spans
      - eyes-closed marker time (seconds from start)
      - eyes-open duration (before marker)
      - eyes-closed duration (after marker)

    If marker is missing, or you choose to adjust it, asks for time in seconds.
    """
    sfreq = raw.info["sfreq"]
    total = float(raw.times[-1])

    # bad spans
    ann = raw.annotations
    if ann is not None and len(ann) > 0:
        bad_mask = [d.lower().startswith(bad_prefix) for d in ann.description]
        bad_total = float(np.sum(ann.duration[bad_mask]))
    else:
        bad_total = 0.0
    clean_total = max(0.0, total - bad_total)

    events, event_id = mne.events_from_annotations(raw)

    def ask_for_marker_time():
        print(f"\nEnter eyes-closed start time (0–{total:.1f} s).")
        while True:
            try:
                t = float(input("Marker time in seconds: ").strip())
                if 0 <= t <= total:
                    print(f"✓ Using marker time: {t:.1f} s")
                    return t
                else:
                    print(f"Time must be between 0 and {total:.1f} s.")
            except ValueError:
                print("Please enter a valid number.")

    # case 1: marker missing → must enter manually
    if eyes_closed_event not in event_id:
        print(f"\n[WARN] Event '{eyes_closed_event}' not found.")
        print(f"Available events: {list(event_id.keys())}")
        marker_time = ask_for_marker_time()

        # add manual annotation
        new_ann = mne.Annotations(
            onset=[marker_time],
            duration=[0],
            description=[eyes_closed_event],
            orig_time=raw.info["meas_date"],
        )
        raw.set_annotations(raw.annotations + new_ann if raw.annotations else new_ann)
        print(f"✓ Added '{eyes_closed_event}' at {marker_time:.1f} s")

    # case 2: marker present → optionally reposition
    else:
        code = event_id[eyes_closed_event]
        marker_sample = events[events[:, 2] == code][0, 0]
        marker_time = float(marker_sample / sfreq)
        print(f"\nCurrent {eyes_closed_event} at {marker_time:.1f} s.")
        ans = input("Reposition marker? (y/n): ").strip().lower()
        if ans == "y":
            marker_time = ask_for_marker_time()
            # remove old annotations of this type and add new one
            if ann is not None and len(ann) > 0:
                keep_mask = [d != eyes_closed_event for d in ann.description]
                kept_ann = ann[keep_mask]
            else:
                kept_ann = ann
            new_ann = mne.Annotations(
                onset=[marker_time],
                duration=[0],
                description=[eyes_closed_event],
                orig_time=raw.info["meas_date"],
            )
            raw.set_annotations(kept_ann + new_ann if kept_ann is not None else new_ann)
            print(f"✓ Replaced '{eyes_closed_event}' at {marker_time:.1f} s")

    # EO / EC durations
    eo = marker_time
    ec = max(0.0, total - marker_time)

    print(
        f"Total: {total:.1f}s | Clean: {clean_total:.1f}s | "
        f"Marker ({eyes_closed_event}): {marker_time:.1f}s | "
        f"EO: {eo:.1f}s | EC: {ec:.1f}s"
    )
    return total, clean_total, marker_time, eo, ec

def summarize_bad_spans(raw, bad_prefix="bad"):
    """
    Print and return number and total duration of bad-span annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with annotations.
    bad_prefix : str
        Descriptions starting with this prefix are treated as bad spans.
    """
    ann = raw.annotations
    if ann is None or len(ann) == 0:
        print("No annotations.")
        return 0, 0.0

    mask = [d.lower().startswith(bad_prefix) for d in ann.description]
    n_bad = sum(mask)
    total_bad = float(sum(ann.duration[mask]))
    print(f"Bad spans: {n_bad}, total bad duration: {total_bad:.1f} s")
    return n_bad, total_bad

def split_and_save_eo_ec(preproc_path, eyes_closed_event="Stimulus/s1050", suffix_raw="raw"):
    preproc_path = Path(preproc_path)
    raw = mne.io.read_raw(preproc_path, preload=True)
    
    # Find marker time
    events, event_id = mne.events_from_annotations(raw)
    marker_time = events[events[:, 2] == event_id[eyes_closed_event]][0, 0] / raw.info["sfreq"]
    
    # Crop into EO and EC
    raw_eo, raw_ec = raw.copy().crop(tmax=marker_time), raw.copy().crop(tmin=marker_time)
    
    # Build filenames
    ext = ".fif.gz" if preproc_path.name.endswith(".fif.gz") else ".fif"
    stem = preproc_path.name[:-7 if ext == ".fif.gz" else -4]
    eo_path, ec_path = preproc_path.with_name(f"{stem}_eo{ext}"), preproc_path.with_name(f"{stem}_ec{ext}")
    
    # Save and print
    raw_eo.save(eo_path, overwrite=True)
    raw_ec.save(ec_path, overwrite=True)
    print(f"Saved EO to {eo_path}\nSaved EC to {ec_path}")
    
    return eo_path, ec_path