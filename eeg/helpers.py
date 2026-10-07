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