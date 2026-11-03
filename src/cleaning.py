"""Data cleaning with NumPy: de-duplicate, validate, impute, derive new columns."""
import numpy as np

from .config import CATEGORICAL_COLS, NUMERIC_COLS, VALID_RANGES
from .filtering import num_records, subset


def standardise_text(data):
    """Trim spaces and fix capitalisation of text columns ('male ' -> 'Male')."""
    out = dict(data)
    for col in CATEGORICAL_COLS:
        if col not in out:
            continue
        values = np.char.strip(out[col])
        out[col] = np.char.upper(values) if col == "blood_group" else np.char.capitalize(values)
    return out


def remove_duplicates(data):
    """Keep the first row for every patient_id. Returns (data, number_removed)."""
    _, first_index = np.unique(data["patient_id"], return_index=True)
    keep = np.sort(first_index)          # np.unique sorts by id; restore file order
    return subset(data, keep), num_records(data) - keep.size


def flag_invalid_values(data):
    """Replace impossible readings with NaN. Returns (data, {column: count})."""
    out = {k: v.copy() for k, v in data.items()}
    counts = {}
    for col, (low, high) in VALID_RANGES.items():
        x = out[col]
        bad = ~np.isnan(x) & ((x < low) | (x > high))
        x[bad] = np.nan
        counts[col] = int(bad.sum())

    # Diastolic pressure can never be >= systolic pressure
    sys_, dia = out["systolic_bp"], out["diastolic_bp"]
    conflict = ~np.isnan(sys_) & ~np.isnan(dia) & (dia >= sys_)
    dia[conflict] = np.nan
    counts["diastolic_bp"] += int(conflict.sum())
    return out, counts


def impute_missing(data, group_col="gender"):
    """Fill NaN with the median of the patient's group. Returns (data, {column: filled})."""
    out = {k: v.copy() for k, v in data.items()}
    groups = np.unique(out[group_col])
    filled = {}
    for col in NUMERIC_COLS:
        x = out[col]                      # a view: edits below change `out`
        missing = np.isnan(x)
        filled[col] = int(missing.sum())
        if not missing.any():
            continue
        overall = np.nanmedian(x)
        for g in groups:
            in_group = out[group_col] == g
            known = x[in_group & ~missing]
            x[in_group & missing] = np.median(known) if known.size else overall
    return out, filled


def add_derived_features(data):
    """Add BMI, age group and clinical category columns."""
    out = dict(data)
    height_m = data["height_cm"] / 100
    out["bmi"] = np.round(data["weight_kg"] / height_m ** 2, 1)

    age_labels = np.array(["18-29", "30-44", "45-59", "60-74", "75+"])
    out["age_group"] = age_labels[np.digitize(data["age"], [30, 45, 60, 75])]

    s, d = data["systolic_bp"], data["diastolic_bp"]
    out["bp_category"] = np.select(
        [(s >= 140) | (d >= 90), (s >= 130) | (d >= 80), s >= 120],
        ["Hypertension Stage 2", "Hypertension Stage 1", "Elevated"],
        default="Normal",
    )

    c = data["cholesterol"]
    out["chol_category"] = np.select(
        [c >= 240, c >= 200], ["High", "Borderline high"], default="Desirable"
    )

    b = out["bmi"]
    out["bmi_category"] = np.select(
        [b >= 30, b >= 25, b >= 18.5], ["Obese", "Overweight", "Normal"], default="Underweight"
    )
    return out


def clean_pipeline(raw):
    """Run every cleaning step. Returns (clean_data, report_dict)."""
    report = {"records_loaded": num_records(raw)}

    data = standardise_text(raw)
    data, report["duplicates_removed"] = remove_duplicates(data)
    report["missing"] = {c: int(np.isnan(data[c]).sum()) for c in NUMERIC_COLS}
    data, report["invalid"] = flag_invalid_values(data)
    data, report["filled"] = impute_missing(data)
    data = add_derived_features(data)

    report["records_final"] = num_records(data)
    return data, report
