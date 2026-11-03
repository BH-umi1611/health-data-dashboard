"""Filtering helpers built on NumPy boolean masks and index arrays."""
import numpy as np


def num_records(data):
    return len(next(iter(data.values())))


def subset(data, mask_or_index):
    """Keep only the rows picked by a boolean mask or an integer index array."""
    return {name: values[mask_or_index] for name, values in data.items()}


def by_age(data, min_age=None, max_age=None):
    mask = np.ones(num_records(data), dtype=bool)
    if min_age is not None:
        mask &= data["age"] >= min_age
    if max_age is not None:
        mask &= data["age"] <= max_age
    return subset(data, mask)


def by_category(data, column, *values):
    """Rows whose `column` equals any of `values`."""
    return subset(data, np.isin(data[column], values))


def high_risk(data):
    """Stage 2 hypertension, high cholesterol (>=240) or diabetic-range glucose (>=126)."""
    mask = (
        (data["bp_category"] == "Hypertension Stage 2")
        | (data["cholesterol"] >= 240)
        | (data["glucose"] >= 126)
    )
    return subset(data, mask)


def top_n(data, column, n=10, largest=True):
    order = np.argsort(data[column])
    index = order[::-1][:n] if largest else order[:n]
    return subset(data, index)


def group_mean(data, group_col, value_col):
    """Mean of `value_col` for each distinct value of `group_col`."""
    groups = np.unique(data[group_col])
    return {str(g): float(np.mean(data[value_col][data[group_col] == g])) for g in groups}
