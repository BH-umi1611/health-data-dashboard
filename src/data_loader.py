"""Load and save patient data as a dict of NumPy column arrays."""
import csv
from pathlib import Path

import numpy as np

from .config import ID_COL, NUMERIC_COLS


def load_patient_data(path):
    """Read a CSV into {column_name: numpy array}.

    * numeric columns  -> float arrays (blank cells become NaN)
    * patient_id       -> int array
    * everything else  -> string arrays
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]

    raw = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1, encoding="utf-8")
    raw = np.atleast_2d(raw)

    data = {}
    for j, name in enumerate(header):
        col = np.char.strip(raw[:, j])
        if name in NUMERIC_COLS:
            data[name] = np.where(col == "", "nan", col).astype(float)
        elif name == ID_COL:
            data[name] = col.astype(int)
        else:
            data[name] = col.astype(str)
    return data


def save_patient_data(data, path):
    """Write a dict of column arrays back to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(data.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in zip(*(data[c].tolist() for c in cols)):
            writer.writerow(row)
    return path
