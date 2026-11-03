"""Generate a synthetic patient dataset.

Real patient records are private, so this module builds realistic *fake* data
with NumPy. It then deliberately damages a small part of it (missing values,
impossible readings and duplicate rows) so the cleaning step has real work to do.

Run directly to (re)create the CSV:
    python -m src.generate_data
"""
import csv
from pathlib import Path

import numpy as np

from .config import ALL_COLS, BLOOD_ORDER, NUMERIC_COLS


def _fmt(value):
    """Turn a NumPy value into a tidy string for the CSV."""
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{value:.1f}"
    return str(value)


def _build_clean_table(n, rng):
    """Create n clean, internally consistent patient records."""
    patient_id = np.arange(1001, 1001 + n)
    age = np.clip(rng.normal(48, 16, n), 18, 90).round()
    gender = rng.choice(["Male", "Female"], size=n, p=[0.48, 0.52])
    blood_group = rng.choice(
        BLOOD_ORDER, size=n, p=[0.36, 0.28, 0.20, 0.06, 0.04, 0.03, 0.02, 0.01]
    )
    smoker = rng.choice(["Yes", "No"], size=n, p=[0.22, 0.78])

    male = (gender == "Male").astype(float)
    smoke = (smoker == "Yes").astype(float)

    height = np.where(male == 1, rng.normal(174, 7, n), rng.normal(161, 6, n)).round(1)
    bmi = np.clip(rng.normal(24.5 + 0.03 * (age - 48), 4.0, n), 16, 45)
    weight = (bmi * (height / 100) ** 2).round(1)

    # Vital signs and lab results depend (loosely) on age, BMI and smoking,
    # so the scatter plots show believable correlations.
    systolic = (96 + 0.55 * age + 0.8 * (bmi - 22) + 4 * smoke + 3 * male
                + rng.normal(0, 8, n)).round()
    diastolic = (0.55 * systolic + 12 + rng.normal(0, 5, n)).round()
    heart_rate = (72 - 0.08 * (age - 48) + 5 * smoke + 0.3 * (bmi - 22)
                  + rng.normal(0, 8, n)).round()
    cholesterol = (135 + 0.9 * age + 1.8 * (bmi - 22) + 12 * smoke
                   + rng.normal(0, 22, n)).round()
    glucose = (78 + 0.25 * age + 0.9 * (bmi - 22) + rng.normal(0, 9, n)
               + rng.exponential(4, n)).round()

    return {
        "patient_id": patient_id, "age": age, "gender": gender,
        "blood_group": blood_group, "smoker": smoker,
        "height_cm": height, "weight_kg": weight,
        "systolic_bp": systolic, "diastolic_bp": diastolic,
        "heart_rate": heart_rate, "cholesterol": cholesterol, "glucose": glucose,
    }


def generate_dataset(path, n=500, seed=42):
    """Write a messy patient CSV to `path` and return the path."""
    path = Path(path)
    rng = np.random.default_rng(seed)
    table = _build_clean_table(n, rng)

    rows = np.empty((n, len(ALL_COLS)), dtype=object)
    for j, col in enumerate(ALL_COLS):
        rows[:, j] = [_fmt(v) for v in table[col]]

    col_idx = {c: j for j, c in enumerate(ALL_COLS)}
    numeric_idx = [col_idx[c] for c in NUMERIC_COLS]

    # 1) ~3% of numeric cells go missing
    n_missing = int(0.03 * n * len(numeric_idx))
    rows[rng.integers(0, n, n_missing), rng.choice(numeric_idx, n_missing)] = ""

    # 2) a handful of impossible readings (typos / sensor faults)
    bad_values = [
        ("age", "250"), ("systolic_bp", "0"), ("cholesterol", "-5"),
        ("heart_rate", "400"), ("glucose", "9999"), ("height_cm", "20"),
        ("weight_kg", "700"), ("diastolic_bp", "300"),
    ]
    bad_rows = rng.choice(n, size=len(bad_values), replace=False)
    for r, (col, bad) in zip(bad_rows, bad_values):
        rows[r, col_idx[col]] = bad

    # 3) duplicate records, then shuffle so they are not all at the end
    dup = rng.choice(n, size=8, replace=False)
    rows = np.vstack([rows, rows[dup]])
    rows = rows[rng.permutation(len(rows))]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ALL_COLS)
        writer.writerows(rows.tolist())
    return path


if __name__ == "__main__":
    out = generate_dataset(Path(__file__).resolve().parent.parent / "data" / "patient_records.csv")
    print(f"Wrote {out}")
