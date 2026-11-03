"""Statistics (all NumPy) and the plain-text report."""
import numpy as np

from . import filtering
from .config import AGE_GROUP_ORDER, BMI_ORDER, BP_ORDER, CHOL_ORDER, NUMERIC_COLS, SHORT

STAT_COLS = ["age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol", "glucose"]
CORR_COLS = STAT_COLS


# ----------------------------------------------------------------- statistics
def summary_statistics(data, cols=STAT_COLS):
    rows = []
    for c in cols:
        x = data[c]
        rows.append({
            "column": c, "mean": np.mean(x), "median": np.median(x),
            "std": np.std(x, ddof=1), "min": np.min(x), "max": np.max(x),
        })
    return rows


def category_counts(data, column, order=None):
    """Return (labels, counts). `order` fixes the label order and keeps empty categories."""
    labels, counts = np.unique(data[column], return_counts=True)
    lookup = dict(zip(labels.tolist(), counts.tolist()))
    order = list(order) if order is not None else labels.tolist()
    return order, [int(lookup.get(label, 0)) for label in order]


def correlation_matrix(data, cols=CORR_COLS):
    return np.corrcoef(np.vstack([data[c] for c in cols]))


def top_correlations(corr, cols=CORR_COLS, k=6):
    """The k strongest pairs (by |r|), excluding the diagonal."""
    i, j = np.triu_indices(len(cols), k=1)
    order = np.argsort(-np.abs(corr[i, j]))[:k]
    return [(cols[i[o]], cols[j[o]], float(corr[i[o], j[o]])) for o in order]


def describe_strength(r):
    a = abs(r)
    word = ("very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6
            else "strong" if a < 0.8 else "very strong")
    return f"{word} {'positive' if r > 0 else 'negative'}"


# --------------------------------------------------------------------- report
def _pct(part, total):
    return f"{100 * part / total:5.1f}%"


def _group_rows(data, group_col, order=None):
    groups = list(order) if order is not None else np.unique(data[group_col]).tolist()
    rows = []
    for g in groups:
        m = data[group_col] == g
        if m.any():
            rows.append((str(g), int(m.sum()), data["systolic_bp"][m].mean(),
                         data["cholesterol"][m].mean(), data["glucose"][m].mean(),
                         data["bmi"][m].mean()))
    return rows


def build_report(clean, cr):
    """Build the full text report. `clean` = cleaned data, `cr` = cleaning report."""
    n = filtering.num_records(clean)
    out = []
    add = out.append

    def section(title):
        add("")
        add(title)
        add("-" * len(title))

    add("=" * 72)
    add("HEALTH DATA ANALYSIS REPORT")
    add("=" * 72)

    # 1 ---------------------------------------------------------------
    section("1. Data cleaning")
    add(f"Records loaded            : {cr['records_loaded']}")
    add(f"Duplicate records removed : {cr['duplicates_removed']}")
    add(f"Records after cleaning    : {cr['records_final']}")
    add("")
    add(f"{'Column':<14}{'Missing':>9}{'Invalid':>9}{'Filled':>9}")
    for c in NUMERIC_COLS:
        add(f"{c:<14}{cr['missing'][c]:>9}{cr['invalid'][c]:>9}{cr['filled'][c]:>9}")
    add("Invalid = outside a plausible range (treated as missing).")
    add("Filled  = missing + invalid, replaced by the median of the patient's gender group.")

    # 2 ---------------------------------------------------------------
    section("2. Patient demographics")
    age = clean["age"]
    add(f"Patients: {n}   Age: mean {age.mean():.1f}, median {np.median(age):.0f}, "
        f"range {age.min():.0f}-{age.max():.0f}")
    for title, col, order in [("Gender", "gender", None), ("Smoker", "smoker", None),
                              ("Age group", "age_group", AGE_GROUP_ORDER)]:
        labels, counts = category_counts(clean, col, order)
        add(f"{title:<10}: " + ",  ".join(f"{l} {c} ({_pct(c, n).strip()})"
                                          for l, c in zip(labels, counts)))

    # 3 ---------------------------------------------------------------
    section("3. Summary statistics")
    add(f"{'Measure':<14}{'Mean':>9}{'Median':>9}{'Std':>9}{'Min':>9}{'Max':>9}")
    for r in summary_statistics(clean):
        add(f"{SHORT[r['column']]:<14}{r['mean']:>9.1f}{r['median']:>9.1f}"
            f"{r['std']:>9.1f}{r['min']:>9.1f}{r['max']:>9.1f}")

    # 4 ---------------------------------------------------------------
    section("4. Health indicators")
    for title, col, order in [("Blood pressure", "bp_category", BP_ORDER),
                              ("Cholesterol", "chol_category", CHOL_ORDER),
                              ("BMI", "bmi_category", BMI_ORDER)]:
        add(f"{title}:")
        labels, counts = category_counts(clean, col, order)
        for l, c in zip(labels, counts):
            add(f"   {l:<24}{c:>5}  {_pct(c, n)}")

    # 5 ---------------------------------------------------------------
    section("5. Group comparisons (means)")
    add(f"{'Group':<18}{'n':>5}{'Systolic':>10}{'Chol.':>9}{'Glucose':>9}{'BMI':>7}")
    for title, col, order in [("By gender", "gender", None), ("By smoking", "smoker", None),
                              ("By age group", "age_group", AGE_GROUP_ORDER)]:
        add(f"[{title}]")
        for g, cnt, sbp, chol, glu, bmi in _group_rows(clean, col, order):
            add(f"{g:<18}{cnt:>5}{sbp:>10.1f}{chol:>9.1f}{glu:>9.1f}{bmi:>7.1f}")

    # 6 ---------------------------------------------------------------
    section("6. Filtering examples")
    seniors = filtering.by_age(clean, min_age=60)
    senior_stage2 = filtering.by_category(seniors, "bp_category", "Hypertension Stage 2")
    smoker_high_chol = filtering.subset(
        clean, (clean["smoker"] == "Yes") & (clean["cholesterol"] >= 240))
    at_risk = filtering.high_risk(clean)
    for text, sub in [
        ("Patients aged 60+", seniors),
        ("Aged 60+ with Stage 2 hypertension", senior_stage2),
        ("Smokers with cholesterol >= 240", smoker_high_chol),
        ("High-risk (Stage 2 BP, chol >= 240 or glucose >= 126)", at_risk),
    ]:
        k = filtering.num_records(sub)
        add(f"{text:<56}{k:>4}  ({_pct(k, n).strip()})")
    add("")
    add("Five highest cholesterol readings:")
    add(f"   {'ID':<8}{'Age':>5}  {'Gender':<8}{'Cholesterol':>12}")
    top = filtering.top_n(clean, "cholesterol", 5)
    for pid, a, g, ch in zip(top["patient_id"], top["age"], top["gender"], top["cholesterol"]):
        add(f"   {pid:<8}{a:>5.0f}  {g:<8}{ch:>12.0f}")

    # 7 ---------------------------------------------------------------
    section("7. Correlations (Pearson r)")
    corr = correlation_matrix(clean)
    add(" " * 14 + "".join(f"{SHORT[c]:>13}" for c in CORR_COLS))
    for name, row in zip(CORR_COLS, corr):
        add(f"{SHORT[name]:<14}" + "".join(f"{v:>13.2f}" for v in row))
    add("")
    add("Strongest relationships:")
    for a, b, r in top_correlations(corr):
        pair = f"{SHORT[a]} vs {SHORT[b]}"
        add(f"   {pair:<28}r = {r:+.2f}  ({describe_strength(r)})")
    add("")
    add("Note: correlation does not prove causation. This sample is synthetic and "
        "for learning only - it is not medical advice.")
    return "\n".join(out)
