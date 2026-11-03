"""Web dashboard for the health data analysis (Flask backend).

    python app.py
    then open  http://127.0.0.1:8000
"""
import os
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

from src import analysis, cleaning, filtering, visualize
from src.config import AGE_GROUP_ORDER, BMI_ORDER, BP_ORDER, CHOL_ORDER, SHORT
from src.data_loader import load_patient_data
from src.generate_data import generate_dataset

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "patient_records.csv"

# ---- load + clean the data once, when the server starts --------------------
if not DATA_FILE.exists():
    generate_dataset(DATA_FILE)
DATA, CLEAN_REPORT = cleaning.clean_pipeline(load_patient_data(DATA_FILE))
visualize.apply_style()

app = Flask(__name__)

SCATTER_CHOICES = analysis.CORR_COLS
TABLE_COLS = ["patient_id", "age", "gender", "smoker", "bmi", "systolic_bp",
              "diastolic_bp", "cholesterol", "glucose", "bp_category"]
TABLE_HEADERS = {
    "patient_id": "ID", "age": "Age", "gender": "Gender", "smoker": "Smoker",
    "bmi": "BMI", "systolic_bp": "Systolic", "diastolic_bp": "Diastolic",
    "cholesterol": "Cholesterol", "glucose": "Glucose", "bp_category": "BP category",
}
SORTABLE = ["patient_id", "age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol", "glucose"]
PAGE_SIZE = 100

# matplotlib is not thread-safe, so charts are drawn one at a time and cached
_plot_lock = Lock()
_plot_cache = {}


# ------------------------------------------------------------------ helpers
def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell(value):
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return value


def kpis():
    n = filtering.num_records(DATA)
    hypertensive = np.isin(DATA["bp_category"], ["Hypertension Stage 1", "Hypertension Stage 2"])
    at_risk = filtering.num_records(filtering.high_risk(DATA))
    return {
        "patients": n,
        "mean_age": round(float(DATA["age"].mean()), 1),
        "mean_cholesterol": round(float(DATA["cholesterol"].mean())),
        "hypertension_pct": round(100 * float(hypertensive.mean()), 1),
        "high_risk_pct": round(100 * at_risk / n, 1),
        "smokers_pct": round(100 * float((DATA["smoker"] == "Yes").mean()), 1),
    }


# ------------------------------------------------------------------- pages
@app.route("/")
def dashboard():
    x = request.args.get("x", "age")
    y = request.args.get("y", "cholesterol")
    if x not in SCATTER_CHOICES:
        x = "age"
    if y not in SCATTER_CHOICES or y == x:
        y = next(c for c in ("cholesterol", "systolic_bp") if c != x)
    return render_template("dashboard.html", active="dashboard", kpi=kpis(),
                           x=x, y=y, choices=SCATTER_CHOICES, short=SHORT)


@app.route("/patients")
def patients():
    f = {k: request.args.get(k, "") for k in
         ("min_age", "max_age", "gender", "smoker", "bp", "sort", "order")}

    sub = filtering.by_age(DATA, _num(f["min_age"]), _num(f["max_age"]))
    if f["gender"] in ("Male", "Female"):
        sub = filtering.by_category(sub, "gender", f["gender"])
    if f["smoker"] in ("Yes", "No"):
        sub = filtering.by_category(sub, "smoker", f["smoker"])
    if f["bp"] in BP_ORDER:
        sub = filtering.by_category(sub, "bp_category", f["bp"])

    sort_col = f["sort"] if f["sort"] in SORTABLE else "patient_id"
    index = np.argsort(sub[sort_col], kind="stable")
    if f["order"] == "desc":
        index = index[::-1]

    total_match = filtering.num_records(sub)
    page = filtering.subset(sub, index[:PAGE_SIZE])
    columns = [page[c].tolist() for c in TABLE_COLS]
    rows = [[_cell(v) for v in row] for row in zip(*columns)]

    return render_template("patients.html", active="patients", f=f, rows=rows,
                           headers=[TABLE_HEADERS[c] for c in TABLE_COLS],
                           matched=total_match, total=filtering.num_records(DATA),
                           shown=len(rows), bp_options=BP_ORDER,
                           sortable=SORTABLE, headers_map=TABLE_HEADERS)


@app.route("/report")
def report():
    text = analysis.build_report(DATA, CLEAN_REPORT)
    return app.response_class(text, mimetype="text/plain; charset=utf-8")


# --------------------------------------------------------------------- API
@app.route("/api/summary")
def api_summary():
    corr = analysis.correlation_matrix(DATA)
    return jsonify({
        "kpis": kpis(),
        "cleaning": CLEAN_REPORT,
        "counts": {
            "gender": dict(zip(*analysis.category_counts(DATA, "gender"))),
            "age_group": dict(zip(*analysis.category_counts(DATA, "age_group", AGE_GROUP_ORDER))),
            "bp_category": dict(zip(*analysis.category_counts(DATA, "bp_category", BP_ORDER))),
            "chol_category": dict(zip(*analysis.category_counts(DATA, "chol_category", CHOL_ORDER))),
            "bmi_category": dict(zip(*analysis.category_counts(DATA, "bmi_category", BMI_ORDER))),
        },
        "top_correlations": [
            {"a": a, "b": b, "r": round(r, 3)} for a, b, r in analysis.top_correlations(corr)
        ],
    })


# ------------------------------------------------------------------- charts
CHARTS = {
    "demographics": visualize.demographics_figure,
    "health_status": visualize.health_status_figure,
    "heatmap": visualize.correlation_heatmap,
}


@app.route("/plot/<name>.png")
def plot(name):
    if name == "scatter":
        x, y = request.args.get("x", "age"), request.args.get("y", "cholesterol")
        if x not in SCATTER_CHOICES or y not in SCATTER_CHOICES or x == y:
            abort(400)
        key = ("scatter", x, y)
        draw = lambda buf: visualize.scatter_figure(DATA, x, y, buf)
    elif name in CHARTS:
        key = (name,)
        draw = lambda buf: CHARTS[name](DATA, buf)
    else:
        abort(404)

    with _plot_lock:
        if key not in _plot_cache:
            buf = BytesIO()
            draw(buf)
            _plot_cache[key] = buf.getvalue()
    return send_file(BytesIO(_plot_cache[key]), mimetype="image/png")


if __name__ == "__main__":
    # Port 5000 is used by AirPlay on recent macOS, so default to 8000.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
