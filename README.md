# Health Data Analysis (NumPy mini project)

Analyses patient records, vital signs and lab results using **NumPy** for all
data handling (loading, cleaning, filtering, statistics) and **Matplotlib** for charts.

> The included dataset is **synthetic** (randomly generated), so no real patient
> data is used. Results are for learning only and are not medical advice.

## What it does

| Step | Where | What happens |
|------|-------|--------------|
| 1. Collect | `src/generate_data.py`, `src/data_loader.py` | Creates 500 sample patients (plus deliberate errors) and loads the CSV into NumPy arrays |
| 2. Clean | `src/cleaning.py` | Removes duplicate IDs, fixes text, flags impossible values, fills gaps with group medians, adds BMI / age group / BP, cholesterol and BMI categories |
| 3. Filter | `src/filtering.py` | Boolean-mask filters: by age, category, high-risk patients, top-N, group means |
| 4. Analyse | `src/analysis.py` | Summary statistics, category counts, Pearson correlation matrix, text report |
| 5. Visualise | `src/visualize.py` | Histograms, bar charts, scatter plots with trend lines, correlation heatmap |

## Project layout

```
health_data_analysis/
├── app.py                   # website (Flask)  ->  python app.py
├── main.py                  # scripts only     ->  python main.py
├── templates/  static/      # web pages and CSS
├── requirements.txt
├── data/
│   ├── patient_records.csv        # raw (messy) data, created on first run
│   └── patient_records_clean.csv  # cleaned data
├── outputs/
│   ├── analysis_report.txt
│   ├── 01_demographics.png
│   ├── 02_health_status.png
│   ├── 03_scatter_correlations.png
│   ├── 04_correlation_heatmap.png
│   └── individual/               # each chart on its own
└── src/
    ├── config.py  generate_data.py  data_loader.py
    ├── cleaning.py  filtering.py  analysis.py  visualize.py
```

## Setup in VS Code

1. Unzip and open the folder: **File > Open Folder...**
2. Open a terminal (**Terminal > New Terminal**) and create a virtual environment:

   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Pick the interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**
4. Run the **website**: `python app.py`, then open <http://127.0.0.1:8000> in your browser.
   (Press **Ctrl+C** in the terminal to stop it.)
   Or run the **scripts only**: `python main.py` (writes charts and a report to `outputs/`).
   In VS Code you can also press **F5** and pick a run configuration.

Requires Python 3.9+. Run all commands from inside the `health_data_analysis` folder.

## The website

`app.py` is a small Flask backend that loads and cleans the data with NumPy when it starts,
then serves:

| Page | URL | What it shows |
|------|-----|---------------|
| Dashboard | `/` | Summary cards, demographics, health status, correlation heatmap, and a scatter-plot explorer (pick any two measurements) |
| Patients | `/patients` | Filter by age, gender, smoker and blood pressure; sort; first 100 matches |
| Text report | `/report` | The full analysis report |
| JSON API | `/api/summary` | Key figures as JSON, handy for a separate front end |

To use another port: `PORT=9000 python app.py` (macOS/Linux) or `set PORT=9000 && python app.py` (Windows).
This is a development server for local use. To put it on the internet, deploy it with a
production server such as `gunicorn app:app` on a host like Render or Railway.

## Options

```bash
python main.py                       # first run creates sample data, then analyses
python main.py --regenerate          # new random sample
python main.py --patients 1000 --seed 7 --regenerate
python main.py --data my_file.csv    # analyse your own file
```

## Using your own data

Provide a CSV with these column names (extra columns are ignored by the analysis):

`patient_id, age, gender, blood_group, smoker, height_cm, weight_kg,
systolic_bp, diastolic_bp, heart_rate, cholesterol, glucose`

Blank cells are fine; they are filled during cleaning. Plausible value limits
are defined in `src/config.py` (`VALID_RANGES`) and can be adjusted.

## Ideas to extend

* Add more tests (HbA1c, triglycerides) to `config.py` and the generator
* Compare groups with a t-test (`scipy.stats.ttest_ind`)
* Fit a simple risk model with `np.linalg.lstsq`
* Export the report to PDF or build a dashboard
