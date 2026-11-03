"""Health Data Analysis - run this file.

    python main.py                       # generate sample data (first run), analyse, plot
    python main.py --regenerate          # create a fresh random sample
    python main.py --data my_file.csv    # analyse your own CSV (same column names)
"""
import argparse
from pathlib import Path

from src import cleaning, analysis, visualize
from src.data_loader import load_patient_data, save_patient_data
from src.generate_data import generate_dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "patient_records.csv"
OUTPUT_DIR = ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Health data analysis with NumPy")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to the patient CSV")
    p.add_argument("--regenerate", action="store_true", help="re-create the sample dataset")
    p.add_argument("--patients", type=int, default=500, help="sample size when generating")
    p.add_argument("--seed", type=int, default=42, help="random seed when generating")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Collect the data
    if args.regenerate or not args.data.exists():
        generate_dataset(args.data, n=args.patients, seed=args.seed)
        print(f"Created sample dataset: {args.data}")
    raw = load_patient_data(args.data)

    # 2. Clean it
    clean, cleaning_report = cleaning.clean_pipeline(raw)
    save_patient_data(clean, ROOT / "data" / "patient_records_clean.csv")

    # 3. Analyse it
    report = analysis.build_report(clean, cleaning_report)
    print(report)
    (OUTPUT_DIR / "analysis_report.txt").write_text(report, encoding="utf-8")

    # 4. Visualise it
    paths = visualize.create_all_plots(clean, OUTPUT_DIR)
    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'analysis_report.txt'}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
