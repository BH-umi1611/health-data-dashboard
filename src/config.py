"""Shared constants: column names, plausible value ranges, labels and category orders."""

ID_COL = "patient_id"

# Order of the columns in the CSV file
ALL_COLS = [
    "patient_id", "age", "gender", "blood_group", "smoker",
    "height_cm", "weight_kg", "systolic_bp", "diastolic_bp",
    "heart_rate", "cholesterol", "glucose",
]

NUMERIC_COLS = [
    "age", "height_cm", "weight_kg", "systolic_bp", "diastolic_bp",
    "heart_rate", "cholesterol", "glucose",
]

CATEGORICAL_COLS = ["gender", "blood_group", "smoker"]

# Anything outside these (min, max) limits is treated as a data-entry error.
VALID_RANGES = {
    "age": (0, 120),
    "height_cm": (100, 220),
    "weight_kg": (25, 250),
    "systolic_bp": (70, 250),
    "diastolic_bp": (40, 150),
    "heart_rate": (30, 220),
    "cholesterol": (80, 500),
    "glucose": (40, 600),
}

# Axis labels (with units) and short names for titles
LABELS = {
    "age": "Age (years)",
    "bmi": "BMI (kg/m²)",
    "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "heart_rate": "Heart rate (bpm)",
    "cholesterol": "Total cholesterol (mg/dL)",
    "glucose": "Fasting glucose (mg/dL)",
}

SHORT = {
    "age": "Age",
    "bmi": "BMI",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "heart_rate": "Heart rate",
    "cholesterol": "Cholesterol",
    "glucose": "Glucose",
}

# Display order for categories (so charts read low -> high)
AGE_GROUP_ORDER = ["18-29", "30-44", "45-59", "60-74", "75+"]
BP_ORDER = ["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"]
CHOL_ORDER = ["Desirable", "Borderline high", "High"]
BMI_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]
BLOOD_ORDER = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
