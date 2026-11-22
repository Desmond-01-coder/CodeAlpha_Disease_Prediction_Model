import joblib
from pathlib import Path

# Locate the project folder
BASE_DIR = Path(__file__).resolve().parent

# Load the trained models
random_forest_model = joblib.load(
    BASE_DIR / "random_forest_model.pkl"
)

logistic_regression_model = joblib.load(
    BASE_DIR / "logistic_regression_model.pkl"
)

# Load preprocessing tools
imputer = joblib.load(
    BASE_DIR / "heart_disease_imputer.pkl"
)

scaler = joblib.load(
    BASE_DIR / "heart_disease_scaler.pkl"
)

print("Heart Disease Prediction System Loaded Successfully!")