
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load trained models and preprocessing tools
rf_model = joblib.load(BASE_DIR / "random_forest_model.pkl")
log_model = joblib.load(BASE_DIR / "logistic_regression_model.pkl")
imputer = joblib.load(BASE_DIR / "heart_disease_imputer.pkl")
scaler = joblib.load(BASE_DIR / "heart_disease_scaler.pkl")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]


def predict_heart_disease(patient_data, model_choice):
    patient = pd.DataFrame([patient_data], columns=FEATURES)
    patient_imputed = imputer.transform(patient)

    if model_choice == "1":
        prediction = rf_model.predict(patient_imputed)[0]

    elif model_choice == "2":
        patient_scaled = scaler.transform(patient_imputed)
        prediction = log_model.predict(patient_scaled)[0]

    else:
        raise ValueError("Invalid model selection.")

    return prediction


def main():
    print("\n====================================")
    print("     HEART DISEASE PREDICTION")
    print("====================================")

    patient_data = []

    for feature in FEATURES:
        value = float(input(f"Enter {feature}: "))
        patient_data.append(value)

    print("\nSelect a prediction model:")
    print("1. Random Forest")
    print("2. Logistic Regression")

    choice = input("Enter choice (1 or 2): ")

    if choice not in ["1", "2"]:
        print("Invalid selection.")
        return

    prediction = predict_heart_disease(patient_data, choice)

    print("\n========== PREDICTION ==========")

    if prediction == 1:
        print("Prediction: Heart disease detected.")
    else:
        print("Prediction: No heart disease detected.")

    print("\nThis project is for educational purposes only.")
    print("It is not a medical diagnostic tool.")


if __name__ == "__main__":
    main()
