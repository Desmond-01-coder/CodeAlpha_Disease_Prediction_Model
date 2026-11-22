
# Heart Disease Prediction System
### A Machine Learning Approach to Cardiovascular Disease Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Project-Completed-success)
![Internship](https://img.shields.io/badge/CodeAlpha-Machine%20Learning-blue)

## 1. Project Overview

The Heart Disease Prediction System is a supervised machine learning project developed to classify the presence or absence of heart disease using patient medical attributes.

The project explores and compares two classification algorithms: Random Forest and Logistic Regression. It applies appropriate data preprocessing techniques, trains the models, evaluates their performance, and provides a prediction interface for new patient records.

This project was developed as part of the CodeAlpha Machine Learning Internship.

## 2. Project Objectives

- Develop machine learning models for binary classification of heart disease.
- Apply data cleaning and preprocessing techniques.
- Handle missing values effectively.
- Compare the performance of different classification algorithms.
- Evaluate model performance using classification metrics.
- Build a reusable prediction program using trained models.

## 3. Dataset Description

**Dataset:** Cleveland Heart Disease Dataset  
**Source:** UCI Machine Learning Repository  
**Number of Records:** 303  
**Input Features:** 13  
**Target Variable:** Binary classification

The dataset contains medical attributes such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Slope of the ST segment
- Number of major vessels
- Thalassemia

The original target variable was converted into a binary classification:

- `0` — No recorded heart disease
- `1` — Heart disease present

## 4. Data Preprocessing

The following preprocessing operations were performed:

1. **Data Inspection:** Examined dataset structure, feature types, and missing values.
2. **Target Transformation:** Converted the original multi-class target into a binary classification problem.
3. **Train-Test Split:** Divided the dataset into 80% training data and 20% testing data using stratified sampling.
4. **Missing Value Imputation:** Used `SimpleImputer` with the most frequent strategy.
5. **Feature Scaling:** Applied `StandardScaler` for Logistic Regression.

Preprocessing tools were fitted on the training data to prevent data leakage.

## 5. Machine Learning Algorithms

Two supervised learning algorithms were implemented:

### Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

### Logistic Regression
A statistical classification algorithm used as a baseline model for predicting binary outcomes.

## 6. Model Evaluation

Both models were evaluated using the test dataset.

| Model | Test Accuracy |
|---|---:|
| Random Forest | 88.52% |
| Logistic Regression | 86.89% |

Additional evaluation metrics included:

- Precision
- Recall
- F1-score
- Confusion Matrix

Random Forest achieved the highest test accuracy of 88.52% and was selected as the preferred model for the prediction interface.

## 7. Project Structure

```text
CodeAlpha_Disease_Prediction_Model/
│
├── main.py
├── random_forest_model.pkl
├── logistic_regression_model.pkl
├── heart_disease_imputer.pkl
├── heart_disease_scaler.pkl
└── README.md
```

## 8. Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Google Colab

## 9. Running the Project

Ensure Python and the required libraries are installed in a compatible environment.

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

Run the prediction program:

```bash
python main.py
```

The program requests 13 patient attributes, allows the user to select a trained model, and displays the model's prediction.

**Note:** The saved models were developed using Scikit-learn. Use a compatible Scikit-learn version when loading the model files.

## 10. Key Learning Outcomes

Through this project, I gained practical experience in:

- Working with real-world medical datasets.
- Data cleaning and missing-value treatment.
- Binary classification.
- Model training and evaluation.
- Comparing machine learning algorithms.
- Saving and loading trained machine learning models.
- Building reusable prediction pipelines.

## 11. Limitations and Disclaimer

This project is an educational machine learning demonstration based on a relatively small historical dataset.

Its predictions are not clinically validated and must not be used for medical diagnosis, treatment decisions, or healthcare recommendations.

## 12. Acknowledgements

- UCI Machine Learning Repository for providing the Cleveland Heart Disease Dataset.
- CodeAlpha for the Machine Learning Internship project opportunity.

---

**Developed by:** Desmond Saro Peter  
**Project:** CodeAlpha Machine Learning Internship  
**Specialization:** Machine Learning
