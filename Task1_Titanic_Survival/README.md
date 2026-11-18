# Titanic Survival Prediction 🚢

APP LINK:  https://titanic-survival-intelligence.onrender.com

## Project Overview
This project aims to predict whether a passenger survived the Titanic disaster using Machine Learning classification algorithms.

The model analyzes passenger information such as passenger class, gender, age, family members aboard, ticket fare, and embarkation port to predict survival.

## Dataset
- Source: [Kaggle – Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- Training records: 891
- Testing records: 418
- Target variable: `Survived`
  - 0 – Not Survived
  - 1 – Survived

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- Google Colab

## Project Workflow

1. Data Loading and Exploration
2. Exploratory Data Analysis (EDA)
3. Missing Value Handling
4. Feature Selection and Preprocessing
5. Categorical Feature Handling
6. Train-Validation Split (80:20)
7. Machine Learning Model Training
8. Model Comparison and Evaluation
9. Final Model Training and Saving
10. Test Dataset Prediction

## Models Implemented

- Logistic Regression / baseline exploration
- XGBoost Classifier
- CatBoost Classifier
- LightGBM Classifier
- Random Forest Classifier

## Model Performance

Validation results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CatBoost | **82.68%** | 85.19% | 66.67% | 74.80% | 84.84% |
| XGBoost | 79.89% | 80.00% | 63.77% | 70.97% | 81.93% |
| Random Forest | 79.89% | 74.63% | 72.46% | 73.53% | 84.32% |
| LightGBM | 78.21% | 76.79% | 62.32% | 68.80% | 82.81% |

## Best Model
CatBoost Classifier achieved the highest validation accuracy of **82.68%**.

The final CatBoost model was trained using all 891 labelled training records and saved as:

`best_titanic_model.cbm`

## Test Predictions
- Total test passengers: 418
- Predicted survivors: 139
- Predicted non-survivors: 279

The predictions were saved in `titanic_predictions.csv`.

Note: The Kaggle test dataset does not contain actual survival labels, so these prediction counts are not test accuracy.

## Live Demo
[ https://titanic-survival-intelligence.onrender.com]

## Conclusion
This project demonstrates an end-to-end Machine Learning classification workflow, including data preprocessing, exploratory analysis, model comparison, evaluation, model saving, and predictions on unseen passenger data.

---

**Developed by:** Devadharshini Murugan  
**Internship:** CodSoft – Data Science Internship  
**Task:** 1 – Titanic Survival Prediction
