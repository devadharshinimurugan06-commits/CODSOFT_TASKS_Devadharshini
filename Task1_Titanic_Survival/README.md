# Titanic Survival Prediction

RENDER APPLINK: https://titanic-survival-intelligence.onrender.com


**CodSoft Data Science Internship – Task 1**  
**Developed by:** Devadharshini M

## Project Overview

Titanic Survival Prediction is a Machine Learning classification project that predicts whether a passenger survived the Titanic disaster based on passenger-related information.

The model uses features such as passenger class, gender, age, number of siblings or spouses, parents or children aboard, fare, and embarkation port.

The objective of this project is to analyze the Titanic dataset, preprocess the data, train classification models, evaluate their performance, and predict passenger survival.

## Dataset

The project uses the Titanic dataset containing two CSV files:

- **train.csv:** 891 passenger records with survival labels.
- **test.csv:** 418 passenger records without survival labels.

The target variable is `Survived`:
- `0` – Not Survived
- `1` – Survived

## Project Workflow

### 1. Data Loading and Exploration

- Loaded training and testing datasets using Pandas.
- Examined dataset structure, dimensions, and data types.
- Generated descriptive statistical summaries.
- Identified missing values in both datasets.
- Analyzed passenger survival counts and percentages.
- Explored survival patterns based on gender and passenger class.
- Created visualizations using Seaborn and Matplotlib.
- Generated a correlation heatmap for numerical features.

### 2. Data Preprocessing

- Separated input features (X) and target variable (y).
- Removed unnecessary columns such as PassengerId, Name, Ticket, and Cabin from model inputs.
- Handled missing numerical values using median imputation.
- Applied StandardScaler to numerical features.
- Handled missing categorical values using most frequent value imputation.
- Applied OneHotEncoder for categorical features.
- Created a preprocessing pipeline using ColumnTransformer and Pipeline.

### 3. Train and Validation Split

- Split the training dataset into training and validation sets.
- Used an 80:20 split.
- Applied stratified sampling to maintain the target class distribution.
- Used random_state = 42 for reproducibility.

### 4. Machine Learning Models

Implemented and compared four classification algorithms:

- XGBoost Classifier
- CatBoost Classifier
- LightGBM Classifier
- Random Forest Classifier

CatBoost was trained using the original numerical and categorical features, while the other three models used the preprocessing pipeline.

### 5. Model Evaluation

Evaluated model performance using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

## Model Performance

The following results were obtained from the validation dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CatBoost | 82.68% | 85.19% | 66.67% | 74.80% | 84.84% |
| XGBoost | 79.89% | 80.00% | 63.77% | 70.97% | 81.93% |
| Random Forest | 79.89% | 74.63% | 72.46% | 73.53% | 84.32% |
| LightGBM | 78.21% | 76.79% | 62.32% | 68.80% | 82.81% |

**Best Performing Model: CatBoost Classifier**

CatBoost achieved the highest validation accuracy of **82.68%** and ROC-AUC of **84.84%** among the evaluated models.

Therefore, CatBoost was selected for final model training.

### 6. Final Model Training and Saving

- Retrained the selected CatBoost model using all 891 labelled training records.
- Used 500 iterations, depth 4, and learning rate 0.03.
- Saved the trained model as `best_titanic_model.cbm`.
- Verified that the model file was successfully saved.

### 7. Final Predictions

- Loaded the saved CatBoost model.
- Generated survival predictions for all 418 test passengers.
- Created a prediction summary.
- Exported the final predictions to `titanic_predictions.csv`.

**Prediction Summary:**

| Prediction | Number of Passengers |
|---|---:|
| Survived | 139 |
| Not Survived | 279 |
| Total | 418 |

The exported CSV contains `PassengerId` and `Survived` columns.

These are predicted labels, not verified survival outcomes, because the test dataset does not contain actual survival labels.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- LightGBM
- Google Colab / Jupyter Notebook

## Project Structure

```text
Task1_Titanic_Survival/
│
├── README.md
├── Task1_Titanic_Survival_devadharshini.ipynb
├── train.csv
├── test.csv
├── best_titanic_model.cbm
└── titanic_predictions.csv
```

## How to Run the Project

1. Clone or download the GitHub repository.
2. Open the Titanic project notebook in Google Colab or Jupyter Notebook.
3. Ensure the training and testing CSV files are available at the dataset paths specified in the notebook.
4. Install the required libraries if necessary:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm
```

5. Execute the notebook cells sequentially to perform data preprocessing, model training, evaluation, and prediction generation.

## Conclusion

This project demonstrates an end-to-end Machine Learning classification workflow using the Titanic dataset. Four classification algorithms were implemented and compared, with CatBoost achieving the best validation accuracy of 82.68%.

The selected model was retrained using all labelled training records and successfully generated survival predictions for 418 test passengers.

The project provided practical experience in data exploration, preprocessing, feature engineering, classification algorithms, model evaluation, and prediction generation.
