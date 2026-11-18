
# Titanic Survival Prediction
APP LINK:  https://titanic-survival-intelligence.onrender.com




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

| Model | Accuracy | Precision | Recall | F1-Score | ROC
