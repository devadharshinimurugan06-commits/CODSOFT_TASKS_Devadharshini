# Movie Rating Prediction with Python

**CodSoft Data Science Internship – Task 2**  
**Developed by:** Devadharshini M

## Project Overview

Movie Rating Prediction is a Machine Learning regression project that predicts movie ratings using historical movie information.

The project analyzes different factors that may influence movie ratings, including genre, director, actors, release year, movie duration, and number of votes.

The main objective is to perform data analysis, preprocessing, feature engineering, and regression modeling to estimate movie ratings based on historical IMDb data.

## Dataset

**Dataset Name:** IMDb Movies India.csv

The dataset contains information about Indian movies, including:

- Movie Name
- Year
- Duration
- Genre
- Rating
- Votes
- Director
- Actor 1
- Actor 2
- Actor 3

**Original Dataset Size:**
- 15,509 rows
- 10 columns

**After Data Cleaning:**
- 7,919 rows
- 10 columns

The target variable is `Rating`.

## Project Workflow

### 1. Data Loading and Exploration

- Loaded the IMDb Movies India dataset using Pandas.
- Used Latin-1 encoding to read the dataset.
- Examined the first and last records.
- Checked dataset dimensions and column information.
- Generated descriptive statistical summaries.
- Identified missing values and their percentages.
- Checked duplicate records and duplicate movie names.

### 2. Data Cleaning and Preprocessing

- Removed 6 exact duplicate records.
- Removed records with missing movie ratings.
- Removed records with missing release years.
- Extracted numerical values from Year and Duration columns.
- Converted Votes into numerical format by removing commas.
- Handled missing Duration and Votes using median imputation.
- Replaced missing Genre, Director, and Actor information with `Unknown`.
- Converted Year into integer format.
- Verified that the cleaned dataset contained no remaining missing values or exact duplicate rows.

**Final Cleaned Dataset:** 7,919 rows × 10 columns.

### 3. Exploratory Data Analysis

Performed exploratory data analysis to understand movie rating patterns.

Created visualizations for:

- Distribution of movie ratings.
- Average movie ratings by decade.
- Top 10 genres based on movie counts.
- Top-rated genres with at least 30 movies.
- Top 10 directors by average rating with at least 5 movies.
- Correlation heatmap for Year, Duration, Votes, and Rating.
- Relationship between movie votes and ratings.

### 4. Feature Engineering

Applied feature engineering techniques to convert movie information into numerical features.

- Performed multi-hot encoding on Genre.
- Retained genre categories appearing at least 50 times.
- Generated 19 genre feature columns.
- Applied logarithmic transformation to Votes using `log1p`.
- Applied smoothed target encoding to Director.
- Applied target encoding to Actor 1, Actor 2, and Actor 3.
- Generated aggregate actor features:
  - `Actor_Enc_Mean`
  - `Actor_Enc_Max`

Used an 80:20 train-test split with `random_state=42`.

**Dataset Split:**
- Training records: 6,335
- Testing records: 1,584

### 5. Machine Learning Models

Implemented and compared three regression algorithms:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

### 6. Model Evaluation

Evaluated the regression models using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Also performed:

- Random Forest feature importance visualization.
- Actual vs Predicted rating comparison.
- Sample movie rating predictions.
- Absolute prediction error analysis.

## Model Performance

The following results were obtained from the test dataset:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 0.785 | 1.034 | 0.425 |
| Random Forest | 0.794 | 1.045 | 0.412 |
| Linear Regression | 0.848 | 1.103 | 0.345 |

**Best Performing Model: Gradient Boosting Regressor**

Gradient Boosting achieved:

- MAE: 0.785
- RMSE: 1.034
- R²: 0.425

It achieved the lowest MAE and RMSE and the highest R² among the three evaluated models.

### 7. Additional Experiment – Without Votes

An additional experiment was performed using Random Forest without the `Log_Votes` feature.

This experiment helps examine model performance when vote-count information is excluded.

| Metric | Random Forest Without Votes |
|---|---:|
| MAE | 0.857 |
| RMSE | 1.117 |
| R² | 0.328 |

The model achieved an R² of 0.328 without Votes, compared with 0.412 for the main Random Forest model.

### 8. Prediction and Visualization

- Generated predicted movie ratings using the trained Gradient Boosting model.
- Compared actual and predicted ratings.
- Calculated absolute prediction errors.
- Displayed sample movie predictions with actual ratings and predicted ratings.
- Visualized Random Forest feature importance to understand influential features.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## Project Structure

```text
Task2_Movie_Rating_Prediction/
│
├── README.md
├── Task2_Movie_Prediction_Devadharshini.ipynb
└── IMDb Movies India.csv
```

## How to Run the Project

1. Clone or download the GitHub repository.
2. Open the Movie Rating Prediction notebook in Google Colab or Jupyter Notebook.
3. Ensure `IMDb Movies India.csv` is available at the dataset path specified in the notebook.
4. Install the required libraries if necessary:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

5. Execute the notebook cells sequentially to perform data cleaning, exploratory analysis, feature engineering, model training, evaluation, and prediction.

## Limitations

- Movie ratings are subjective and can depend on factors not included in the dataset.
- The model's performance depends on the quality and availability of historical movie information.
- Votes may not be available before a movie is released.
- The experiment without Votes provides an additional comparison, but it does not represent a complete evaluation of every possible pre-release prediction scenario.

## Conclusion

This project demonstrates an end-to-end Machine Learning regression workflow using the IMDb Movies India dataset.

The dataset was cleaned and analyzed, and features were engineered using genre, director, actors, year, duration, and votes.

Three regression algorithms were implemented and evaluated. Gradient Boosting achieved the best performance with an MAE of 0.785, RMSE of 1.034, and R² of 0.425.

The project provided practical experience in data preprocessing, exploratory data analysis, feature engineering, regression modeling, performance evaluation, and movie rating prediction.
