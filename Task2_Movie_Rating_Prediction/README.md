# 🎬 Task 2: Movie Rating Prediction with Python

**CodSoft Data Science Internship**

## 📌 Objective
Build a regression model that predicts the rating of a movie from features such as **genre, director and actors**, and find out which factors influence movie ratings the most.

## 📂 Dataset
`IMDb Movies India.csv`: IMDb data on Indian movies.

- Total records: **15,509 movies**
- Movies with a rating (used for modeling): **7,919**
- Columns: Name, Year, Duration, Genre, Rating, Votes, Director, Actor 1, Actor 2, Actor 3
- Target variable: **Rating**

## 🛠️ Project Workflow

### 1. Data Exploration
- Viewed the data with `head()`, `info()` and `describe()`
- Checked missing values (Duration 53%, Rating 49%, Votes 49%, and others)
- Checked duplicates (6 exact duplicate rows found)

### 2. Data Cleaning and Preprocessing
- Removed duplicate rows
- Dropped rows with no rating (the target) or no year
- Converted `Year` (e.g. "(2019)"), `Duration` (e.g. "109 min") and `Votes` to numbers
- Filled missing `Duration` and `Votes` with the median
- Filled missing `Genre`, `Director` and actors with "Unknown"

### 3. Data Visualization (5 plots)
1. Distribution of movie ratings
2. Average rating by decade
3. Top genres by count and by average rating
4. Top directors by average rating
5. Correlation heatmap and Votes vs Rating scatter plot

### 4. Feature Engineering
- **Genre:** multi-hot encoding (one 0/1 column per genre)
- **Director and actors:** smoothed target encoding (the average rating of the person's other movies), computed on the training data only to avoid data leakage
- **Votes:** log transformation
- Final feature set: **28 features** (6,335 training rows, 1,584 test rows, 80/20 split)

### 5. Model Building
Three regression models were trained and compared:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### 6. Evaluation
Metrics used: **MAE, RMSE, R²**

## 📊 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting** | **0.785** | **1.034**
