# 🎬 Task 2: Movie Rating Prediction with Python

**CodSoft Data Science Internship** · #codsoft

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
- Viewed the data with `head()`, `tail()`, `info()` and `describe()`
- Missing values: Duration (53.3%), Rating (48.9%), Votes (48.9%), Actor 3 (20.3%), Actor 2 (15.4%), Genre (12.1%), Actor 1 (10.4%), Year (3.4%), Director (3.4%)
- Duplicates: 6 exact duplicate rows found and removed

### 2. Data Cleaning and Preprocessing
- Dropped rows with no rating (the target) or no year
- Converted `Year` (e.g. "(2019)"), `Duration` (e.g. "109 min") and `Votes` to numbers
- Filled missing `Duration` and `Votes` with the median
- Filled missing `Genre`, `Director` and actors with "Unknown"
- Final clean data: **7,919 rows, 0 missing values, 0 duplicates**

### 3. Data Visualization (5 plots)
1. Distribution of movie ratings
2. Average rating by decade
3. Top genres by count and by average rating
4. Top directors by average rating
5. Correlation heatmap and Votes vs Rating scatter plot

### 4. Feature Engineering
- **Genre:** multi-hot encoding (19 genre columns)
- **Director and actors:** smoothed target encoding (the average rating of the person's other movies), computed on training data only to avoid data leakage
- **Votes:** log transformation
- Final feature set: **28 features**
- 80/20 train-test split: **6,335 training rows, 1,584 test rows**

### 5. Model Building
Three regression models were trained and compared:
- Linear Regression
- Random Forest Regressor (200 trees)
- Gradient Boosting Regressor (200 trees)

### 6. Evaluation
Metrics used: **MAE, RMSE, R2 Score**

## 📊 Results

| Model | MAE | RMSE | R2 Score |
|---|---|---|---|
| **Gradient Boosting (best)** | **0.785** | **1.034** | **0.425** |
| Random Forest | 0.794 | 1.045 | 0.412 |
| Linear Regression | 0.848 | 1.103 | 0.345 |

**Without the Votes feature** (a pre-release scenario), Random Forest scored MAE 0.857, RMSE 1.117 and R2 0.328.

The best model's R2 of 0.425 means it explains about 42.5% of the variation in movie ratings, and its predictions are off by about 0.8 rating points on average.

### Sample predictions (test set, Gradient Boosting)

| Movie | Actual | Predicted |
|---|---|---|
| Veerappan | 5.3 | 5.4 |
| Commando 3 | 5.7 | 5.1 |
| Fashion | 6.9 | 6.4 |
| Firaaq | 7.4 | 6.8 |
| Main Suhagan Hoon | 5.2 | 5.8 |

## 🔍 Key Insights
- The most influential factors are the **actors' past average ratings**, **number of votes**, **release year** and the **director's track record**.
- **Genre** has the smallest effect on the rating.
- Tree-based models beat Linear Regression because ratings depend on combinations of features, not a straight line.
- The model tends to predict toward the middle (5 to 7) and is weaker on very high or very low rated movies.

## ⚠️ Limitations
- Ratings are subjective and depend on things not in the data (story, music, marketing, budget).
- `Votes` is only known after a movie is released.
- Many movies had missing duration or actor information.

## 🚀 Future Improvements
- Add budget, box office and cast popularity data
- Add text features from plots or reviews
- Try XGBoost / LightGBM and hyperparameter tuning

## 🧰 Tools and Libraries
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Google Colab

## 📁 Files
- `Task2_Movie_Prediction_Devadharshini.ipynb`: complete project notebook
- `IMDb Movies India.csv`: dataset

## 👩‍💻 Author
**Devadharshini M**
CodSoft Data Science Internship
