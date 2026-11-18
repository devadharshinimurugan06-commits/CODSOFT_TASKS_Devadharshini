# 📈 Sales Prediction using Python

**CodSoft Data Science Internship | Sales Prediction Task**

A machine learning project that predicts product sales from advertising spend on **TV**, **Radio** and **Newspaper**, with a live Streamlit web app for testing budgets.

🔗 **Live app:** [https://codsofttasksdevadharshini-j65b5cmgxntpms8nwhhbm7.streamlit.app/](https://codsofttasksdevadharshini-j65b5cmgxntpms8nwhhbm7.streamlit.app/)

---

## 📌 Problem Statement

Sales prediction means forecasting how much of a product customers will buy, based on factors such as advertising expenditure. By predicting sales from ad budgets, a business can decide where to spend its advertising money to get the most sales.

**Goal:** Build a model that predicts `Sales` from `TV`, `Radio` and `Newspaper` advertising spend, and turn it into an app that helps plan budgets.

---

## 📂 Dataset

The dataset is `CodSoft_Sales_Prediction_Dataset.xlsx`, provided by CodSoft.

| Column | Description |
|---|---|
| `TV` | Advertising spend on TV |
| `Radio` | Advertising spend on Radio |
| `Newspaper` | Advertising spend on Newspaper |
| `Sales` | Sales achieved (target) |

- **Rows:** 200
- **Columns:** 4
- **Missing values:** none
- **Duplicate rows:** none

---

## 🔄 Project Workflow

1. **Load data** from the Excel file into pandas
2. **Data checks:** shape, data types, missing values, duplicates, summary statistics
3. **Exploratory data analysis:** distributions, scatter plots with regression lines, correlation heatmap, box plots
4. **Train/test split:** 80% training, 20% testing (`random_state=42`)
5. **Model training:** Linear Regression and Random Forest Regressor
6. **Evaluation:** MAE, RMSE and R² on the test set
7. **Insights:** feature importance and regression coefficients
8. **Save the model** with `joblib` as `sales_model.pkl`
9. **Deploy** as a Streamlit web app

---

## 🔍 Key Findings

**Correlation with Sales**

| Channel | Correlation |
|---|---|
| TV | 0.78 |
| Radio | 0.58 |
| Newspaper | 0.23 |

- **TV** has the strongest relationship with sales.
- **Radio** is the second strongest.
- **Newspaper** has a weak relationship with sales.

**Feature importance (Random Forest)**

| Channel | Importance |
|---|---|
| TV | ~62.5% |
| Radio | ~36.2% |
| Newspaper | ~1.3% |

**Linear Regression coefficients** (change in sales per 1 unit of spend)

| Channel | Coefficient |
|---|---|
| TV | 0.045 |
| Radio | 0.189 |
| Newspaper | 0.003 |

**Business takeaway:** Focus the advertising budget on TV and Radio. Newspaper spend adds almost nothing to sales in this data.

---

## 🤖 Model Performance

Results on the 20% test set:

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 1.46 | 1.78 | 0.899 |
| **Random Forest** | **0.63** | **0.76** | **0.982** |

**Random Forest was selected as the final model**, since it has the lowest error and the highest R² score. It explains about 98% of the variation in sales on unseen data.

---

## 🌐 Streamlit App

The app loads the trained Random Forest model and lets you test ad budgets. It has four tabs:

| Tab | What it does |
|---|---|
| **Prediction** | Set TV, Radio and Newspaper spend with sliders to get the expected sales, a likely range, total spend, sales per unit of spend and a budget split chart |
| **What-if analysis** | Shows how predicted sales change as one channel moves while the other two stay fixed |
| **Budget optimizer** | Enter a total budget and the app finds the TV, Radio and Newspaper split with the highest predicted sales |
| **About the model** | Feature importance, test scores and model details |

The interface uses an olive-green, glass-style design.

**Try it here:** [Open the live app](https://codsofttasksdevadharshini-j65b5cmgxntpms8nwhhbm7.streamlit.app/)

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** scikit-learn
- **Model saving:** joblib
- **Web app:** Streamlit
- **Notebook environment:** Google Colab

---

## 📁 Repository Structure

```
Task3_Sales_Prediction/
│
├── App.py                                  # Streamlit app
├── sales_model.pkl                         # Trained Random Forest model
├── CodSoft_Sales_Prediction_Dataset.xlsx   # Dataset
├── requirements.txt                        # Python dependencies
├── .streamlit/
│   └── config.toml                         # App theme settings
├── <your_notebook_name>.ipynb              # Google Colab notebook
└── README.md
```

---

## ▶️ Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/devadharshinimurugan06-commits/CODSOFT_TASKS_Devadharshini.git
   cd CODSOFT_TASKS_Devadharshini/Task3_Sales_Prediction
   ```

2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the app**
   ```bash
   streamlit run App.py
   ```

The app opens at `http://localhost:8501`.

> **Note:** The model was saved with `scikit-learn==1.6.1`. Use the same version (it is pinned in `requirements.txt`) to avoid version warnings or wrong results.

---

## ⚠️ Limitations

- A Random Forest cannot predict beyond the spend ranges it was trained on (TV 0 to 300, Radio 0 to 50, Newspaper 0 to 120), so the app sliders stay inside these ranges.
- The dataset is small (200 records), so results show relationships in past data and are not guaranteed outcomes.
- The model uses only advertising spend and does not account for season, pricing, competition or audience.

---

## 🚀 Future Improvements

- Hyperparameter tuning with cross-validation
- Compare more models (Gradient Boosting, XGBoost)
- Add more features such as audience segment and platform
- Let users upload their own data inside the app

---

## 👩‍💻 Author

**Devadharshini**


---

#codsoft #datascience #machinelearning #python #streamlit #internship
