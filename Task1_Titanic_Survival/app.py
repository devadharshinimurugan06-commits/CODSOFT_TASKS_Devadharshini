
from flask import Flask, render_template, request
from catboost import CatBoostClassifier, Pool
import pandas as pd
import os

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static"
)

# ---------------------------------------
# LOAD CATBOOST MODEL
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = CatBoostClassifier()
model_path = os.path.join(BASE_DIR, "best_titanic_model.cbm")
model.load_model(model_path)

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

CAT_FEATURES = ["Pclass", "Sex", "Embarked"]

DEFAULT_VALUES = {
    "age": 25,
    "sex": "female",
    "pclass": "3",
    "fare": 20,
    "sibsp": 0,
    "parch": 0,
    "embarked": "S"
}


# ---------------------------------------
# PREPARE PASSENGER DATA
# ---------------------------------------

def prepare_passenger(data, prefix=""):
    passenger = {
        "Pclass": str(data.get(prefix + "pclass", "3")),
        "Sex": str(data.get(prefix + "sex", "female")),
        "Age": float(data.get(prefix + "age", 25)),
        "SibSp": int(data.get(prefix + "sibsp", 0)),
        "Parch": int(data.get(prefix + "parch", 0)),
        "Fare": float(data.get(prefix + "fare", 20)),
        "Embarked": str(data.get(prefix + "embarked", "S"))
    }

    return pd.DataFrame([passenger], columns=FEATURES)


# ---------------------------------------
# MODEL PREDICTION
# ---------------------------------------

def predict_passenger(X):
    probabilities = model.predict_proba(X)[0]

    survival_probability = float(probabilities[1]) * 100
    death_probability = float(probabilities[0]) * 100

    prediction = int(model.predict(X)[0].item())

    prediction_label = (
        "Survival Predicted"
        if prediction == 1
        else "Survival Not Predicted"
    )

    # SHAP explanation
    pool = Pool(X, cat_features=CAT_FEATURES)

    shap_values = model.get_feature_importance(
        data=pool,
        type="ShapValues"
    )[0]

    explanations = []

    for feature, value in zip(FEATURES, shap_values[:-1]):
        explanations.append({
            "feature": feature,
            "value": round(float(value), 4),
            "direction": (
                "positive" if value >= 0 else "negative"
            )
        })

    explanations.sort(
        key=lambda item: abs(item["value"]),
        reverse=True
    )

    return {
        "prediction": prediction_label,
        "survival": round(survival_probability, 1),
        "not_survival": round(death_probability, 1),
        "explanations": explanations,
        "top_explanations": explanations[:3]
    }


# ---------------------------------------
# HISTORICAL DATASET INSIGHTS
# ---------------------------------------

def get_historical_insights():
    train_path = os.path.join(BASE_DIR, "train.csv")

    if not os.path.exists(train_path):
        return None

    try:
        df = pd.read_csv(train_path)

        total = len(df)
        survived = int(df["Survived"].sum())
        not_survived = total - survived

        survival_rate = round(
            (survived / total) * 100, 1
        )

        female_rate = round(
            df.loc[df["Sex"] == "female", "Survived"].mean() * 100,
            1
        )

        male_rate = round(
            df.loc[df["Sex"] == "male", "Survived"].mean() * 100,
            1
        )

        class_rates = (
            df.groupby("Pclass")["Survived"]
            .mean()
            .mul(100)
            .round(1)
            .to_dict()
        )

        return {
            "total": total,
            "survived": survived,
            "not_survived": not_survived,
            "survival_rate": survival_rate,
            "female_rate": female_rate,
            "male_rate": male_rate,
            "class_rates": class_rates
        }

    except Exception as error:
        print("Historical insights error:", error)
        return None


# ---------------------------------------
# HOME PAGE
# ---------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    comparison = None
    values = DEFAULT_VALUES.copy()

    historical = get_historical_insights()

    if request.method == "POST":

        action = request.form.get("action", "predict")

        try:

            # -----------------------------------
            # NORMAL PASSENGER PREDICTION
            # -----------------------------------

            if action == "predict":

                values = {
                    "age": request.form.get("age", "25"),
                    "sex": request.form.get("sex", "female"),
                    "pclass": request.form.get("pclass", "3"),
                    "fare": request.form.get("fare", "20"),
                    "sibsp": request.form.get("sibsp", "0"),
                    "parch": request.form.get("parch", "0"),
                    "embarked": request.form.get("embarked", "S")
                }

                X = prepare_passenger(request.form)

                result = predict_passenger(X)

            # -----------------------------------
            # WHAT-IF COMPARISON
            # -----------------------------------

            elif action == "compare":

                passenger_a = prepare_passenger(
                    request.form,
                    prefix="scenario_a_"
                )

                passenger_b = prepare_passenger(
                    request.form,
                    prefix="scenario_b_"
                )

                prediction_a = predict_passenger(passenger_a)
                prediction_b = predict_passenger(passenger_b)

                comparison = {
                    "scenario_a": prediction_a,
                    "scenario_b": prediction_b
                }

        except (ValueError, TypeError, KeyError) as error:
            result = {
                "error": "Please enter valid passenger details. " + str(error)
            }

        except Exception as error:
            result = {
                "error": str(error)
            }

    return render_template(
        "index.html",
        result=result,
        values=values,
        comparison=comparison,
        historical=historical
    )


# ---------------------------------------
# RUN APPLICATION
# ---------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
