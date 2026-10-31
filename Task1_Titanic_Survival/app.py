

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

# Load the saved CatBoost model
model = CatBoostClassifier()
model_path = os.path.join(os.path.dirname(__file__), "best_titanic_model.cbm")
model.load_model(model_path)

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked"]


def prepare_passenger(form):
    passenger = {
        "Pclass": str(form.get("pclass", "3")),
        "Sex": str(form.get("sex", "female")),
        "Age": float(form.get("age", 25)),
        "SibSp": int(form.get("sibsp", 0)),
        "Parch": int(form.get("parch", 0)),
        "Fare": float(form.get("fare", 20)),
        "Embarked": str(form.get("embarked", "S"))
    }

    return pd.DataFrame([passenger], columns=FEATURES)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    passenger_values = {
        "age": 25,
        "sex": "female",
        "pclass": "3",
        "fare": 20,
        "sibsp": 0,
        "parch": 0,
        "embarked": "S"
    }

    if request.method == "POST":
        try:
            passenger_values = {
                "age": request.form.get("age", "25"),
                "sex": request.form.get("sex", "female"),
                "pclass": request.form.get("pclass", "3"),
                "fare": request.form.get("fare", "20"),
                "sibsp": request.form.get("sibsp", "0"),
                "parch": request.form.get("parch", "0"),
                "embarked": request.form.get("embarked", "S")
            }

            X = prepare_passenger(request.form)

            # Model prediction
            probabilities = model.predict_proba(X)[0]
            survival_probability = float(probabilities[1]) * 100
            death_probability = float(probabilities[0]) * 100

            prediction = int(model.predict(X)[0].item())
            prediction_label = (
                "Survival Predicted" if prediction == 1
                else "Survival Not Predicted"
            )

            # SHAP values from CatBoost
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
                    "direction": "positive" if value >= 0 else "negative"
                })

            explanations.sort(
                key=lambda item: abs(item["value"]),
                reverse=True
            )

            result = {
                "prediction": prediction_label,
                "survival": round(survival_probability, 1),
                "not_survival": round(death_probability, 1),
                "explanations": explanations
            }

        except Exception as error:
            result = {"error": str(error)}

    return render_template(
        "index.html",
        result=result,
        values=passenger_values
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
