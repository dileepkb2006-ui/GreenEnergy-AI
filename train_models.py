
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train_regressor(filename, target, model_name):
    df = pd.read_csv(DATA_DIR / filename)
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    joblib.dump(
        {"model": model, "features": list(X.columns)},
        MODEL_DIR / f"{model_name}.joblib"
    )

    mae = mean_absolute_error(y_test, predictions)
    print(f"{model_name}: MAE = {mae:.2f}")

    return {"model": model_name, "mae": round(float(mae), 2)}

def train_classifier():
    df = pd.read_csv(DATA_DIR / "source_recommendation.csv")
    target = "recommended_source"
    X = df.drop(columns=[target])
    y = df[target]

    # Avoid a small-dataset stratification failure.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    joblib.dump(
        {"model": model, "features": list(X.columns)},
        MODEL_DIR / "source_recommendation.joblib"
    )

    accuracy = accuracy_score(y_test, predictions)
    print(f"source_recommendation: Accuracy = {accuracy:.2%}")

    return {"model": "source_recommendation",
            "accuracy": round(float(accuracy), 4)}

def main():
    results = []

    results.append(train_regressor(
        "solar_generation.csv", "energy_kwh", "solar_generation"
    ))
    results.append(train_regressor(
        "wind_generation.csv", "energy_kwh", "wind_generation"
    ))
    results.append(train_regressor(
        "energy_consumption.csv", "demand_kwh", "energy_consumption"
    ))
    results.append(train_classifier())

    with open(MODEL_DIR / "training_metrics.json", "w") as file:
        json.dump(results, file, indent=4)

    print("\nTraining complete. Models saved in the models folder.")

if __name__ == "__main__":
    main()
