
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import prepare_data, get_feature_columns

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_logistic():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions,
            labels=["H", "D", "A"],
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["H", "D", "A"]
        ).tolist(),
    }


def run_baselines():
    train, validation, test = prepare_data()

    # Baseline uses the memo's Group C, five-match feature configuration.
    features = get_feature_columns(train, group="C", window=5)

    X_train, y_train = train[features], train["FTR"]
    X_test, y_test = test[features], test["FTR"]

    models = [
        ("Majority-class benchmark",
         DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", make_logistic()),
    ]

    results = [
        evaluate(name, model, X_train, y_train, X_test, y_test)
        for name, model in models
    ]

    table = pd.DataFrame([
        {
            "model": r["model"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "confusion_matrix": str(r["confusion_matrix"]),
        }
        for r in results
    ])

    table.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    print(table.to_string(index=False))

    return results


if __name__ == "__main__":
    run_baselines()
