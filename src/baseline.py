
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import prepare_data, get_feature_columns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["H", "D", "A"]


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
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision_score(
            y_test, predictions,
            labels=LABELS, average="macro", zero_division=0,
        ),
        "macro_recall": recall_score(
            y_test, predictions,
            labels=LABELS, average="macro", zero_division=0,
        ),
        "macro_f1": f1_score(
            y_test, predictions,
            labels=LABELS, average="macro", zero_division=0,
        ),
        "confusion_matrix": str(
            confusion_matrix(y_test, predictions, labels=LABELS).tolist()
        ),
        "predictions": predictions,
    }


def save_dataset_summary(train, validation, test):
    rows = []

    for split_name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        counts = split["FTR"].value_counts()

        for label in LABELS:
            count = int(counts.get(label, 0))
            rows.append({
                "split": split_name,
                "samples": len(split),
                "class": label,
                "class_count": count,
                "class_percentage": (
                    count / len(split) * 100 if len(split) else 0
                ),
                "start_date": split["Date"].min(),
                "end_date": split["Date"].max(),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "baseline_data_summary.csv", index=False)
    print("\nDataset summary:")
    print(summary.to_string(index=False))


def run_baselines():
    train, validation, test = prepare_data()

    # Keep the approved Group C five match baseline configuration
    features = get_feature_columns(train, group="C", window=5)

    X_train = train[features]
    y_train = train["FTR"]
    X_test = test[features]
    y_test = test["FTR"]

    models = [
        (
            "Majority-class benchmark",
            DummyClassifier(strategy="most_frequent"),
        ),
        ("Logistic Regression", make_logistic()),
    ]

    results = []
    reports = []

    for name, model in models:
        result = evaluate(
            name, model, X_train, y_train, X_test, y_test
        )

        predictions = result.pop("predictions")
        results.append(result)

        report = classification_report(
            y_test,
            predictions,
            labels=LABELS,
            target_names=["Home win (H)", "Draw (D)", "Away win (A)"],
            output_dict=True,
            zero_division=0,
        )

        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                reports.append({
                    "model": name,
                    "class": class_name,
                    **metrics,
                })

    table = pd.DataFrame(results)
    table.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

    pd.DataFrame(reports).to_csv(
        RESULTS_DIR / "baseline_classification_report.csv",
        index=False,
        )

    save_dataset_summary(train, validation, test)

    print("\nFeatures:", features)
    print("Feature count:", len(features))
    print("\nBaseline results:")
    print(table.to_string(index=False))

    return table


if __name__ == "__main__":
    run_baselines()
