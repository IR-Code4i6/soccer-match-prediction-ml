
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

from src.data_pipeline import (
    prepare_data,
    get_feature_columns,
    get_recency_columns,
)
from src.random_forest import build_random_forest
from src.neural_network import build_neural_network

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_logistic():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def create_model(name):
    if name == "Majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "Logistic Regression":
        return build_logistic()
    if name == "Random Forest":
        return build_random_forest()
    if name == "MLP (32)":
        return build_neural_network((32,), use_pca=False)
    if name == "MLP (64,32)":
        return build_neural_network((64, 32), use_pca=False)
    if name == "MLP (32) + PCA":
        return build_neural_network((32,), use_pca=True)
    if name == "MLP (64,32) + PCA":
        return build_neural_network((64, 32), use_pca=True)
    raise ValueError(f"Unknown model: {name}")


def evaluate(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y, predictions,
            labels=["H", "D", "A"],
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=["H", "D", "A"]
        ).tolist(),
    }


def experiment_configurations(train):
    """Yield feature definitions for history windows and recency."""
    for group in ["A", "B", "C"]:
        for window in [3, 5, 10]:
            yield {
                "experiment": "history_window",
                "group": group,
                "window": window,
                "features": get_feature_columns(
                    train, group=group, window=window
                ),
            }

    yield {
        "experiment": "recency",
        "group": "C_recency",
        "window": "recent5_vs_older5",
        "features": get_recency_columns(),
    }


def run_experiments():
    train, validation, test = prepare_data()

    model_names = [
        "Majority",
        "Logistic Regression",
        "Random Forest",
        "MLP (32)",
        "MLP (64,32)",
        "MLP (32) + PCA",
        "MLP (64,32) + PCA",
    ]

    validation_rows = []

    # Compare models and feature configurations on validation data.
    for config in experiment_configurations(train):
        features = config["features"]
        X_train = train[features]
        y_train = train["FTR"]
        X_val = validation[features]
        y_val = validation["FTR"]

        for model_name in model_names:
            model = create_model(model_name)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_val, y_val)

            validation_rows.append({
                "experiment": config["experiment"],
                "feature_group": config["group"],
                "window": config["window"],
                "model": model_name,
                "feature_count": len(features),
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
            })

    validation_results = pd.DataFrame(validation_rows)
    validation_results.to_csv(
        RESULTS_DIR / "validation_results.csv", index=False
    )

    # Choose settings using validation macro F1 only.
    best = validation_results.sort_values(
        "validation_macro_f1", ascending=False
    ).iloc[0]

    if best["experiment"] == "recency":
        best_features = get_recency_columns()
    else:
        best_features = get_feature_columns(
            train,
            group=best["feature_group"],
            window=int(best["window"]),
        )

    best_model = create_model(best["model"])

    # Refit on training + validation after selecting the configuration.
    train_validation = pd.concat(
        [train, validation], ignore_index=True
    )
    best_model.fit(
        train_validation[best_features],
        train_validation["FTR"],
    )

    test_metrics = evaluate(
        best_model, test[best_features], test["FTR"]
    )

    final_result = {
        "experiment": best["experiment"],
        "feature_group": best["feature_group"],
        "window": best["window"],
        "model": best["model"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_confusion_matrix": str(test_metrics["confusion_matrix"]),
    }

    pd.DataFrame([final_result]).to_csv(
        RESULTS_DIR / "final_test_results.csv", index=False
    )

    print("\nBest validation configuration:")
    print(best.to_string())
    print("\nFinal test results:")
    print(pd.DataFrame([final_result]).to_string(index=False))
    print("\nConfusion matrix (rows/columns H, D, A):")
    print(test_metrics["confusion_matrix"])

    return validation_results, final_result


if __name__ == "__main__":
    run_experiments()
