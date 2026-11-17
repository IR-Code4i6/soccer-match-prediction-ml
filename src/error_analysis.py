"""Error analysis for football match outcome prediction.

Compare the original Random Forest with the tuned depth-2 forest using the
approved Group C, 10-match history features. Analyze TRAIN and VALIDATION only;
the held-out test set is reserved for final evaluation.

Run from the project root:
    python -m src.error_analysis

Outputs are written to results/error_analysis/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data_pipeline import get_feature_columns, prepare_data
from src.random_forest import build_random_forest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "error_analysis"
LABELS = ["H", "D", "A"]
LABEL_NAMES = ["Home win", "Draw", "Away win"]
SEED = 42


def plot_confusion_matrix(matrix, title, output_path):
    """Save a count-based confusion matrix with explicit class labels."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of matches")
    ax.set_xticks(range(len(LABELS)), LABEL_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(len(LABELS)), LABEL_NAMES)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            count = int(matrix[row, col])
            ax.text(col, row, str(count), ha="center", va="center",
                    color="white" if count > threshold else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_split(model, frame, features, model_name, split_name):
    """Return summary metrics, per-class metrics, and prediction-level records."""
    y_true = frame["FTR"]
    y_pred = model.predict(frame[features])
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    summary = {
        "model": model_name,
        "split": split_name,
        "samples": len(frame),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS,
                             average="macro", zero_division=0),
    }
    per_class = []
    for label, name in zip(LABELS, LABEL_NAMES):
        per_class.append({
            "model": model_name,
            "split": split_name,
            "class": label,
            "class_name": name,
            **{key: report[label][key] for key in
               ("precision", "recall", "f1-score", "support")},
        })

    records = frame[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    records = records.rename(columns={"FTR": "actual"})
    records["predicted"] = y_pred
    records["correct"] = records["actual"] == records["predicted"]
    records.insert(0, "model", model_name)
    records.insert(1, "split", split_name)
    return summary, per_class, records, matrix


def analyze_feature_importance(model, validation, features):
    """Measure importance on validation using macro F1, not test performance."""
    result = permutation_importance(
        model, validation[features], validation["FTR"],
        scoring="f1_macro", n_repeats=30, random_state=SEED, n_jobs=-1
    )
    table = pd.DataFrame({
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    table.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

    ordered = table.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, max(4, len(ordered) * 0.48)))
    ax.barh(ordered["feature"], ordered["importance_mean"],
            xerr=ordered["importance_std"], capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in validation macro F1 when feature is shuffled")
    ax.set_title("Tuned Random Forest: permutation importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "permutation_importance.png", dpi=160)
    plt.close(fig)
    return table


def run_error_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, validation, _test = prepare_data()  # Never inspect the test set here.
    features = get_feature_columns(train, group="C", window=10)

    configurations = [
        ("Random Forest (untuned)", build_random_forest(max_depth=None)),
        ("Random Forest (depth 2)", build_random_forest(max_depth=2)),
    ]
    summaries, class_rows, prediction_tables = [], [], []
    fitted_models = {}

    for model_name, model in configurations:
        model.fit(train[features], train["FTR"])
        fitted_models[model_name] = model
        for split_name, frame in (("train", train), ("validation", validation)):
            summary, classes, records, matrix = evaluate_split(
                model, frame, features, model_name, split_name
            )
            summaries.append(summary)
            class_rows.extend(classes)
            prediction_tables.append(records)
            safe_name = "tuned" if "depth 2" in model_name else "untuned"
            if split_name == "validation":
                plot_confusion_matrix(
                    matrix, f"{model_name}: validation confusion matrix",
                    OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"
                )
                pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
                    OUTPUT_DIR / f"confusion_matrix_{safe_name}.csv"
                )

    metrics = pd.DataFrame(summaries)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUTPUT_DIR / "class_metrics.csv", index=False)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "all_predictions.csv", index=False)
    predictions.loc[
        (predictions["split"] == "validation") & (~predictions["correct"])
        ].to_csv(OUTPUT_DIR / "validation_errors.csv", index=False)

    comparison = metrics.pivot(index="model", columns="split", values="macro_f1")
    comparison["train_validation_gap"] = comparison["train"] - comparison["validation"]
    comparison.to_csv(OUTPUT_DIR / "overfitting_comparison.csv")

    importance = analyze_feature_importance(
        fitted_models["Random Forest (depth 2)"], validation, features
    )

    print("\nModel performance (macro F1 and accuracy):")
    print(metrics.to_string(index=False))
    print("\nTrain-validation macro F1 gap:")
    print(comparison.to_string())
    print("\nValidation permutation importance (top features):")
    print(importance.head(8).to_string(index=False))
    print(f"\nSaved error analysis to: {OUTPUT_DIR}")
    return metrics, comparison, importance


if __name__ == "__main__":
    run_error_analysis()
