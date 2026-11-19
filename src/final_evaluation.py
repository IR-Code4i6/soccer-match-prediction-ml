"""Final test evaluation of the default and tuned models on Group C, window 10.

Tuned settings were picked by the highest mean validation macro F1 over 5 seeds
in results/tuning_results.csv. Each model is refit on train + validation, the
same way experiments.py does it, then scored on the test set once per seed.
"""

import time
import warnings
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from src.data_pipeline import prepare_data, get_feature_columns
from src.experiments import create_model, evaluate
from src.tuning import build_random_forest, build_neural_network, SEEDS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_GROUP = "C"
WINDOW = 10

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODELS = [
    ("Majority", "baseline", lambda seed: create_model("Majority")),
    ("Logistic Regression", "baseline", lambda seed: create_model("Logistic Regression")),
    ("Random Forest", "default (max_depth=None)",
     lambda seed: build_random_forest(max_depth=None, random_state=seed)),
    ("Random Forest", "tuned (max_depth=2)",
     lambda seed: build_random_forest(max_depth=2, random_state=seed)),
    ("MLP (32)", "default (alpha=0.001)",
     lambda seed: build_neural_network((32,), alpha=0.001, random_state=seed)),
    ("MLP (32)", "tuned (alpha=10)",
     lambda seed: build_neural_network((32,), alpha=10, random_state=seed)),
    ("MLP (32) + PCA", "default (95% variance)",
     lambda seed: build_neural_network((32,), use_pca=True, pca_variance=0.95,
                                       random_state=seed)),
    ("MLP (32) + PCA", "tuned (90% variance)",
     lambda seed: build_neural_network((32,), use_pca=True, pca_variance=0.90,
                                       random_state=seed)),
]


def mean_sd_ci(values):
    """Mean, sample SD and a 95% t-interval for a handful of seed scores."""
    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / n ** 0.5
    return mean, sd, mean - half, mean + half


def run_final_evaluation():
    train, validation, test = prepare_data()
    features = get_feature_columns(train, group=FEATURE_GROUP, window=WINDOW)

    train_validation = pd.concat([train, validation], ignore_index=True)
    X_fit, y_fit = train_validation[features], train_validation["FTR"]
    X_test, y_test = test[features], test["FTR"]

    rows = []
    for model_name, setting, build in MODELS:
        for seed in SEEDS:
            model = build(seed)
            start = time.perf_counter()
            model.fit(X_fit, y_fit)
            train_seconds = time.perf_counter() - start

            metrics = evaluate(model, X_test, y_test)
            rows.append({
                "model": model_name,
                "setting": setting,
                "seed": seed,
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "train_seconds": train_seconds,
                "confusion_matrix_HDA": str(metrics["confusion_matrix"]),
            })

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS_DIR / "final_evaluation_per_seed.csv", index=False)

    summary_rows = []
    for (model_name, setting), group in per_seed.groupby(["model", "setting"], sort=False):
        acc = mean_sd_ci(group["test_accuracy"])
        f1 = mean_sd_ci(group["test_macro_f1"])
        summary_rows.append({
            "model": model_name,
            "setting": setting,
            "accuracy_mean": acc[0], "accuracy_sd": acc[1],
            "accuracy_ci_low": acc[2], "accuracy_ci_high": acc[3],
            "macro_f1_mean": f1[0], "macro_f1_sd": f1[1],
            "macro_f1_ci_low": f1[2], "macro_f1_ci_high": f1[3],
            "train_seconds_mean": group["train_seconds"].mean(),
        })

    summary = pd.DataFrame(summary_rows).round(3)
    summary.to_csv(RESULTS_DIR / "final_evaluation.csv", index=False)

    print(f"Fit on train + validation ({len(train_validation)} matches), "
          f"tested on {len(test)} matches, seeds {SEEDS}\n")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_final_evaluation()
