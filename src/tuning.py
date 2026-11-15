"""Tests model settings, PCA variance and random seeds on Group C, window 10.

Only train and validation scores are used here. The test set is saved for
the final evaluation.
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import prepare_data, get_feature_columns
from src.experiments import evaluate

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_GROUP = "C"
WINDOW = 10
SEEDS = [42, 0, 1, 2, 3]

# The MLPs often stop at max_iter=800 before converging, which floods the output
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def build_random_forest(max_depth=None, min_samples_leaf=1, random_state=42):
    """Matches the team's forest in src/random_forest.py, so keep them in sync."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def build_neural_network(hidden_layers=(32,), use_pca=False, alpha=0.001,
                         pca_variance=0.95, random_state=42):
    """Matches the team's MLP in src/neural_network.py, so keep them in sync."""
    steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))
    steps.append(("model", MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=0.001,
        max_iter=800,
        early_stopping=False,
        random_state=random_state,
    )))
    return Pipeline(steps)


def tuning_configurations():
    # d=depth saves the current value. Without it, every lambda uses the last one
    for depth in [2, 3, 5, 8, None]:
        yield ("rf_max_depth", "Random Forest", f"max_depth={depth}",
               lambda seed, d=depth: build_random_forest(
                   max_depth=d, random_state=seed))

    for leaf in [1, 5, 10, 20]:
        yield ("rf_min_samples_leaf", "Random Forest",
               f"min_samples_leaf={leaf}",
               lambda seed, l=leaf: build_random_forest(
                   min_samples_leaf=l, random_state=seed))

    for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
        yield ("mlp_alpha", "MLP (32)", f"alpha={alpha}",
               lambda seed, a=alpha: build_neural_network(
                   (32,), alpha=a, random_state=seed))

    for variance in [0.80, 0.90, 0.95, 0.99]:
        yield ("pca_variance", "MLP (32) + PCA", f"pca_variance={variance}",
               lambda seed, v=variance: build_neural_network(
                   (32,), use_pca=True, pca_variance=v, random_state=seed))


def run_tuning():
    train, validation, _ = prepare_data()
    features = get_feature_columns(train, group=FEATURE_GROUP, window=WINDOW)

    X_train, y_train = train[features], train["FTR"]
    X_val, y_val = validation[features], validation["FTR"]

    rows = []
    for experiment, model_name, setting, build in tuning_configurations():
        for seed in SEEDS:
            model = build(seed)
            model.fit(X_train, y_train)

            rows.append({
                "experiment": experiment,
                "model": model_name,
                "setting": setting,
                "seed": seed,
                "train_macro_f1": evaluate(model, X_train, y_train)["macro_f1"],
                "validation_macro_f1": evaluate(model, X_val, y_val)["macro_f1"],
            })

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS_DIR / "tuning_results_per_seed.csv", index=False)

    summary = (
        per_seed
        .groupby(["experiment", "model", "setting"], sort=False)
        .agg(
            train_f1_mean=("train_macro_f1", "mean"),
            val_f1_mean=("validation_macro_f1", "mean"),
            val_f1_std=("validation_macro_f1", "std"),
        )
        .reset_index()
    )
    summary["gap"] = summary["train_f1_mean"] - summary["val_f1_mean"]
    summary = summary.round(3)
    summary.to_csv(RESULTS_DIR / "tuning_results.csv", index=False)

    print(f"Features: group {FEATURE_GROUP}, window {WINDOW} "
          f"({len(features)} columns)")
    print(f"Train matches: {len(train)}, validation matches: {len(validation)}")
    print(f"Seeds per setting: {SEEDS}\n")
    for experiment, table in summary.groupby("experiment", sort=False):
        print(f"== {experiment} ==")
        print(table.drop(columns=["experiment"]).to_string(index=False))
        print()

    return summary


if __name__ == "__main__":
    run_tuning()
