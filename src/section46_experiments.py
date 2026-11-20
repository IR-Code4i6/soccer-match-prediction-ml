"""Frozen validation-only analyses for Milestone 5 section 4.6.

Run with ``python -m src.section46_experiments``. Each invocation creates a new
run directory. No test predictions are made and earlier results are preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data_pipeline import get_feature_columns, prepare_data
from src.tuning import build_random_forest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 0, 1, 2, 3)
LABELS = ("H", "D", "A")
MASK_MASTER_SEED = 4602026
MASK_REPLICATES = 20
MASK_PROBABILITIES = (0.1, 0.2)
ALL_STATS = ("goals_for", "goals_against", "points", "shots_on_target")
BASELINE_ID = "full_c10_t200"


@dataclass(frozen=True)
class Configuration:
    config_id: str
    label: str
    window: int = 10
    n_estimators: int = 200
    stats: tuple[str, ...] = ALL_STATS
    training_fraction: float = 1.0


def configurations():
    return [
        Configuration(BASELINE_ID, "Full tuned model"),
        Configuration("history_3", "3 venue matches", window=3),
        Configuration("history_5", "5 venue matches", window=5),
        Configuration("trees_100", "100 trees", n_estimators=100),
        Configuration("trees_500", "500 trees", n_estimators=500),
        Configuration("without_shots", "Without shots on target", stats=ALL_STATS[:3]),
        Configuration("without_points", "Without points", stats=("goals_for", "goals_against", "shots_on_target")),
        Configuration("goals_only", "Goals only", stats=ALL_STATS[:2]),
        Configuration("training_50", "Latest 50% of training rows", training_fraction=0.5),
        Configuration("training_75", "Latest 75% of training rows", training_fraction=0.75),
    ]


STUDIES = {
    "history_window": ("history_3", "history_5", BASELINE_ID),
    "forest_size": ("trees_100", BASELINE_ID, "trees_500"),
    "ablation": (BASELINE_ID, "without_shots", "without_points", "goals_only"),
    "training_size": ("training_50", "training_75", BASELINE_ID),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def _environment():
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")
    return {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "forest_n_jobs": 1,
        "timezone": "America/Toronto",
    }


def _metrics(actual, predicted):
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    assert matrix.sum() == len(actual) == 28
    assert set(predicted).issubset(LABELS)
    return {
        "validation_accuracy": float(accuracy_score(actual, predicted)),
        "validation_macro_f1": float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix_HDA": json.dumps(matrix.tolist()),
    }


def _mean_sd_ci(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == len(SEEDS)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def _paired_summary(rows, baseline):
    ordered = rows.set_index("model_seed").loc[list(SEEDS)]
    reference = baseline.set_index("model_seed").loc[list(SEEDS)]
    result = {}
    for metric in ("validation_accuracy", "validation_macro_f1"):
        for name, value in _mean_sd_ci(ordered[metric]).items():
            result[f"{metric}_{name}"] = value
        for name, value in _mean_sd_ci(ordered[metric].values - reference[metric].values).items():
            result[f"delta_{metric}_{name}"] = value
    return result


def _make_figures(output_dir, config_summary, robustness_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure_dir = output_dir / "figures"
    figure_dir.mkdir()
    groups = [
        ("history_window", "History window", ["3", "5", "10"], "Previous venue matches"),
        ("forest_size", "Forest size", ["100", "200", "500"], "Trees"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, (study, title, labels, xlabel) in zip(axes, groups):
        frame = config_summary[config_summary.experiment == study]
        ax.errorbar(range(len(frame)), frame.validation_macro_f1_mean, yerr=frame.validation_macro_f1_sd,
                    marker="o", capsize=4, color="#245C81", linewidth=1.8)
        ax.set(xticks=range(len(frame)), xticklabels=labels, xlabel=xlabel, ylabel="Validation macro-F1", title=title, ylim=(0, 1))
    fig.savefig(figure_dir / "sensitivity.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    frame = config_summary[config_summary.experiment == "training_size"]
    axes[0].errorbar(range(3), frame.validation_macro_f1_mean, yerr=frame.validation_macro_f1_sd,
                     marker="o", capsize=4, color="#245C81")
    axes[0].set(xticks=range(3), xticklabels=[str(n) for n in frame.train_samples], xlabel="Recent training fixtures", ylabel="Validation macro-F1", title="Fitting-sample stress test", ylim=(0, 1))
    axes[1].errorbar(robustness_summary.missing_probability * 100, robustness_summary.validation_macro_f1_mean,
                     yerr=robustness_summary.validation_macro_f1_sd, marker="o", capsize=4, color="#245C81")
    axes[1].set(xticks=[0, 10, 20], xlabel="Probability each feature value is missing (%)", ylabel="Validation macro-F1", title="Missing-feature stress test", ylim=(0, 1))
    fig.savefig(figure_dir / "robustness.png", dpi=220)
    plt.close(fig)

    frame = config_summary[config_summary.experiment == "ablation"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6), layout="constrained")
    ax.bar(range(4), frame.validation_macro_f1_mean, yerr=frame.validation_macro_f1_sd,
           capsize=4, color=["#245C81", "#6B96AE", "#6B96AE", "#6B96AE"])
    ax.set(xticks=range(4), xticklabels=["Full (8)", "Without shots (6)", "Without points (6)", "Goals only (4)"],
           ylabel="Validation macro-F1", title="Tuned Random Forest feature ablation", ylim=(0, 1))
    fig.savefig(figure_dir / "ablation.png", dpi=220)
    plt.close(fig)


def run_section46(output_root=None):
    """Run the pre-specified protocol, preserving outputs in a unique directory."""
    started = datetime.now(timezone.utc)
    output_root = Path(output_root) if output_root is not None else ROOT / "results" / "section46"
    output_dir = output_root / started.strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = _environment()
    snapshot = json.loads((ROOT / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8")) if (ROOT / "SOURCE_SNAPSHOT.json").exists() else {}
    source_files = [Path(__file__), ROOT / "main.py", ROOT / "requirements.txt", *sorted((ROOT / "src").glob("*.py"))]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): _sha256(p) for p in source_files}
    original_results = {p.name: _sha256(p) for p in (ROOT / "results").glob("*.csv")}
    data_hash = _sha256(ROOT / "Data" / "E0_cleaned.csv")
    manifest = {
        "status": "running", "started_utc": started.isoformat(),
        "started_local": started.astimezone(ZoneInfo("America/Toronto")).isoformat(),
        "base_commit": snapshot.get("source_commit", "not recorded"),
        "source_branch": snapshot.get("source_branch", "not recorded"),
        "source_files_sha256": source_hashes, "cleaned_data_sha256": data_hash,
        "existing_results_sha256": original_results, "environment": environment,
        "protocol": {
            "model": "Random Forest", "max_depth": 2, "min_samples_leaf": 1,
            "baseline_n_estimators": 200, "model_seeds": SEEDS,
            "configurations": [asdict(c) for c in configurations()], "studies": STUDIES,
            "mask_master_seed": MASK_MASTER_SEED, "mask_replicates": MASK_REPLICATES,
            "missing_probabilities": MASK_PROBABILITIES,
            "selection": "Frozen validation-only diagnostics; do not select a new final model.",
            "test_evaluation": "No test predictions or metrics are computed.",
            "training_subset": "Latest contiguous 62/93/125 training rows; all original past match histories remain available. Sample size, date coverage and class mix change together.",
            "missingness": "Independent uniform draws per fixture-feature cell; common draws across model seeds and nested thresholds across missingness levels. Imputer is fitted only on clean training data.",
            "uncertainty": "Mean/sample SD and t intervals over five model seeds on one fixed split. For missingness, average 20 masks within seed first. Mask variability is reported separately; 100 seed-mask scores are not independent trials.",
            "failed_or_discarded_runs": "All attempted evaluations are logged, including failures. No unsuccessful condition is discarded from reporting.",
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "protocol.json", manifest["protocol"])
    print(f"Section 4.6 output: {output_dir}", flush=True)
    records, predictions, fitted, fit_records = [], [], {}, {}

    def append_record(record, predicted=None):
        records.append(record)
        with (output_dir / "experiment_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str, allow_nan=False) + "\n")
        if predicted is not None:
            for position, (_, row) in enumerate(validation.iterrows()):
                predictions.append({
                    "run_id": record["run_id"], "fit_id": record["fit_id"],
                    "validation_position": position, "date": str(row.Date.date()),
                    "time": row.Time, "home_team": row.HomeTeam, "away_team": row.AwayTeam,
                    "actual": row.FTR, "predicted": str(predicted[position]),
                })

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train, validation, _ = prepare_data()
        assert (len(train), len(validation)) == (125, 28)
        assert validation.FTR.value_counts().to_dict() == {"H": 15, "A": 7, "D": 6}
        assert train.index.intersection(validation.index).empty
        manifest["data_warnings"] = [str(w.message) for w in caught]
        validation_hash = _frame_hash(validation)
        manifest["split"] = {
            "train_count": len(train), "validation_count": len(validation),
            "train_dates": [str(train.Date.min().date()), str(train.Date.max().date())],
            "validation_dates": [str(validation.Date.min().date()), str(validation.Date.max().date())],
            "train_classes": train.FTR.value_counts().to_dict(),
            "validation_classes": validation.FTR.value_counts().to_dict(),
            "train_frame_sha256": _frame_hash(train), "validation_frame_sha256": validation_hash,
        }
        base_features = get_feature_columns(train, group="C", window=10)
        assert len(base_features) == 8
        assert not train[base_features].isna().any().any()
        assert not validation[base_features].isna().any().any()
        baseline_X = validation[base_features]
        fixture_info = validation[["Date", "Time", "HomeTeam", "AwayTeam", "FTR"]].copy()
        fixture_info.to_csv(output_dir / "validation_fixtures.csv", index=False)
        for config in configurations():
            features = [f"{side}_{stat}_{config.window}" for side in ("home", "away") for stat in config.stats]
            count = int(len(train) * config.training_fraction)
            subset = train.iloc[-count:]
            assert set(subset.index).issubset(train.index)
            assert subset.index.equals(train.index[-count:])
            assert len(features) == 2 * len(config.stats)
            for seed in SEEDS:
                run_id = f"S46-F{len(fit_records) + 1:03d}"
                model = build_random_forest(max_depth=2, min_samples_leaf=1, random_state=seed)
                model.set_params(model__n_estimators=config.n_estimators, model__n_jobs=1)
                record = {
                    "run_id": run_id, "fit_id": run_id, "run_kind": "fit_and_validation",
                    "config_id": config.config_id, "label": config.label,
                    "model": "Random Forest", "model_seed": seed,
                    "window": config.window, "n_estimators": config.n_estimators,
                    "max_depth": 2, "min_samples_leaf": 1, "n_jobs": 1,
                    "feature_count": len(features), "features": json.dumps(features),
                    "train_samples": len(subset), "validation_samples": len(validation),
                    "train_start": str(subset.Date.min().date()), "train_end": str(subset.Date.max().date()),
                    "train_classes_HDA": json.dumps({label: int((subset.FTR == label).sum()) for label in LABELS}),
                    "train_frame_sha256": _frame_hash(subset), "validation_frame_sha256": validation_hash,
                    "base_commit": manifest["base_commit"], "script_sha256": _sha256(__file__),
                    "data_sha256": data_hash, "environment": json.dumps(environment),
                    "hyperparameters": json.dumps(model.get_params(), default=str, sort_keys=True),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "missing_probability": 0.0, "mask_replicate": None, "mask_seed": None,
                    "status": "running", "rationale": "Pre-specified controlled comparison; no additional model selection.",
                }
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        tic = time.perf_counter()
                        model.fit(subset[features], subset.FTR)
                        record["fit_seconds"] = time.perf_counter() - tic
                        tic = time.perf_counter()
                        predicted = model.predict(validation[features])
                        record["validation_predict_seconds"] = time.perf_counter() - tic
                        train_predicted = model.predict(subset[features])
                    assert np.allclose(model.named_steps["imputer"].statistics_, subset[features].median().values)
                    record.update(_metrics(validation.FTR, predicted))
                    record["train_macro_f1"] = float(f1_score(subset.FTR, train_predicted, labels=LABELS, average="macro", zero_division=0))
                    record["train_validation_gap"] = record["train_macro_f1"] - record["validation_macro_f1"]
                    record["warnings"] = json.dumps([str(w.message) for w in caught])
                    record["status"] = "success"
                except Exception:
                    record["status"] = "failed"
                    record["error"] = traceback.format_exc()
                    append_record(record)
                    raise
                append_record(record, predicted)
                fitted[(config.config_id, seed)] = model
                fit_records[(config.config_id, seed)] = record
            print(f"Completed {config.label}: {count} training rows, {len(features)} predictors, 5 seeds", flush=True)

        seed42 = fit_records[(BASELINE_ID, 42)]
        assert abs(seed42["validation_macro_f1"] - 0.621001221001221) < 1e-12, "Archived seed-42 baseline did not reproduce."
        assert abs(seed42["train_macro_f1"] - 0.6163660613071422) < 1e-12
        uniforms = [np.random.default_rng(MASK_MASTER_SEED + replicate).random(baseline_X.shape) for replicate in range(MASK_REPLICATES)]
        mask_records = []
        for probability in MASK_PROBABILITIES:
            for replicate, uniform in enumerate(uniforms):
                mask = uniform < probability
                assert np.array_equal(mask, uniform < probability)
                assert not np.any((uniform < 0.1) & ~(uniform < 0.2))
                masked_X = baseline_X.mask(mask)
                mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()
                mask_records.append({"missing_probability": probability, "mask_replicate": replicate, "mask_seed": MASK_MASTER_SEED + replicate,
                                     "mask_sha256": mask_hash, "masked_values": int(mask.sum()), "total_values": int(mask.size),
                                     "actual_missing_fraction": float(mask.mean()), "positions": np.argwhere(mask).tolist()})
                for seed in SEEDS:
                    parent = fit_records[(BASELINE_ID, seed)]
                    model = fitted[(BASELINE_ID, seed)]
                    record = dict(parent)
                    record.update({
                        "run_id": f"S46-M{sum(r['run_kind'] == 'masked_validation' for r in records) + 1:03d}",
                        "run_kind": "masked_validation", "started_utc": datetime.now(timezone.utc).isoformat(),
                        "fit_seconds": 0.0, "fit_reused": True,
                        "missing_probability": probability, "mask_replicate": replicate,
                        "mask_seed": MASK_MASTER_SEED + replicate, "mask_sha256": mask_hash,
                        "masked_values": int(mask.sum()), "actual_missing_fraction": float(mask.mean()),
                        "rationale": "Synthetic MCAR stress test using the frozen baseline model and training-fitted imputer.",
                    })
                    try:
                        tic = time.perf_counter()
                        predicted = model.predict(masked_X)
                        record["validation_predict_seconds"] = time.perf_counter() - tic
                        record.update(_metrics(validation.FTR, predicted))
                        record["train_validation_gap"] = record["train_macro_f1"] - record["validation_macro_f1"]
                    except Exception:
                        record.update(status="failed", error=traceback.format_exc())
                        append_record(record)
                        raise
                    append_record(record, predicted)
            print(f"Completed missing-feature probability {probability:.0%}: 20 common masks x 5 model seeds", flush=True)
        _write_json(output_dir / "masks.json", mask_records)
        log = pd.DataFrame(records)
        fit_log = log[log.run_kind == "fit_and_validation"]
        baseline = fit_log[fit_log.config_id == BASELINE_ID]
        summary_rows = []
        for study, identifiers in STUDIES.items():
            for identifier in identifiers:
                rows = fit_log[fit_log.config_id == identifier]
                first = rows.iloc[0]
                row = {"experiment": study, "config_id": identifier, "label": first.label,
                       "train_samples": int(first.train_samples), "feature_count": int(first.feature_count),
                       "window": int(first.window), "n_estimators": int(first.n_estimators),
                       "model_seed_count": len(SEEDS), "source_run_ids": "|".join(rows.run_id),
                       "train_start": first.train_start, "train_end": first.train_end,
                       "train_classes_HDA": first.train_classes_HDA}
                row.update(_paired_summary(rows, baseline))
                row["train_macro_f1_mean"] = float(rows.train_macro_f1.mean())
                summary_rows.append(row)
        config_summary = pd.DataFrame(summary_rows)
        config_summary.to_csv(output_dir / "config_summary.csv", index=False)

        masked = log[log.run_kind == "masked_validation"]
        seed_rows = []
        robustness_rows = []
        for probability in (0.0, *MASK_PROBABILITIES):
            for seed in SEEDS:
                rows = baseline[baseline.model_seed == seed] if probability == 0 else masked[(masked.missing_probability == probability) & (masked.model_seed == seed)]
                parent = fit_records[(BASELINE_ID, seed)]
                seed_rows.append({"missing_probability": probability, "model_seed": seed,
                                  "mask_count": len(rows), "validation_accuracy": float(rows.validation_accuracy.mean()),
                                  "validation_macro_f1": float(rows.validation_macro_f1.mean()),
                                  "delta_validation_macro_f1": float(rows.validation_macro_f1.mean()) - parent["validation_macro_f1"],
                                  "within_seed_mask_f1_sd": float(rows.validation_macro_f1.std(ddof=1)) if len(rows) > 1 else 0.0,
                                  "source_run_ids": "|".join(rows.run_id)})
            grouped = pd.DataFrame([r for r in seed_rows if r["missing_probability"] == probability])
            row = {"missing_probability": probability, "model_seed_count": len(SEEDS),
                   "mask_replicates_per_seed": 1 if probability == 0 else MASK_REPLICATES,
                   "source_run_ids": "|".join(grouped.source_run_ids)}
            row.update(_paired_summary(grouped, baseline))
            if probability:
                condition = masked[masked.missing_probability == probability]
                row["mask_effect_f1_sd"] = float(condition.groupby("mask_replicate").validation_macro_f1.mean().std(ddof=1))
                row["actual_missing_fraction_mean"] = float(condition.actual_missing_fraction.mean())
            else:
                row.update(mask_effect_f1_sd=0.0, actual_missing_fraction_mean=0.0)
            robustness_rows.append(row)
        pd.DataFrame(seed_rows).to_csv(output_dir / "robustness_by_seed.csv", index=False)
        robustness_summary = pd.DataFrame(robustness_rows)
        robustness_summary.to_csv(output_dir / "robustness_summary.csv", index=False)
        pd.DataFrame(predictions).to_csv(output_dir / "validation_predictions.csv", index=False)
        log.to_csv(output_dir / "experiment_log.csv", index=False)
        assert len(fit_log) == 50 and len(masked) == 200 and len(log) == 250
        assert log.run_id.is_unique and set(log.status) == {"success"}
        assert len(predictions) == 250 * 28
        assert log.validation_frame_sha256.nunique() == 1
        assert masked.groupby(["missing_probability", "mask_replicate"]).mask_sha256.nunique().eq(1).all()
        assert _frame_hash(validation) == validation_hash
        assert _sha256(ROOT / "Data" / "E0_cleaned.csv") == data_hash
        assert {p.name: _sha256(p) for p in (ROOT / "results").glob("*.csv")} == original_results
        _make_figures(output_dir, config_summary, robustness_summary)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        fit_count=50, masked_evaluation_count=200, successful_log_rows=len(log), failed_runs=0,
                        validation_prediction_rows=len(predictions),
                        checks={"archived_seed42_baseline_reproduced": True, "common_nested_masks": True,
                                "training_only_imputation": True, "same_validation_fixtures": True,
                                "original_data_and_results_unchanged": True, "test_metrics_computed": False})
        _write_json(output_dir / "run_manifest.json", manifest)
        (output_root / "latest_run.txt").write_text(output_dir.name + "\n", encoding="utf-8")
        print("Completed: 50 fits, 200 masked evaluations, 250 log rows; no test evaluation.", flush=True)
        print(config_summary[["experiment", "label", "validation_macro_f1_mean", "validation_macro_f1_sd", "delta_validation_macro_f1_mean"]].to_string(index=False), flush=True)
        print(robustness_summary.to_string(index=False), flush=True)
        return output_dir
    except Exception:
        manifest.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(), error=traceback.format_exc())
        _write_json(output_dir / "run_manifest.json", manifest)
        if records:
            pd.DataFrame(records).to_csv(output_dir / "experiment_log.csv", index=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    arguments = parser.parse_args()
    run_section46(arguments.output_root)
