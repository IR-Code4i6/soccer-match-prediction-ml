
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from typing import Any


def build_neural_network(hidden_layers=(32,), use_pca=False):
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]

    if use_pca:
        steps.append((
            "pca",
            PCA(n_components=0.95, svd_solver="full"),
        ))

    steps.append((
        "model",
        MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=800,
            early_stopping=False,
            random_state=42,
        ),
    ))

    return Pipeline(steps)
