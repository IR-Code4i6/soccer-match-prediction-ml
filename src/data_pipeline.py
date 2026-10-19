from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = (
        Path(__file__).resolve().parents[1]
        / "Data"
        / "E0_cleaned.csv"
)

WINDOWS = [3, 5, 10]
RECENCY_WINDOW = 5


def load_matches(path=DATA_PATH):
    """Load, validate, sort the match dataset"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "HST", "AST",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )

    for col in ["FTHG", "FTAG", "HST", "AST"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "HST", "AST", "FTR",
    ])
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()

    if "Time" not in df.columns:
        df["Time"] = "00:00"
    df["Time"] = df["Time"].fillna("00:00").astype(str)

    df = df.sort_values(
        ["Date", "Time"], kind="stable"
    ).reset_index(drop=True)

    return df


def team_record(row, is_home):
    """Convert a finished match into one teams perspective"""
    if is_home:
        result = row["FTR"]
        goals_for = row["FTHG"]
        goals_against = row["FTAG"]
        shots_on_target = row["HST"]
    else:
        result = {"H": "A", "A": "H", "D": "D"}[row["FTR"]]
        goals_for = row["FTAG"]
        goals_against = row["FTHG"]
        shots_on_target = row["AST"]

    points = 3 if result == "H" else 1 if result == "D" else 0

    return {
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "points": float(points),
        "shots_on_target": float(shots_on_target),
    }


def mean_stat(history, stat):
    return float(np.mean([match[stat] for match in history]))


def add_history_features(matches):
    """
    Home team features use its previous home matches
    Away team features use its previous away matches

    Each match is added to history only after its features
    have been calculated to prevent current match leakage
    """
    histories = {}
    feature_rows = []
    eligible_rows = []

    for _, row in matches.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_history = histories.setdefault((home, "home"), [])
        away_history = histories.setdefault((away, "away"), [])

        # Require 10 prior venue specific matches for fair comparison
        eligible = len(home_history) >= 10 and len(away_history) >= 10

        features = {}

        if eligible:
            for window in WINDOWS:
                home_recent = home_history[-window:]
                away_recent = away_history[-window:]

                for side, history in [
                    ("home", home_recent),
                    ("away", away_recent),
                ]:
                    # Group A: goals scored and conceded
                    for stat in ["goals_for", "goals_against"]:
                        features[
                            f"{side}_{stat}_{window}"
                        ] = mean_stat(history, stat)

                    # Group B: goals plus points
                    features[
                        f"{side}_points_{window}"
                    ] = mean_stat(history, "points")

                    # Group C: goals, points, and shots on target
                    features[
                        f"{side}_shots_on_target_{window}"
                    ] = mean_stat(history, "shots_on_target")

            # Recency experiment:
            # latest five venue specific appearances vs previous five
            for side, history in [
                ("home", home_history),
                ("away", away_history),
            ]:
                recent_five = history[-5:]
                older_five = history[-10:-5]

                for period, records in [
                    ("recent5", recent_five),
                    ("older5", older_five),
                ]:
                    for stat in [
                        "goals_for",
                        "goals_against",
                        "points",
                        "shots_on_target",
                    ]:
                        features[
                            f"{side}_{period}_{stat}"
                        ] = mean_stat(records, stat)

            feature_rows.append(features)
            eligible_rows.append(row.to_dict())

        # Update histories only after features for this match are built
        histories[(home, "home")].append(team_record(row, True))
        histories[(away, "away")].append(team_record(row, False))

    if not eligible_rows:
        raise ValueError(
            "No eligible matches. Check that the CSV has enough "
            "chronologically ordered home and away history."
        )

    metadata = pd.DataFrame(eligible_rows).reset_index(drop=True)
    feature_df = pd.DataFrame(feature_rows).reset_index(drop=True)

    return pd.concat([metadata, feature_df], axis=1)


def get_feature_columns(df, group="C", window=5):
    """Return features for group A, B, or C and a history window"""
    if group not in {"A", "B", "C"}:
        raise ValueError("Feature group must be A, B, or C.")

    stats = ["goals_for", "goals_against"]

    if group in {"B", "C"}:
        stats.append("points")

    if group == "C":
        stats.append("shots_on_target")

    return [
        f"{side}_{stat}_{window}"
        for side in ["home", "away"]
        for stat in stats
    ]


def get_recency_columns():
    """Return features for the recent five versus older five experiment"""
    return [
        f"{side}_{period}_{stat}"
        for side in ["home", "away"]
        for period in ["recent5", "older5"]
        for stat in [
            "goals_for", "goals_against",
            "points", "shots_on_target",
        ]
    ]


def chronological_split(df):
    """Split the eligible matches into older and newer periods"""
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    if train_end == 0 or validation_end <= train_end or validation_end >= n:
        raise ValueError(
            f"Not enough eligible matches for a 70/15/15 split: {n}"
        )

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:validation_end].copy(),
        df.iloc[validation_end:].copy(),
    )


def prepare_data(path=DATA_PATH):
    matches = load_matches(path)
    featured = add_history_features(matches)
    return chronological_split(featured)


if __name__ == "__main__":
    train, validation, test = prepare_data()

    print("Eligible matches:", len(train) + len(validation) + len(test))
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))
    print("\nOutcome distribution:")
    print(pd.concat([train, validation, test])["FTR"].value_counts())
