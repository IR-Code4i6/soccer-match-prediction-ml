
import argparse

from src.data_pipeline import prepare_data
from src.baseline import run_baselines
from src.experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="English Premier League match prediction"
    )
    parser.add_argument(
        "--task",
        choices=["pipeline", "baseline", "experiments"],
        default="baseline",
    )
    args = parser.parse_args()

    if args.task == "pipeline":
        train, validation, test = prepare_data()

        print("Data pipeline completed.")
        print(f"Training examples: {len(train)}")
        print(f"Validation examples: {len(validation)}")
        print(f"Test examples: {len(test)}")
        print("\nTraining target distribution:")
        print(train["FTR"].value_counts())

    elif args.task == "baseline":
        run_baselines()

    elif args.task == "experiments":
        run_experiments()


if __name__ == "__main__":
    main()
