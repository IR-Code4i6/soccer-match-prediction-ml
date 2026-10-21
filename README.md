# Soccer Match Prediction ML

This project uses machine learning to predict English Premier League match outcomes.

The goal is to classify each match as:
- Home Win
- Draw
- Away Win

We use historical match data from Football-Data.co.uk and create features from recent team performance, such as:
- Goals scored
- Goals conceded
- Points earned
- Shots on target

These experiments include comparisons with the past 3, 5, and 10 venue-specific matches. The home team feature consists of past home games, while the away team feature includes past away games.

## Reproduce Milestone 3

All of the workflow is run using **CPython 3.11.0 on Windows x86-64**. You should use pinned dependencies from the `requirements.txt` file; these will include Seaborn and necessary packages for running the cleaning notebook. Run all commands below in **PowerShell starting from the repo root**.

Create an isolated environment and install the dependencies:

```powershell
python --version  # The tested version is Python 3.11.0.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The commands utilize Python from the environment, so you can activate it optionally. Create a kernel for your notebook within this environment to make sure that the cleanup process utilizes the same dependencies as the training:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name soccer-match-prediction-ml --display-name "Soccer Match Prediction ML"
```

Execute the complete cleaning notebook:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --stdout --ExecutePreprocessor.timeout=120 --ExecutePreprocessor.kernel_name=soccer-match-prediction-ml Filtering_Cleaning.ipynb > $null
```

The command runs against `Data/E0.csv` (380 matches, 132 columns) and outputs to `Data/E0_cleaned.csv` (380 matches, 24 columns). The JSON output of the executed notebook is deleted using `> $null`; the source of the notebook is saved, as well as the cleaned CSV. Alternatively, you can load the notebook in any Jupyter-aware editor and run all cells with the registered kernel.


Run the data pipeline, fixed baseline, and experiment comparison in that order:

```powershell
.\.venv\Scripts\python.exe main.py --task pipeline
.\.venv\Scripts\python.exe main.py --task baseline
.\.venv\Scripts\python.exe main.py --task experiments
```

[![Watch the demo](https://img.youtube.com/vi/PSxzFBh0fqg/0.jpg)](https://www.youtube.com/watch?v=PSxzFBh0fqg)

The pipeline uses **`Data/E0_cleaned.csv`** file and not the raw CSV file. For a match to be eligible, the home team must have at least 10 previous home matches and the away team must have at least 10 previous away matches. From the provided dataset, there are 200 warm-up matches and 180 valid matches.


| Split | Expected examples | Date range |
| --- | ---: | --- |
| Training | 125 | 2026-01-06 to 2026-04-19 |
| Validation | 28 | 2026-04-19 to 2026-05-09 |
| Test | 27 | 2026-05-09 to 2026-05-24 |

The split utilizes row-based time splits. Dates may appear in neighboring splits. The above numbers indicate the split implementation.

The baseline command estimates the majority classifier and the Logistic Regression for Group C, window 5 (8 historical features). The experiments command evaluates 70 model-feature combinations using validation macro-F1, retrains the best configuration on the combined training and validation sets (153 matches), and evaluates it once on the 27-match test set. The historical features represent sequential prediction; therefore, results from earlier fixtures are available when features for later fixtures are created.

The output files are relative to the root directory of the repository. Re-execution of a command will overwrite its associated CSV files:

| Command | Output files |
| --- | --- |
| Cleaning notebook | `Data/E0_cleaned.csv` |
| `--task pipeline` | Console split counts and training-label distribution; no output file |
| `--task baseline` | `results/baseline_results.csv`, `results/baseline_classification_report.csv`, `results/baseline_data_summary.csv` |
| `--task experiments` | `results/validation_results.csv`, `results/final_test_results.csv` |

The fixed baseline results are:

| Model | Test accuracy | Macro-precision | Macro-recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Majority-class benchmark | 0.444444 | 0.148148 | 0.333333 | 0.205128 |
| Logistic Regression | 0.370370 | 0.425000 | 0.333333 | 0.342287 |

The confusion matrices used for the results CSV files are in the format of actual row labels and predicted column labels, all in the order of H, D, A. The chosen experiment was the Random Forest model from Group C, window size 10, with test accuracy of 0.333333 and macro-f1 of 0.318835.

## Models

The project compares:
- Logistic Regression
- Random Forest
- Neural Networks (MLP(32), MLP(64,32))
- MLP (with PCA)

## Tools

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Dataset

Source: Football-Data.co.uk

Available: https://football-data.co.uk/data.php

## Team Members

- Irfan Nasery
- Mohammad Ihsan Wadid
- Levent Cakir
- Bruce Hatten

## Repository

This repository contains the dataset cleaning, preprocessing, visualizations, and machine learning work for the project.
