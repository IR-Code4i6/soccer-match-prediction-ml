import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns

#set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

#load data
df = pd.read_csv(Path(__file__).parent.parent / "Data/E0.csv")


#setup
cols_to_corr = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
existing_cols = [c for c in cols_to_corr if c in df.columns]

fig, ax = plt.subplots(figsize=(8, 6))
corr = df[existing_cols].corr()
sns.heatmap(
    corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax, vmin=-1, vmax=1
)
ax.set_title("Correlation Matrix of Match Statistics")

plt.savefig(Path(__file__).parent.parent / "DataVisualization/CorralationChart.png", dpi=300)

plt.close()
