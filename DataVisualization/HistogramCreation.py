import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Load
df = pd.read_csv(Path(__file__).parent.parent / "Data/E0.csv")

stat_name_mapping = {
    "FTHG": "Full-Time Home Goals",
    "FTAG": "Full-Time Away Goals",
    "HTHG": "Half-Time Home Goals",
    "HTAG": "Half-Time Away Goals",
    "HS": "Home Total Shots",
    "AS": "Away Total Shots",
    "HST": "Home Shots on Target",
    "AST": "Away Shots on Target",
    "HF": "Home Fouls Committed",
    "AF": "Away Fouls Committed",
    "HC": "Home Corners",
    "AC": "Away Corners",
    "HY": "Home Yellow Cards",
    "AY": "Away Yellow Cards",
    "HR": "Home Red Cards",
    "AR": "Away Red Cards",
}

existing_stats = [s for s in stat_name_mapping.keys() if s in df.columns]

# Create a histogram grid
fig, axes = plt.subplots(
    nrows=4, ncols=4, figsize=(16, 13), sharex=False, sharey=False
)
axes = axes.flatten()

for i, stat in enumerate(existing_stats):
    ax = axes[i]
    df[stat].dropna().hist(ax=ax, bins=10, edgecolor="black", grid=False)

    ax.set_title(stat_name_mapping[stat], fontsize=10, fontweight="bold")
    ax.set_xlabel("Stat Value per Match", fontsize=8)
    ax.set_ylabel("Number of Matches", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)

# Hide any unused subplots if fewer than 16
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

#plt.suptitle("Distribution of All Match Statistics")
plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "DataVisualization/StatsHistogram.png", dpi=300)
plt.close()