import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("./Coffee_Chain_Sales.csv")

states = [
    "California",
    "Utah",
    "Nevada",
    "Colorado",
    "Oregon",
    "Washington",
    "Ohio",
    "Florida",
    "Illinois",
    "Wisconsin",
    "Missouri",
    "Iowa",
    "New York",
    "Texas",
    "Connecticut",
    "Oklahoma",
    "Louisiana",
    "New Hampshire",
    "New Mexico",
    "Massachusetts",
]

colors = plt.cm.rainbow(np.linspace(0, 1, len(states)))
color_dict = dict(zip(states, colors))

# Frequency
market_size_counts = df["State"].value_counts()

plt.figure(figsize=(24, 8))
bars = market_size_counts.plot(
    kind="bar", color=[color_dict[state] for state in market_size_counts.index]
)
plt.title("Frequency of States")
plt.xlabel("States", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, fontsize=12)

for i, v in enumerate(market_size_counts):
    plt.text(i, v, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.show()

print(market_size_counts)

# Scatter
plt.figure(figsize=(10, 6))

for state in states:
    state_data = df[df["State"] == state]
    plt.scatter(
        state_data["Marketing"],
        state_data["Sales"],
        color=color_dict[state],
        label=state,
        alpha=0.8,
    )

plt.xlabel("Marketing", fontsize=16)
plt.ylabel("Sales", fontsize=16)
plt.title("Sales vs Marketing by State", fontsize=18)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.7)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

# Average Heatmap

state_avg = df.groupby("State")[["Marketing", "Sales"]].mean()

plt.figure(figsize=(12, 10))
sns.heatmap(state_avg, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Average Marketing and Sales by State", fontsize=18)
plt.xlabel("Metrics", fontsize=14)
plt.ylabel("States", fontsize=14)

plt.tight_layout()
plt.show()

# バカデカいHeatmap
correlation = df[["Marketing", "Profit", "Sales", "Total_expenses"]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Correlation Heatmap", fontsize=18)

plt.tight_layout()
plt.show()

columns_of_interest = ["Marketing", "Profit", "Sales", "Total_expenses"]

n_states = len(states)
n_cols = 5  # 1行に表示する州の数を増やす
n_rows = (n_states - 1) // n_cols + 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

for i, state in enumerate(states):
    state_data = df[df["State"] == state]
    correlation = state_data[columns_of_interest].corr()
    row = i // n_cols
    col = i % n_cols

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=axes[row, col],
        annot_kws={"size": 6},
        cbar=False,
    )

    axes[row, col].set_title(f"{state}", fontsize=8)
    axes[row, col].set_xticklabels(
        axes[row, col].get_xticklabels(), rotation=45, ha="right", fontsize=4
    )
    axes[row, col].set_yticklabels(
        axes[row, col].get_yticklabels(), rotation=0, fontsize=6
    )

for j in range(i + 1, n_rows * n_cols):
    row = j // n_cols
    col = j % n_cols
    axes[row, col].axis("off")

plt.tight_layout()
plt.show()
