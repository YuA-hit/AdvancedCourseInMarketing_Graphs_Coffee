import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./Coffee_Chain_Sales.csv")

market_size_counts = df["Market_size"].value_counts()

plt.figure(figsize=(10, 6))
market_size_counts.plot(kind="bar")
plt.title("Frequency of Market Sizes")
plt.xlabel("Market Size")
plt.ylabel("Count")
plt.xticks(rotation=0)

for i, v in enumerate(market_size_counts):
    plt.text(i, v, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.show()

print(market_size_counts)

print("\nData Summary:")
print(df["Market_size"].describe())

print("\nUnique Market Sizes:")
print(df["Market_size"].unique())

small_market = df[df["Market_size"] == "Small Market"]
major_market = df[df["Market_size"] == "Major Market"]

plt.figure(figsize=(10, 6))
plt.scatter(
    small_market["Marketing"],
    small_market["Profit"],
    color="blue",
    label="Small Market",
    alpha=0.6,
)
plt.scatter(
    major_market["Sales"],
    major_market["Profit"],
    color="red",
    label="Major Market",
    alpha=0.6,
)

plt.xlabel("Marketing")
plt.ylabel("Profit")
plt.title("Profit vs Marketing: Small Market vs Major Market")
plt.legend()

plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
