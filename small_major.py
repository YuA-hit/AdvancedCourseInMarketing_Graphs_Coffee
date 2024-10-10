import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./Coffee_Chain_Sales.csv")

# market_sizeの出現回数を集計
market_size_counts = df["Market_size"].value_counts()

# 棒グラフの作成
plt.figure(figsize=(10, 6))
market_size_counts.plot(kind="bar")
plt.title("Frequency of Market Sizes")
plt.xlabel("Market Size")
plt.ylabel("Count")
plt.xticks(rotation=0)

# 各バーの上に値を表示
for i, v in enumerate(market_size_counts):
    plt.text(i, v, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.show()

# 結果の出力
print(market_size_counts)

# データの概要を表示
print("\nData Summary:")
print(df["Market_size"].describe())

# ユニークな値を表示
print("\nUnique Market Sizes:")
print(df["Market_size"].unique())

# Small MarketとMajor Marketのデータを分離
small_market = df[df["Market_size"] == "Small Market"]
major_market = df[df["Market_size"] == "Major Market"]

# 散布図の作成
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

# グリッドの追加
plt.grid(True, linestyle="--", alpha=0.7)

# プロットの表示
plt.tight_layout()
plt.show()
