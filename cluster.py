import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./Coffee_Chain_Sales.csv")
X = df[["Marketing", "Sales"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nBasic statistics per cluster:")
print(df.groupby("Cluster")[["Marketing", "Sales"]].describe())

# ANOVA
clusters_sales = [group["Sales"].values for name, group in df.groupby("Cluster")]
f_stat_sales, p_value_sales = stats.f_oneway(*clusters_sales)

clusters_marketing = [
    group["Marketing"].values for name, group in df.groupby("Cluster")
]
f_stat_marketing, p_value_marketing = stats.f_oneway(*clusters_marketing)

print("\nANOVA Results for Numerical Variables:")
print(f"Sales -  F-Statistic: {f_stat_sales:.4f}, P-Value: {p_value_sales:.4e}")
print(
    f"Marketing - F-Statistic: {f_stat_marketing:.4f}, P-Value: {p_value_marketing:.4e}"
)

# Chi-squared test
categorical_vars = [
    "Market_size",
    "Product_line",
    "Product_type",
    "Product",
    "State",
    "Type",
]
print("\nChi-square Test Results for Categorical Variables:")
for var in categorical_vars:
    contingency_table = pd.crosstab(df["Cluster"], df[var])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\n{var}:")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"Degrees of freedom: {dof}")

# Cluster visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df["Marketing"], df["Sales"], c=df["Cluster"], cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="x",
    s=200,
    linewidths=3,
    color="r",
    label="Centroids",
)
plt.title("Coffee Chain Sales Clustering Results")
plt.xlabel("Marketing")
plt.ylabel("Sales")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.grid(True)
plt.show()

# Boxplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(x="Cluster", y="Sales", data=df, ax=ax1)
ax1.set_title("Sales Distribution by Cluster")
sns.boxplot(x="Cluster", y="Marketing", data=df, ax=ax2)
ax2.set_title("Marketing Distribution by Cluster")
plt.tight_layout()
plt.show()

categorical_vars = ["Market_size", "Product", "State", "Type"]
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.ravel()

categorical_vars = ["Market_size", "Product", "State", "Type"]

for var in categorical_vars:
    plt.figure(figsize=(12, 6))
    cross_tab = pd.crosstab(df[var], df["Cluster"])

    ax = cross_tab.plot(kind="bar", width=0.8)

    plt.title(f"Distribution of {var} by Cluster", fontsize=14, pad=20)
    plt.xlabel(var, fontsize=10)
    plt.ylabel("Frequency", fontsize=10)

    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    if var in ["Product", "State"]:
        plt.xticks(rotation=45, ha="right", fontsize=8)
    else:
        plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=7)
    plt.tight_layout()
    plt.show()
