import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from scipy import stats
import statsmodels.api as sm

# グローバル変数としてcategorical_varsを定義
categorical_vars = [
    "Market_size",
    "Product_line",
    "Product_type",
    "Product",
    "State",
    "Type",
]

# データの読み込みとクラスタリング
df = pd.read_csv("./Coffee_Chain_Sales.csv")

# クラスタリングのための前処理
X = df[["Marketing", "Sales"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)


def perform_cluster_analysis(df, cluster_id):
    # Filter data for the specific cluster
    cluster_data = df[df["Cluster"] == cluster_id].copy()

    # Create dummy variables, excluding one category from each variable to avoid multicollinearity
    dummy_data = pd.get_dummies(cluster_data[categorical_vars], drop_first=True)

    # Prepare final dataset for regression
    X_vars = pd.DataFrame()
    X_vars["Marketing"] = cluster_data["Marketing"]  # Add numeric predictor
    X_vars = pd.concat([X_vars, dummy_data], axis=1)  # Add dummy variables

    # Convert to numeric type explicitly
    X_vars = X_vars.astype(float)

    # Add constant
    X_vars = sm.add_constant(X_vars)

    # Prepare dependent variable
    y = cluster_data["Sales"].astype(float)

    try:
        # Fit regression model
        model = sm.OLS(y, X_vars).fit()

        # Print results
        print(f"\n{'='*80}")
        print(f"Cluster {cluster_id} Analysis")
        print(f"{'='*80}")
        print(f"Number of observations: {len(cluster_data)}")
        print(f"\nR-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.4f}")
        print(f"F-statistic p-value: {model.f_pvalue:.4e}")

        # Print coefficients summary
        print("\nRegression Results:")
        print(model.summary().tables[1])

        # Calculate correlation with Sales for numeric variables
        numeric_corr = (
            cluster_data[["Sales", "Marketing"]].corr()["Sales"].drop("Sales")
        )
        print("\nCorrelations with Sales (Numeric Variables):")
        print(numeric_corr)

        # Calculate mean Sales for each categorical variable
        print("\nMean Sales by Categories:")
        for var in categorical_vars:
            mean_sales = cluster_data.groupby(var)["Sales"].agg(["mean", "count"])
            print(f"\n{var}:")
            print(mean_sales)

        return model

    except Exception as e:
        print(f"\nError in cluster {cluster_id} analysis:")
        print(str(e))
        print("\nVariables in the model:")
        print(X_vars.columns.tolist())
        return None


# Perform analysis for each cluster
print("Starting cluster analysis...")
cluster_models = {}
for cluster_id in sorted(df["Cluster"].unique()):
    cluster_models[cluster_id] = perform_cluster_analysis(df, cluster_id)

# Compare clusters
print("\n" + "=" * 80)
print("Cluster Comparison Summary")
print("=" * 80)
print("\nCluster Sizes:")
print(df["Cluster"].value_counts().sort_index())

# Compare average sales and marketing spend across clusters
cluster_summary = (
    df.groupby("Cluster")
    .agg(
        {
            "Sales": ["mean", "std", "min", "max"],
            "Marketing": ["mean", "std", "min", "max"],
        }
    )
    .round(2)
)

print("\nCluster Summary Statistics:")
print(cluster_summary)

# Categorical distribution across clusters
print("\nCategorical Variable Distribution Across Clusters:")
for var in categorical_vars:
    print(f"\n{var} Distribution:")
    dist = pd.crosstab(df["Cluster"], df[var], normalize="index") * 100
    print(dist.round(2))

# Save results to file
with open("cluster_analysis_results.txt", "w") as f:
    f.write("Cluster Analysis Results\n")
    f.write("=" * 80 + "\n")
    f.write(f"\nCluster Sizes:\n{df['Cluster'].value_counts().sort_index()}")
    f.write(f"\n\nCluster Summary Statistics:\n{cluster_summary}")
