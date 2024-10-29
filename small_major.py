import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv("./Coffee_Chain_Sales.csv")

small_market = df[df["Market_size"] == "Small Market"]
major_market = df[df["Market_size"] == "Major Market"]

plt.figure(figsize=(12, 8))

# Small Market
plt.scatter(
    small_market["Marketing"],
    small_market["Sales"],
    color="blue",
    label="Small Market",
    alpha=0.6,
)
slope_small, intercept_small, r_small, _, _ = stats.linregress(
    small_market["Marketing"], small_market["Sales"]
)
line_small = slope_small * small_market["Marketing"] + intercept_small
plt.plot(small_market["Marketing"], line_small, color="blue", linestyle="--")

# Major Market
plt.scatter(
    major_market["Marketing"],
    major_market["Sales"],
    color="red",
    label="Major Market",
    alpha=0.6,
)
slope_major, intercept_major, r_major, _, _ = stats.linregress(
    major_market["Marketing"], major_market["Sales"]
)
line_major = slope_major * major_market["Marketing"] + intercept_major
plt.plot(major_market["Marketing"], line_major, color="red", linestyle="--")

plt.xlabel("Marketing")
plt.ylabel("Sales")
plt.title("Sales vs Marketing: Small Market vs Major Market (with Linear Regression)")
plt.legend()

plt.grid(True, linestyle="--", alpha=0.7)

# Add correlation coefficients to the legend
plt.legend(title=f"Small Market: r = {r_small:.3f}\nMajor Market: r = {r_major:.3f}")

plt.tight_layout()
plt.show()

# Print correlation coefficients
print(f"Small Market correlation coefficient: {r_small:.3f}")
print(f"Major Market correlation coefficient: {r_major:.3f}")
