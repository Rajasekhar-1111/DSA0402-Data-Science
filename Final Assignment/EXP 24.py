import pandas as pd
import numpy as np
from scipy.stats import norm

# Load dataset
df = pd.read_csv("rare_elements.csv")

data = df.iloc[:, 0].dropna().values

# User input
n = int(input("Enter sample size: "))
confidence = float(input("Enter confidence level (e.g. 0.95): "))
precision = float(input("Enter desired precision: "))

# Random sample
sample = np.random.choice(
    data,
    size=n,
    replace=False
)

# Point estimate
mean = np.mean(sample)

# Standard error
se = np.std(sample, ddof=1) / np.sqrt(n)

# Z value
z = norm.ppf(
    1 - (1 - confidence) / 2
)

# Confidence interval
lower = mean - z * se
upper = mean + z * se

print("\nSample Mean:", round(mean, 4))
print("Standard Error:", round(se, 4))

print("Confidence Interval:",
      round(lower, 4),
      "to",
      round(upper, 4))

print("Desired Precision:", precision)
