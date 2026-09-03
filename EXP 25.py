import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("customer_reviews.csv")

# Change "rating" if your column has another name
ratings = df["rating"].dropna()

# Mean
mean = ratings.mean()

# Standard deviation
std = ratings.std()

# Sample size
n = len(ratings)

# Standard error
se = std / np.sqrt(n)

# 95% Confidence Interval
lower = mean - 1.96 * se
upper = mean + 1.96 * se

print("Mean Rating:", round(mean, 2))
print("Standard Error:", round(se, 4))

print("95% Confidence Interval:",
      round(lower, 2),
      "to",
      round(upper, 2))
