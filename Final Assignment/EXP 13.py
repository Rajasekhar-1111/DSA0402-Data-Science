import pandas as pd

df = pd.read_csv("stock_data.csv")

prices = df["Close"]

mean = prices.mean()
variance = prices.var()
std = prices.std()

print("Mean Closing Price:", round(mean, 2))
print("Variance:", round(variance, 2))
print("Standard Deviation:", round(std, 2))
