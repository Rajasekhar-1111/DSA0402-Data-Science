import pandas as pd

df = pd.read_csv("temperature_data.csv")

temperature = df.iloc[:, 1:]

df["Mean"] = temperature.mean(axis=1)
df["Standard Deviation"] = temperature.std(axis=1)
df["Range"] = temperature.max(axis=1) - temperature.min(axis=1)

print(df[["City", "Mean", "Standard Deviation", "Range"]])

highest_range = df.loc[df["Range"].idxmax(), "City"]
most_consistent = df.loc[df["Standard Deviation"].idxmin(), "City"]

print("\nHighest Temperature Range:", highest_range)
print("Most Consistent City:", most_consistent)
