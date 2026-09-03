import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("soccer_players.csv")

print("Top 5 Goal Scorers:")
print(df.nlargest(5,"Goals")[["Name","Goals"]])

print("\nTop 5 Salaries:")
print(df.nlargest(5,"Salary")[["Name","Salary"]])

avg = df["Age"].mean()

print("\nAverage Age:", round(avg,2))
print("\nAbove Average Age:")
print(df[df["Age"] > avg]["Name"])

df["Position"].value_counts().plot(kind="bar")
plt.xlabel("Position")
plt.ylabel("Players")
plt.title("Player Positions")
plt.show()
