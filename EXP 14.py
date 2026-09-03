import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9],
    "Exam_Score": [50, 55, 60, 65, 70, 75, 82, 88]
}

df = pd.DataFrame(data)

correlation = df["Study_Hours"].corr(df["Exam_Score"])

print("Correlation:", round(correlation, 2))

plt.scatter(df["Study_Hours"], df["Exam_Score"])
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.show()
