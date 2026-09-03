import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Spending": [200,250,300,800,850,900,450,500,550,600],
    "Items": [2,3,2,8,9,10,4,5,6,6]
}

df = pd.DataFrame(data)

X = df[["Spending","Items"]]

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X)

print(df)

plt.scatter(
    df["Spending"],
    df["Items"],
    c=df["Cluster"]
)

plt.xlabel("Total Amount Spent")
plt.ylabel("Number of Items")
plt.title("Customer Segmentation using K-Means")
plt.show()
