import pandas as pd
from sklearn.cluster import KMeans

data = {
    "Spending": [200,250,300,800,850,900,450,500,550],
    "Purchases": [2,3,2,8,9,10,5,4,6]
}

df = pd.DataFrame(data)

X = df[["Spending","Purchases"]]

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model.fit(X)

spending = float(input("Enter Spending: "))
purchases = int(input("Enter Number of Purchases: "))

cluster = model.predict(
    [[spending,purchases]]
)

print("Customer Segment:",
      cluster[0])
