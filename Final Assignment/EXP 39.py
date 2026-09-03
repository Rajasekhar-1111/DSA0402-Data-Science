import pandas as pd
from sklearn.cluster import KMeans

data = {
    "Customer_ID": [1,2,3,4,5,6,7,8,9,10],
    "Amount": [200,250,300,800,850,900,450,500,550,600],
    "Visits": [2,3,2,8,9,10,4,5,6,6]
}

df = pd.DataFrame(data)

X = df[["Amount","Visits"]]

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Segment"] = model.fit_predict(X)

print(df)
