import pandas as pd
from sklearn.cluster import KMeans

data = {
    "Age": [20,22,25,35,38,40,50,52,55],
    "Spending": [200,250,300,700,750,800,400,450,500],
    "Visits": [2,3,2,7,8,9,4,5,4]
}

df = pd.DataFrame(data)

X = df[["Age","Spending","Visits"]]

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X)

print(df)
