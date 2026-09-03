import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    "Usage": [100,200,150,300,250,120,350,180,280,220],
    "Contract": [12,24,12,36,24,12,36,18,30,24],
    "Churn": [1,0,1,0,0,1,0,1,0,0]
}

df = pd.DataFrame(data)

X = df[["Usage","Contract"]]
y = df["Churn"]

model = LogisticRegression()
model.fit(X,y)

usage = float(input("Enter Usage Minutes: "))
contract = float(input("Enter Contract Duration: "))

prediction = model.predict([[usage,contract]])

if prediction[0] == 1:
    print("Customer will Churn")
else:
    print("Customer will Not Churn")
