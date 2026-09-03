import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Area":[1000,1200,1500,1800,2000,2200],
    "Bedrooms":[2,2,3,3,4,4],
    "Price":[3000000,3500000,4500000,
             5200000,6000000,6500000]
}

df = pd.DataFrame(data)

X = df[["Area","Bedrooms"]]
y = df["Price"]

model = LinearRegression()
model.fit(X,y)

area = float(input("Enter Area: "))
bed = int(input("Enter Bedrooms: "))

print("Predicted Price:",
      round(model.predict([[area,bed]])[0],2))
