import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.DataFrame({
    "Engine":[1.2,1.5,1.8,2.0,2.2,2.5,3.0],
    "Horsepower":[80,100,120,140,160,180,220],
    "Mileage":[20,18,17,16,15,14,12],
    "Price":[500000,650000,800000,950000,
             1100000,1300000,1700000]
})

X = df[["Engine","Horsepower","Mileage"]]
y = df["Price"]

model = LinearRegression()
model.fit(X,y)

pred = model.predict(X)

print("RMSE:",
      np.sqrt(mean_squared_error(y,pred)))

print("R2 Score:",
      r2_score(y,pred))

print("\nCoefficients:")

for f,c in zip(X.columns,model.coef_):
    print(f,":",round(c,2))
