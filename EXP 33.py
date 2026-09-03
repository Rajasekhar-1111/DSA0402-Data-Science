import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.DataFrame({
    "Size":[800,1000,1200,1500,1800,2000,2200],
    "Price":[2500000,3000000,3500000,
             4500000,5200000,6000000,6500000]
})

X = df[["Size"]]
y = df["Price"]

model = LinearRegression()
model.fit(X,y)

pred = model.predict(X)

print("RMSE:", np.sqrt(mean_squared_error(y,pred)))
print("R2 Score:", r2_score(y,pred))

plt.scatter(X,y)
plt.plot(X,pred)
plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("House Size vs Price")
plt.show()
