import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

data = {
    "Mileage": [20000, 30000, 40000, 50000, 60000, 70000],
    "Age": [2, 3, 4, 5, 6, 7],
    "Price": [800000, 750000, 650000, 550000, 450000, 350000]
}

df = pd.DataFrame(data)

X = df[["Mileage", "Age"]]
y = df["Price"]

model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# New car
new_car = pd.DataFrame({
    "Mileage": [30000],
    "Age": [3]
})

prediction = model.predict(new_car)

print("Predicted Price:", prediction[0])

print("\nDecision Path:")
print(export_text(model, feature_names=list(X.columns)))
