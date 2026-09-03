import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = {
    "Fever":[1,1,0,0,1,0,1,0],
    "Cough":[1,1,0,1,1,0,1,0],
    "Fatigue":[1,0,0,1,1,0,1,0],
    "Condition":[1,1,0,0,1,0,1,0]
}

df = pd.DataFrame(data)

X = df[["Fever","Cough","Fatigue"]]
y = df["Condition"]

k = int(input("Enter K: "))

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X,y)

f = int(input("Fever: "))
c = int(input("Cough: "))
fa = int(input("Fatigue: "))

print("Predicted Condition:",
      model.predict([[f,c,fa]])[0])
