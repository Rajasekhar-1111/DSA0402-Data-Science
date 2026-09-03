import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = {
    "Age":[25,35,45,50,30,60,40,55,28,48],
    "Gender":[0,1,0,1,1,0,0,1,1,0],
    "BP":[120,130,140,150,125,160,135,145,118,138],
    "Cholesterol":[180,200,220,240,190,250,210,230,175,215],
    "Outcome":[1,1,0,0,1,0,1,0,1,1]
}

df = pd.DataFrame(data)

X = df[["Age","Gender","BP","Cholesterol"]]
y = df["Outcome"]

Xtr,Xte,ytr,yte = train_test_split(
    X,y,test_size=0.3,random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(Xtr,ytr)

pred = model.predict(Xte)

print("Accuracy:", accuracy_score(yte,pred))
print("Precision:", precision_score(yte,pred,zero_division=0))
print("Recall:", recall_score(yte,pred,zero_division=0))
print("F1-Score:", f1_score(yte,pred,zero_division=0))
