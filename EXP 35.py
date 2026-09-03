from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

iris = load_iris()

X_train,X_test,y_train,y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.3,
    random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",
      round(accuracy_score(y_test,pred),4))

print("Precision:",
      round(precision_score(y_test,pred,average="weighted"),4))

print("Recall:",
      round(recall_score(y_test,pred,average="weighted"),4))

print("F1-Score:",
      round(f1_score(y_test,pred,average="weighted"),4))
