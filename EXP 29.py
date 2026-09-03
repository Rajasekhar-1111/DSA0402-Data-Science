from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

model = DecisionTreeClassifier(random_state=42)
model.fit(iris.data, iris.target)

x = []

x.append(float(input("Sepal Length: ")))
x.append(float(input("Sepal Width: ")))
x.append(float(input("Petal Length: ")))
x.append(float(input("Petal Width: ")))

pred = model.predict([x])

print("Predicted Species:",
      iris.target_names[pred[0]])
