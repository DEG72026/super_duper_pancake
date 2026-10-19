from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib


iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target
)

classifier = DecisionTreeClassifier(max_depth=3, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

def classification(input_values):
    class_pred = classifier.predict(input_values)
    return iris.target_names[class_pred[0]]

joblib.dump(classifier, "iris_model.joblib")
print("Model saved successfully.")

# test = [[1,2,3,4]]
# answer = classification(test)
# print(answer)

# load the model later

# import joblib

# classifier = joblib.load("iris_model.joblib")

# prediction = classifier.predict([[5.1, 3.5, 1.4, 0.2]])
# print(prediction)