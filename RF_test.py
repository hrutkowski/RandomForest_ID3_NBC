import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from algorithms.id3_classifier import ID3
from algorithms.nbc_classifier import NBC
from algorithms.random_forest_algorithm import RandomForest

# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForest(n=10, classifier_list=[ID3, NBC], classifier_ratio_list=[0.5, 0.5])

random_forest.fit(X_train, y_train)

accuracy, f1_score = random_forest.eval(X_test, y_test)

print("ACCURACY: {accuracy}")
print("F1 Score: {f1_score}")