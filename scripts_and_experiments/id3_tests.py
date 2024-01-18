from datasets_manager import *
import numpy as np
from algorithms.id_3_classifier import TreeNode
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris


X, y = get_dataset_corona()

# X = X.iloc[:100]
# y = y.iloc[:100]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print("Dimension of X:", X.shape)
#print("Dimension of y:", y.shape)


t = TreeNode(min_sample_num=5)
t.fit(X_train, y_train)
predictions, err_fp, err_fn = t.predict_all(X_test, y_test)

# Evaluate the accuracy
accuracy = (predictions == y_test.values).mean()
print("Accuracy implementation:", accuracy)

tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = tree_classifier.predict(X_test)

# Evaluate the accuracy
accuracy2 = accuracy_score(y_test, predictions)
print("Accuracy library:", accuracy2)