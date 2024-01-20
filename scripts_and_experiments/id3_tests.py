from datasets_manager import *
import numpy as np
from algorithms.id3_classifier import ID3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import precision_score, recall_score, f1_score
from algorithms.ID3_class import ID3_class
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X, y = get_dataset_divorce()

# X = X.iloc[:100]
# y = y.iloc[:100]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print("Dimension of X:", X.shape)
#print("Dimension of y:", y.shape)

classifier = ID3_class()
classifier.fit(X, y)
#predictions = classifier.predict_all(X_test)
accuracy = classifier.get_accuracy()

# Assuming you have X_test, y_test, and predictions

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Display classification report
class_report = classification_report(y_test, predictions)
print("Classification Report:")
print(class_report)

# t = ID3(min_sample_num=5)
# t.fit(X_train, y_train)
# predictions, err_fp, err_fn = t.predict_all(X_test, y_test)
#
# # Evaluate the accuracy
# accuracy = (predictions == y_test.values).mean()
# print("Accuracy implementation:", accuracy)

# tree_classifier = DecisionTreeClassifier(random_state=42)
# tree_classifier.fit(X_train, y_train)
#
# # Make predictions on the test set
# predictions = tree_classifier.predict(X_test)
#
# precision_val = precision_score(predictions, y_test.values)
# recall_val = recall_score(predictions, y_test.values)
# f1_score_val = f1_score(predictions, y_test.values)
#
# print("Precision:", precision_val)
# print("Recall:", recall_val)
# print("F1 Score:", f1_score_val)
#
# # Evaluate the accuracy
# accuracy2 = accuracy_score(y_test, predictions)
# print("Accuracy library:", accuracy2)