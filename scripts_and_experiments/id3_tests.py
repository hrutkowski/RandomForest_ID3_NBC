from datasets_manager import *
from sklearn.model_selection import train_test_split
from algorithms.id3_classifier import ID3
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X, y = get_dataset_corona()

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print("Dimension of X:", X.shape)
#print("Dimension of y:", y.shape)

classifier = ID3()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
classifier.get_accuracy(X_test, y_test)
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
