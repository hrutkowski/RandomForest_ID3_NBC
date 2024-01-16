from algorithms.ID3 import *
from datasets_manager import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = get_dataset_corona()
df = pd.concat([X, y], axis=1)
train_data, test_data = train_test_split(df, test_size=0.2)
label = 'Corona'
columns_except_corona = [col for col in df.columns if col != label]

tree = fit(train_data, 'Corona', columns_except_corona)

predictions = []
for index, row in test_data.iterrows():
    query = row[columns_except_corona].to_dict()
    prediction = predict(query, tree)
    predictions.append(prediction)

actual_labels = test_data[label]

accuracy = get_accuracy(test_data, tree, label)
print("Decision Tree:")
print(tree)
print("Accuracy:", accuracy)

precision = precision_score(actual_labels, predictions, average='weighted', zero_division=0)
print("Precision:", precision)

accuracy = accuracy_score(actual_labels, predictions)
print("Accuracy:", accuracy)

recall = recall_score(actual_labels, predictions, average='weighted')
print("Recall:", recall)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)

predictions = tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
weighted_recall = recall_score(y_test, predictions, average='weighted')
weighted_precision = precision_score(y_test, predictions, average='weighted')
print("Precision for sklearn:", weighted_precision)
print("Accuracy for sklearn:", accuracy)
print("Recall for sklearn:", weighted_recall)

