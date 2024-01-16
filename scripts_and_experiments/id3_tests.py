from datasets_manager import *
import numpy as np
from algorithms.id_3_classifier import TreeNode
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris


X, y = get_dataset_corona()

X = X.iloc[:250]
y = y.iloc[:250]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dimension of X:", X.shape)
print("Dimension of y:", y.shape)

# iris = load_iris()
#
# df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                      columns= iris['feature_names'] + ['target'])
#
# Xiris = df.drop( "target", axis = 1)
# yiris = df["target"]
# X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(Xiris, yiris, test_size=0.2, random_state=42)
#
# print("Dimension of X:", Xiris.shape)
# print("Dimension of y:", yiris.shape)

t = TreeNode(min_sample_num=50)
t.fit(X_train, y_train)

predictions = []
corr = 0
err_fp = 0
err_fn = 0
for (i, ct), tgt in zip(X_test.iterrows(), y_test):
    a = t.predict(ct)
    predictions.append(a)
    if a and not tgt:
        err_fp += 1
    elif not a and tgt:
        err_fn += 1
    else:
        corr += 1

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