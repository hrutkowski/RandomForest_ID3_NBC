from sklearn.model_selection import train_test_split
from algorithms.id3_classifier_old import ID3
from algorithms.nbc_classifier import NBC
from algorithms.id3_classifier import ID3
from algorithms.random_forest_algorithm import RandomForest
from datasets_manager import *
from experients_helpers import *
import numpy as np


X, y = get_dataset_corona()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

random_forest = RandomForest(n=10, classifiers=[ID3, NBC], classifiers_ratios=[0.5, 0.5])

acc = 0
f1 = 0
razy = 3
matrices = []

for i in range(razy):
    random_forest.fit(X_train, y_train)
    accuracy, f1_score, conf_matrix = random_forest.eval(X_test, y_test)
    acc += accuracy
    f1 += f1_score
    print(f"CONF MATRIX {i+1}")
    print(conf_matrix)
    matrices.append(conf_matrix)
    avg_matrix = np.round(np.sum(matrices, axis=0) / len(matrices))

print(f"CONF MATRIX AVG")
print(avg_matrix)
print(f"ACC AVG = {acc/razy}")
print(f"F1 AVG = {f1/razy}")