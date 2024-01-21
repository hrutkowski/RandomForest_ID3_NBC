import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn import metrics
from itertools import product
from sklearn.naive_bayes import CategoricalNB
from typing import List, Tuple

from algorithms.id3_classifier import ID3
from algorithms.nbc_classifier import NBC
from sklearn.tree import DecisionTreeClassifier
from algorithms.random_forest_algorithm import RandomForest
from scripts_and_experiments.datasets_manager import get_dataset_corona, get_dataset_divorce, get_dataset_glass, \
    get_dataset_loan_approval
from scripts_and_experiments.experiment_scripts import cross_validation_score, test_accuracy, get_conf_matrix, \
    run_experiment
from experients_helpers import random_forest_experiments



# Porównanie klasyfikacji przy użyciu wybranej przez Nas gotowej i lekko przerobionej pod Nasze
# potrzeby implementacji ID3 z gotową implementacją z biblioteki sklearn
def id3_comparison():
    print('======================== EKSPERYMENT: Porównanie ID3 ============================')
    print('Porównanie wybranej implementacji ID3 z implementacją drzewa z biblioteki sklearn')

    datasets = [get_dataset_corona(), get_dataset_divorce(), get_dataset_glass(), get_dataset_loan_approval()]

    for dataset in datasets:
        X, y = dataset
        label_column = y.name
        print("Zbiór", label_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        t = ID3()
        start = time.time()
        t.fit(X_train, y_train)
        print('Czas fit() id3 :', time.time() - start)
        predictions = t.predict(X_test)
        our_acc = metrics.accuracy_score(y_test, predictions)

        tree_classifier = DecisionTreeClassifier(criterion="entropy")
        start = time.time()
        tree_classifier.fit(X_train, y_train)
        print('Czas fit() sklearn. :', time.time() - start)
        y_pred = tree_classifier.predict(X_test)
        our_preds = np.array(y_pred, dtype=int)
        sklearn_id3_acc = metrics.accuracy_score(y_test, our_preds)

        print(f"Dokładność obu implementacji jest zbliżona: - {our_acc}=={sklearn_id3_acc}")
        print('=========================================================')


# Porównanie klasyfikacji przy użyciu Naszej implementacji algorytmu NBC z gotową implementacją z biblioteki sklearn
def nbc_comparison():
    print('====================== EKSPERYMENT: Porównanie NBC ==========================')
    print('Porównanie własnej implementacji NBC z implementacją z biblioteki sklearn')

    datasets = [get_dataset_corona(), get_dataset_divorce(), get_dataset_glass(), get_dataset_loan_approval()]

    for dataset in datasets:
        X, y = dataset
        label_column = y.name
        print("Zbiór", label_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        nbc_classifier = NBC(1)
        start = time.time()
        nbc_classifier.fit(X_train, y_train)
        print('Czas fit() impl. wł. :', time.time() - start)
        y_pred = nbc_classifier.predict(X_test)
        our_nbc_acc = metrics.accuracy_score(y_test, y_pred)

        clf = CategoricalNB()
        start = time.time()
        clf.fit(X_train, y_train)
        print('Czas fit() sklearn:', time.time() - start)
        y_pred_sklearn = clf.predict(X_test)
        sklearn_nbc_acc = metrics.accuracy_score(y_test, y_pred_sklearn)

        print(f"Accuracy score dla impl. wł.: {our_nbc_acc}")
        print(f"Accuracy score dla sklearn.: {sklearn_nbc_acc}")
        print('=========================================================')


def tree_number_influence():
    print('====================== EKSPERYMENT: Optymalizacja stosunku klasyfikatorów =============================')
    print('Badanie wpływu różnych proprocji między rodzajami klasyfikatorów w lesie lodowym')

    experiments_number = 3
    samples_percentage_list = [0.75]
    attributes_percentage_list = [0.75]
    classifiers = [NBC, ID3]
    classifiers_ratios = [[0.5, 0.5]]
    n = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    X, y = get_dataset_divorce()

    random_forest_experiments(experiments_number, X, y, n, samples_percentage_list, attributes_percentage_list,
                             classifiers, classifiers_ratios)

# Porównanie wpływu parametru proporcji między rodzajami klasyfikatorów na klasyfikację
def classifier_ratio_influence():
    print('====================== EKSPERYMENT: Optymalizacja stosunku klasyfikatorów =============================')
    print('Badanie wpływu różnych proprocji między rodzajami klasyfikatorów w lesie lodowym')

    experiments_number = 3
    samples_percentage_list = [0.75]
    attributes_percentage_list = [0.75]
    classifiers = [NBC, ID3]
    classifiers_ratios = [[0, 1], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1, 0]]

    X, y = get_dataset_divorce()

    random_forest_experiments(experiments_number, X, y, samples_percentage_list, attributes_percentage_list,
                             classifiers, classifiers_ratios)


# Porównanie wpływu ilości przykładów w węźle na klasyfikację
def examples_number_in_node_influence():
    print('====================== EKSPERYMENT: Optymalizacja stosunku klasyfikatorów =============================')
    print('Badanie wpływu różnych proprocji między rodzajami klasyfikatorów w lesie lodowym')


if __name__ == "__main__":
    id3_comparison()
    nbc_comparison()
