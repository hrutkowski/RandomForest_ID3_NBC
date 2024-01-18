import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn import metrics
from itertools import product
from sklearn.naive_bayes import CategoricalNB

from algorithms.id3_classifier import ID3
from algorithms.nbc_classifier import NBC
from algorithms.random_forest_algorithm import RandomForest
from scripts_and_experiments.datasets_manager import get_dataset_corona, get_dataset_divorce, get_dataset_glass, \
    get_dataset_loan_approval_dataset
from scripts_and_experiments.experiment_scripts import cross_validation_score, test_accuracy, get_conf_matrix, \
    run_experiment


def experiment1():
    # Eksperyment 1
    # Porównanie własnej implementacji NBC z implementacją z biblioteki scikit-learn
    print('=================')
    print("Eksperyment 1\nPorównanie własnej implementacji NBC z implementacją z biblioteki scikit-learn")

    X, y = get_dataset_corona()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    nbc_classifier = NBC(1)
    start = time.time()
    nbc_classifier.fit(X_train, y_train, 'Corona')
    print('Czas fit() impl. wł. :', time.time() - start)
    y_pred = nbc_classifier.predict(X_test)
    our_preds = np.array(y_pred, dtype=int)
    our_nbc_acc = metrics.accuracy_score(y_test, our_preds)

    clf = CategoricalNB()
    start = time.time()
    clf.fit(X_train, y_train)
    print('Czas fit() sklearn:', time.time() - start)
    y_pred = clf.predict(X_test)
    sklearn_nbc_acc = metrics.accuracy_score(y_test, y_pred)

    print(f"Predykcje są takie same: {(our_preds == y_pred).all()}")
    print(f"Dokładność jest taka sama: {(our_preds == y_pred).all()} - {sklearn_nbc_acc}=={our_nbc_acc}")

def experiment3():
    # Eksperyment 3
    #5.3.	Wyniki ewaluacji lasu losowego z dobranymi parametrami
    # Eksperyment ten wykonano dla trzech zbiorów danych
    # W rzeczywistości puszczano go pojedynczo dla każdego zbioru, na 5 notatnikach google colab po 5 powtórzen
    # Tutaj jednocześnie robiony jest test klasycznej implementacji lasu losowego z samymi drzewami
    print('=================')
    print("Eksperyment 3\nWyniki ewaluacji lasu losowego z dobranymi parametrami")
    experiment_repetitions = 5
    dataset_loadDataset_valMethod = [("Corona", get_dataset_corona)]
    model_param_attribute_part = [0.75]
    model_param_instances_per_classifier = [1.0]
    model_param_id3_to_NBC = [[0.50, 0.50], [1, 0]]
    model_param_num_of_classifiers = [64]

    exp3_models_results_df = run_experiment("exp3", experiment_repetitions,
                dataset_loadDataset_valMethod,
                model_param_attribute_part,
                model_param_instances_per_classifier,
                model_param_id3_to_NBC,
                model_param_num_of_classifiers)
def experiment_eval_ID3_NBC(dataset="corona"):
    print('\n=====================')
    print(f"Only ID3 and only NBC evaluation on dataset {dataset}")
    if dataset=="corona":
        X, y = get_dataset_corona()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("\nNaive Bayes Classifier:")
    nbc_classifier = NBC()
    nbc_classifier.fit(X_train, y_train, 'Corona')
    y_pred = nbc_classifier.predict(X_test)
    our_preds = np.array(y_pred, dtype=int)
    nbc_acc = metrics.accuracy_score(y_test, our_preds)
    print(f"NBC: {nbc_acc}")

    print("\nID3:")
    id3_tree = ID3(min_sample_num=5)
    id3_tree.fit(X_train, y_train)
    id3_acc = metrics.accuracy_score(X_test, y_test)

    print(f"ID3: {id3_acc}")

if __name__ == "__main__":
    experiment3()
