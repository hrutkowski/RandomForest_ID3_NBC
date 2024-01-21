import numpy as np
import time

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from algorithms.id3_classifier import ID3
from algorithms.nbc_classifier import NBC
from experients_helpers import *
from scripts_and_experiments.datasets_manager import get_dataset_corona, get_dataset_divorce, get_dataset_glass, \
    get_dataset_loan_approval


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
    print('====================== EKSPERYMENT: Optymalizacja liczby drzew =============================')
    print('Badanie wpływu różnych liczby drzew w lesie losowym')

    experiments_number = 3
    n = [10, 100]
    samples_percentage = 0.75
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [0.5, 0.5]

    X, y = get_dataset_divorce()
    class_name = y.name

    acc, acc_std, f1, f1_std, avg_conf_mtx = rf_experiment_tree_number(experiments_number, X, y, n,
                                                                       samples_percentage, attributes_percentage,
                                                                       classifiers, classifiers_ratios)

    print(f"Accuracy = {acc}")
    print(f"Accuracy std = {acc_std}")
    print(f"F1 Score = {f1}")
    print(f"F1 Score std = {f1_std}")
    print(f"CONF MATRIX AVG")
    print(avg_conf_mtx)

    plot_results(n, acc, acc_std, 'n', 'Accuracy', class_name)
    plot_results(n, f1, f1_std, 'n ratio', 'F1 Score', class_name)


# Porównanie wpływu parametru proporcji między rodzajami klasyfikatorów na klasyfikację
def classifier_ratio_influence():
    print('====================== EKSPERYMENT: Optymalizacja stosunku klasyfikatorów =============================')
    print('Badanie wpływu różnych proporcji między rodzajami klasyfikatorów w lesie losowym')

    experiments_number = 3
    n = 10
    samples_percentage = 0.75
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [[0, 1], [1, 0]]

    X, y = get_dataset_divorce()
    class_name = y.name

    acc, acc_std, f1, f1_std, avg_conf_mtx = rf_experiment_classifier_ratio(experiments_number, X, y, n,
                                                                            samples_percentage, attributes_percentage,
                                                                            classifiers, classifiers_ratios)

    print(f"Accuracy = {acc}")
    print(f"Accuracy std = {acc_std}")
    print(f"F1 Score = {f1}")
    print(f"F1 Score std = {f1_std}")
    print(f"CONF MATRIX AVG")
    print(avg_conf_mtx)

    plot_results(classifiers_ratios, acc, acc_std, 'Classifiers ratio', 'Accuracy', class_name)
    plot_results(classifiers_ratios, f1, f1_std, 'Classifiers ratio', 'F1 Score', class_name)


# Porównanie wpływu ilości przykładów w węźle na klasyfikację
def samples_percentage_influence():
    print('====================== EKSPERYMENT: Optymalizacja liczby przykładów w węźle =============================')
    print('Badanie wpływu różnej liczby przykładów w weźle w lesie losowym')

    experiments_number = 3
    n = 10
    samples_percentage = [0.25, 0.75]
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [0.5, 0.5]

    X, y = get_dataset_divorce()
    class_name = y.name

    acc, acc_std, f1, f1_std, avg_conf_mtx = rf_experiment_samples_percentages(experiments_number, X, y, n,
                                                                               samples_percentage,
                                                                               attributes_percentage,
                                                                               classifiers, classifiers_ratios)

    print(f"Accuracy = {acc}")
    print(f"Accuracy std = {acc_std}")
    print(f"F1 Score = {f1}")
    print(f"F1 Score std = {f1_std}")
    print(f"CONF MATRIX AVG")
    print(avg_conf_mtx)

    plot_results(samples_percentage, acc, acc_std, 'Samples percentage', 'Accuracy', class_name)
    plot_results(samples_percentage, f1, f1_std, 'Samples percentage', 'F1 Score', class_name)


if __name__ == "__main__":
    # id3_comparison()
    # id3_comparison()
    # nbc_comparison()
    classifier_ratio_influence()
    samples_percentage_influence()
