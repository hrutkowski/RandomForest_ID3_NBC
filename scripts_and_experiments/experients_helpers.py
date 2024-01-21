import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import CategoricalNB
from typing import List, Tuple, Iterable
from algorithms.random_forest_algorithm import RandomForest


def rf_experiment_classifier_ratio(experiments_number: int, X, y, n: int, samples_percentage: float,
                                   attributes_percentage: float, classifiers: List,
                                   classifiers_ratios: List[List[float]]) \
        -> Tuple[List[float], List[float], List[float], List[float], np.ndarray]:
    final_accuracies = []
    final_accuracies_std_list = []
    final_f1_scores = []
    final_f1_scores_std_list = []
    final_conf_matrices = []

    for classifiers_ratio in classifiers_ratios:
        accuracies = []
        accuracies_std_list = []
        f1_scores = []
        f1_scores_std_list = []
        conf_matrices = []
        random_forest = RandomForest(n, samples_percentage, attributes_percentage, classifiers, classifiers_ratio)

        for i in range(experiments_number):
            acc, acc_std, f1, f1_std, conf_matrix = eval_cross_validation(X, y, random_forest)
            accuracies.append(acc)
            accuracies_std_list.append(acc_std)
            f1_scores.append(f1)
            f1_scores_std_list.append(f1_std)
            conf_matrices.append(conf_matrix)
            print(f"Experiment nr {i + 1}")
        print(f"Po eksperymentach accuracy: {accuracies}")
        final_accuracies.append(round(np.mean(accuracies), 2))
        final_f1_scores.append(round(np.mean(f1_scores), 2))
        final_conf_matrices.append(np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))
        # Wyliczanie złożonego odchylenia standardowego jako pierwiastek z sumy kwadratów odchyleń standardowych
        final_accuracies_std_list.append(round(np.sqrt(np.mean(accuracies_std_list) ** 2 + np.std(accuracies) ** 2), 2))
        final_f1_scores_std_list.append(round(np.sqrt(np.mean(f1_scores_std_list) ** 2 + np.std(f1_scores) ** 2), 2))

    return (final_accuracies, final_accuracies_std_list, final_f1_scores, final_f1_scores_std_list,
            np.round(np.sum(final_conf_matrices, axis=0) / len(final_conf_matrices)))


def rf_experiment_samples_percentages(experiments_number: int, X, y, n: int, samples_percentages: List[float],
                                      attributes_percentage: float, classifiers: List,
                                      classifiers_ratio: List[float]) \
        -> Tuple[List[float], List[float], List[float], List[float], np.ndarray]:
    final_accuracies = []
    final_accuracies_std_list = []
    final_f1_scores = []
    final_f1_scores_std_list = []
    final_conf_matrices = []

    for samples_percentage in samples_percentages:
        accuracies = []
        accuracies_std_list = []
        f1_scores = []
        f1_scores_std_list = []
        conf_matrices = []
        random_forest = RandomForest(n, samples_percentage, attributes_percentage, classifiers, classifiers_ratio)

        for i in range(experiments_number):
            acc, acc_std, f1, f1_std, conf_matrix = eval_cross_validation(X, y, random_forest)
            accuracies.append(acc)
            accuracies_std_list.append(acc_std)
            f1_scores.append(f1)
            f1_scores_std_list.append(f1_std)
            conf_matrices.append(conf_matrix)
            print(f"Experiment nr {i + 1}")
        print(f"Po eksperymentach accuracy: {accuracies}")
        final_accuracies.append(round(np.mean(accuracies), 2))
        final_f1_scores.append(round(np.mean(f1_scores), 2))
        final_conf_matrices.append(np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))
        # Wyliczanie złożonego odchylenia standardowego jako pierwiastek z sumy kwadratów odchyleń standardowych
        final_accuracies_std_list.append(np.sqrt(np.mean(accuracies_std_list) ** 2) + np.std(accuracies) ** 2)
        final_f1_scores_std_list.append(np.sqrt(np.mean(f1_scores_std_list) ** 2) + np.std(f1_scores) ** 2)

    return (final_accuracies, final_accuracies_std_list, final_f1_scores, final_f1_scores_std_list,
            np.round(np.sum(final_conf_matrices, axis=0) / len(final_conf_matrices)))


def rf_experiment_tree_number(experiments_number: int, X, y, n_list: List[int], samples_percentage: float,
                              attributes_percentage: float, classifiers: List, classifiers_ratio: List[float]) \
        -> Tuple[List[float], List[float], List[float], List[float], np.ndarray]:
    final_accuracies = []
    final_accuracies_std_list = []
    final_f1_scores = []
    final_f1_scores_std_list = []
    final_conf_matrices = []

    for n in n_list:
        accuracies = []
        accuracies_std_list = []
        f1_scores = []
        f1_scores_std_list = []
        conf_matrices = []
        random_forest = RandomForest(n, samples_percentage, attributes_percentage, classifiers, classifiers_ratio)

        for i in range(experiments_number):
            acc, acc_std, f1, f1_std, conf_matrix = eval_cross_validation(X, y, random_forest)
            accuracies.append(acc)
            accuracies_std_list.append(acc_std)
            f1_scores.append(f1)
            f1_scores_std_list.append(f1_std)
            conf_matrices.append(conf_matrix)
            print(f"Experiment nr {i + 1}")
        print(f"Po eksperymentach accuracy: {accuracies}")
        final_accuracies.append(round(np.mean(accuracies), 2))
        final_f1_scores.append(round(np.mean(f1_scores), 2))
        final_conf_matrices.append(np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))
        # Wyliczanie złożonego odchylenia standardowego jako pierwiastek z sumy kwadratów odchyleń standardowych
        final_accuracies_std_list.append(np.sqrt(np.mean(accuracies_std_list) ** 2) + np.std(accuracies) ** 2)
        final_f1_scores_std_list.append(np.sqrt(np.mean(f1_scores_std_list) ** 2) + np.std(f1_scores) ** 2)

    return (final_accuracies, final_accuracies_std_list, final_f1_scores, final_f1_scores_std_list,
            np.round(np.sum(final_conf_matrices, axis=0) / len(final_conf_matrices)))


def eval_cross_validation(X, y, model, splits_number: int = 5) -> Tuple[float, float, float, float, np.ndarray]:
    accuracies = []
    f1_scores = []
    conf_matrices = []

    kf = KFold(n_splits=splits_number, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X, y):
        X_train, y_train = X.loc[train_index, :], y.loc[train_index]
        X_test, y_test = X.loc[test_index, :], y.loc[test_index]
        model.fit(X_train, y_train)
        accuracy, f1_score, conf_matrix = model.eval(X_test, y_test)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        conf_matrices.append(conf_matrix)

    return (np.mean(accuracies), np.std(accuracies, axis=0), np.mean(f1_scores),
            np.std(f1_scores, axis=0), np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))


def plot_confusion_matrix(conf_mtx: np.ndarray, class_labels: list):
    sns.set(font_scale=1.4)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Confusion Matrix')
    plt.show()


def format_label(val):
    if isinstance(val[0], (list, tuple)):
        return '[' + ', '.join(map(str, val)) + ']'
    else:
        return str(val)


def plot_results(x_val: List, y_val: List[float], y_std_val: List[float], x_label: str, y_label: str, class_name: str):
    plt.figure(figsize=(8, 6))

    if isinstance(x_val[0], Iterable):
        x_ticks = [format_label(val) for val in x_val]
    else:
        x_ticks = x_val
    plt.errorbar(x_ticks, y_val, yerr=y_std_val, fmt='o', label=x_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} = f({x_label})")
    plt.grid(True)

    file_name = f'../images/{class_name}_plot.png'
    plt.savefig(file_name)
