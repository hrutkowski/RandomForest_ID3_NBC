import pandas as pd
import numpy as np
import seaborn as sns
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


def random_forest_experiments(experiments_number: int, X, y, n_list: List[int], samples_percentage_list: List[float],
                              attributes_percentage_list: List[float], classifiers: List[float],
                              classifiers_ratios: List[List[float]]):
    accuracies = []
    f1_scores = []
    conf_matrices = []

    for _ in range(experiments_number):
        for samples_percentage in samples_percentage_list:
            for attributes_percentage in attributes_percentage_list:
                for classifiers_ratio in classifiers_ratios:
                    for n in n_list:
                        random_forest = RandomForest(n, samples_percentage, attributes_percentage, classifiers, classifiers_ratio)
                        accuracy, f1_score, conf_matrix = eval_cross_validation(X, y, random_forest)
                        accuracies.append(accuracy)
                        f1_scores.append(f1_score)
                        conf_matrices.append(conf_matrix)
    
    # TO TRZEBA JAKOŚ OGARNĄC SENSOWNIE
    return (round(np.mean(accuracies)), round(np.mean(f1_scores)),
            np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))



def eval_cross_validation(X, y, model, splits_number: int = 5) -> Tuple[float, float, np.ndarray]:
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

    return (round(np.mean(accuracies)), round(np.mean(f1_scores)),
            np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_labels: list):
    sns.set(font_scale=1.4)
    plt.figure(figsize=(8, 6))

    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Confusion Matrix')
    plt.show()
