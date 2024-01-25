import time
from sklearn.tree import DecisionTreeClassifier
from algorithms.id3_classifier import ID3
from algorithms.nbc_classifier import NBC
from experients_helpers import *
from scripts_and_experiments.datasets_manager import get_dataset_corona, get_dataset_divorce, get_dataset_glass, \
    get_dataset_loan_approval, load_proper_dataset, get_class_labels_for_dataset


# Porównanie klasyfikacji przy użyciu wybranej przez Nas gotowej i lekko przerobionej pod Nasze
# potrzeby implementacji ID3 z gotową implementacją z biblioteki sklearn
def id3_comparison():
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


# Porównanie wpływu liczby drzew na klasyfikację
def tree_number_influence(dataset_name: str):
    experiments_number = 5
    n = [10, 20, 50, 100]
    samples_percentage = 0.75
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [0.5, 0.5]

    X, y = load_proper_dataset(dataset_name)
    class_labels = get_class_labels_for_dataset(dataset_name)

    acc, acc_std, f1, f1_std, avg_conf_mtx = rf_experiment_tree_number(experiments_number, X, y, n,
                                                                       samples_percentage, attributes_percentage,
                                                                       classifiers, classifiers_ratios)

    print(f"Accuracy = {acc}")
    print(f"Accuracy std = {acc_std}")
    print(f"F1 Score = {f1}")
    print(f"F1 Score std = {f1_std}")
    print(f"CONF MATRIX AVG")
    print(avg_conf_mtx)

    plot_results(n, acc, acc_std, 'number of trees', 'Accuracy', dataset_name, 'number_of_trees')
    plot_results(n, f1, f1_std, 'number of trees', 'F1 Score', dataset_name, 'number_of_trees')
    generate_excel_table(n, acc, acc_std, f1, f1_std, 'Number of trees', 'Accuracy', 'F1_Score',
                         dataset_name, 'number_of_trees')
    plot_confusion_matrix(avg_conf_mtx, class_labels, dataset_name, 'number_of_trees')


# Porównanie wpływu parametru proporcji między rodzajami klasyfikatorów na klasyfikację
def classifier_ratio_influence(dataset_name: str):
    print('====================== EKSPERYMENT: Optymalizacja stosunku klasyfikatorów =============================')
    print('Badanie wpływu różnych proporcji między rodzajami klasyfikatorów w lesie losowym')

    experiments_number = 1
    n = 6
    samples_percentage = 0.75
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [[0, 1], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1, 0]]

    X, y = load_proper_dataset(dataset_name)
    class_labels = get_class_labels_for_dataset(dataset_name)

    acc, acc_std, f1, f1_std, avg_conf_mtx = rf_experiment_classifier_ratio(experiments_number, X, y, n,
                                                                            samples_percentage, attributes_percentage,
                                                                            classifiers, classifiers_ratios)

    print(f"Accuracy = {acc}")
    print(f"Accuracy std = {acc_std}")
    print(f"F1 Score = {f1}")
    print(f"F1 Score std = {f1_std}")
    print(f"CONF MATRIX AVG")
    print(avg_conf_mtx)

    plot_results(classifiers_ratios, acc, acc_std, 'Classifiers ratio', 'Accuracy', dataset_name, 'classifiers_ratios')
    plot_results(classifiers_ratios, f1, f1_std, 'Classifiers ratio', 'F1_Score', dataset_name, 'classifiers_ratios')
    generate_excel_table(classifiers_ratios, acc, acc_std, f1, f1_std, 'Classifiers ratio', 'Accuracy', 'F1_Score', dataset_name, 'classifiers_ratios')
    plot_confusion_matrix(avg_conf_mtx, class_labels, dataset_name, 'classifiers_ratios')


# Porównanie wpływu ilości przykładów w węźle na klasyfikację
def samples_percentage_influence(dataset_name: str):
    experiments_number = 1
    n = 2
    samples_percentage = [0.2]
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [0.5, 0.5]

    X, y = load_proper_dataset(dataset_name)
    class_labels = get_class_labels_for_dataset(dataset_name)

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

    plot_results(samples_percentage, acc, acc_std, 'Samples percentage', 'Accuracy', dataset_name, 'samples_percentage')
    plot_results(samples_percentage, f1, f1_std, 'Samples percentage', 'F1 Score', dataset_name, 'samples_percentage')
    generate_excel_table(samples_percentage, acc, acc_std, f1, f1_std, 'Samples percentage', 'Accuracy', 'F1_Score', dataset_name, 'samples_percentage')
    plot_confusion_matrix(avg_conf_mtx, class_labels, dataset_name, 'samples_percentage')


if __name__ == "__main__":
    #tree_number_influence('loan_approval')
    #tree_number_influence('divorce')

    #classifier_ratio_influence('letter')
    samples_percentage_influence('corona')
    #classifier_ratio_influence('divorce')
    #samples_percentage_influence('corona')
    #
    #tree_number_influence('loan_approval')
    #classifier_ratio_influence('corona')
    #samples_percentage_influence('corona')


