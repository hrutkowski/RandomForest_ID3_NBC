from datasets_manager import *
from sklearn.model_selection import train_test_split
from algorithms.nbc import NBC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

X, y = get_dataset_corona()


def test_nbc_for_corona():
    className = 'Corona'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print("Dimension of X:", X.shape)
    # print("Dimension of y:", y.shape)
    nbc = NBC(1)
    nbc.fit(X_train, y_train, className)
    scores = nbc.accuracy_score(X_test, y_test)
    print(scores)

    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    return scores


test_nbc_for_corona()