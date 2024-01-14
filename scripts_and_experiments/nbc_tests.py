from datasets_manager import *
from sklearn.model_selection import train_test_split
from algorithms.nbc import NBC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

corona_df = get_dataset_corona()


def test_nbc_for_corona():
    train_df, test_df = train_test_split(corona_df, test_size=0.2, random_state=42)
    className = 'Corona'

    X_train = train_df.loc[:, train_df.columns != className]
    y_train = train_df.loc[:, train_df.columns == className]
    X_test = test_df.loc[:, test_df.columns != className]
    y_test = test_df.loc[:, test_df.columns == className]

    nbc = NBC(1)
    nbc.fit(X_train, y_train)
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