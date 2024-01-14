import pandas as pd
import numpy as np
from sklearn import metrics
from id3_classifier import ID3
from nbc_classifier import NBC
from typing import List, Tuple


class RandomForest:

    def __init__(self, n: int = 100, samples_percentage: float = 0.5, attributes_percentage: float = 0.5,
                 classifier_list: List = None, classifier_ratio_list: List[float] = None):
        assert len(classifier_list) == len(classifier_ratio_list), \
            "List of classifiers and classifier ratios must have the same size!"
        self.n = n
        self.samples_percentage = samples_percentage
        self.attributes_percentage = attributes_percentage
        if classifier_list is None:
            self.classifier_list = [ID3, NBC]
        if classifier_ratio_list is None:
            self.classifier_ratio_list = [0.5, 0.5]
        self.forest = []
        self.attributes = []

    def fit(self, X_train, y_train):
        for classifier, ratio in zip(self.classifier_list, self.classifier_ratio_list):
            for _ in range(round(self.n * ratio)):
                clf = classifier()
                X_train_bagging, y_train_bagging = self.bagging_data(X_train.copy(), y_train.copy())
                clf.fit(X_train_bagging.copy(), y_train_bagging.copy())
                self.forest.append(clf)

    def bagging_data(self, X_train, y_train):
        X_train = X_train.sample(frac=self.attributes_percentage, axis=1)
        self.attributes.append(X_train.columns)
        X_train['results'] = y_train
        X_train = X_train.sample(frac=self.attributes_percentage)
        return X_train.drop(columns=['results']), X_train['results']

    def predict(self, X_test) -> pd.Series:
        predictions = np.empty([len(self.forest), len(X_test)])
        for i in range(len(self.forest)):
            predictions[i] = np.array(self.forest[i].predict(X_test[self.attributes[i]]))
        pred_df = pd.DataFrame(predictions)
        pred_df = pd.DataFrame.mode(pred_df, axis=0).T
        return pd.Series(pred_df[0])

    def get_scores(self, X_test, y_test) -> Tuple[float, float]:
        y_pred = np.array(self.predict(X_test))
        y_test = np.array(y_test)
        return metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average='macro')

    def eval(self, X_test, y_test) -> Tuple[float, float]:
        accuracy, f1_score = self.get_scores(X_test, y_test)
        print(f"Accuracy: {accuracy}")
        print(f"F1 score: {f1_score}")
        return accuracy, f1_score
