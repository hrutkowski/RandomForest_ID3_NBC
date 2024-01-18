from sklearn import metrics
import pandas as pd


def prepare_data(X_set, y_set, classColumn):
    X = pd.DataFrame(X_set)
    y = pd.DataFrame(y_set)
    df = pd.concat([X, y], axis=1)
    X = df.loc[:, df.columns != classColumn]
    y = df.loc[:, df.columns == classColumn]
    return X, y

class NBC:

    def __init__(self, alpha=None):
        self.alpha = 1.0 if alpha is None else alpha
        self.labels = None
        self.attributes = None
        self.classColumn = None
        self.valuesPerAttribute = {}  # storing number of unique values per attribute
        self.conditionalProbabilities = {}
        self.pClass = {}
        self.nClassOccurrences = {}

    def count_conditional_probabilities(self, df):
        for label in self.labels:
            self.nClassOccurrences[label] = df[self.classColumn].value_counts()[label]
            self.pClass[label] = (self.nClassOccurrences[label] + self.alpha) / (
                    len(df.index) + self.alpha * len(self.labels))
            df_for_label = df[df[self.classColumn] == label]
            for attribute in self.attributes:
                uniqueAttributes = df[attribute].unique().tolist()
                for attributeValue in uniqueAttributes:
                    attrForClassOccurrences = len(df_for_label[df_for_label[attribute] == attributeValue])
                    self.conditionalProbabilities[(label, attribute, attributeValue)] = (
                        attrForClassOccurrences + self.alpha) / (
                            len(df_for_label) + self.alpha *
                            self.valuesPerAttribute[attribute])

    def fit(self, X_train, y_train, className):
        X_train, y_train = prepare_data(X_train, y_train, className)
        df = X_train.join(y_train)

        # labels
        self.classColumn = y_train.keys().tolist()[0]
        self.labels = y_train[self.classColumn].unique()

        # attributes
        self.attributes = X_train.keys().tolist()

        for attr in self.attributes:
            self.valuesPerAttribute[attr] = X_train[attr].nunique()

        self.count_conditional_probabilities(df)

    def predict(self, X):
        return X.apply(lambda row: self.predict_row(row.values.flatten().tolist()), axis=1)

    def predict_row(self, row):
        p = float('-inf')
        predictedClass = 'undefined'

        for label in self.labels:
            pClass = self.pClass[label]
            for i in range(len(self.attributes)):
                attr_value = row[i]
                if (label, self.attributes[i], attr_value) in self.conditionalProbabilities:
                    pClass = pClass * self.conditionalProbabilities[(label, self.attributes[i], attr_value)]
                else:
                    nvals = float(self.valuesPerAttribute[self.attributes[i]])
                    pClass = pClass * self.alpha / (self.nClassOccurrences[label] + nvals * self.alpha)
            if pClass > p:
                predictedClass = label
                p = pClass
        return predictedClass

    def accuracy_score(self, X, y):
        y_pred = self.predict(X)
        y = y.values.flatten().tolist()
        acc = metrics.accuracy_score(y, y_pred)
        f1 = metrics.f1_score(y, y_pred, average='macro')
        return acc, f1