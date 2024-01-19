# Link to GitHub repo: https://www.kaggle.com/code/hinsontsui/iris-prediction-id3-decision-tree
# Classifier modified for out needs

import numpy as np
from sklearn import metrics

def compute_entropy(y):
    """
    :param y: The data samples of a discrete distribution
    """
    if len(y) < 2:  # a trivial case
        return 0
    freq = np.array(y.value_counts(normalize=True))
    return -(freq * np.log2(freq + 1e-6)).sum()  # the small eps for
    # safe numerical computation


def compute_info_gain(samples, attr, target):
    values = samples[attr].value_counts(normalize=True)
    split_ent = 0
    for v, fr in values.items():
        index = samples[attr] == v
        sub_ent = compute_entropy(target[index])
        split_ent += fr * sub_ent

    ent = compute_entropy(target)
    return ent - split_ent


class ID3:
    """
    A recursively defined data structure to store a tree.
    Each node can contain other nodes as its children
    """

    def __init__(self, node_name="", min_sample_num=10, default_decision=None):
        self.children = {}  # Sub nodes --
        # recursive, those elements of the same type (ID3)
        self.decision = None  # Undecided
        self.split_feat_name = None  # Splitting feature
        self.name = node_name
        self.default_decision = default_decision
        self.min_sample_num = min_sample_num

    def pretty_print(self, prefix=''):
        if self.split_feat_name is not None:
            for k, v in self.children.items():
                v.pretty_print(f"{prefix}:When {self.split_feat_name} is {k}")
                # v.pretty_print(f"{prefix}:{k}:")
        else:
            print(f"{prefix}:{self.decision}")

    def predict(self, sample):
        if self.decision is not None:
            # uncomment to get log information of code execution
            print("Decision:", self.decision)
            return self.decision
        else:
            if self.split_feat_name is None or self.split_feat_name not in sample:
                # Handle the case when split feature is None or not in the sample
                print("Invalid split feature or feature not present in sample.")
                return self.default_decision  # You might want to define a default decision or handle it differently

            attr_val = sample[self.split_feat_name]
            if attr_val not in self.children:
                # Handle the case when the attribute value is not in the children
                print("Attribute value not found in children.")
                return self.default_decision  # You might want to define a default decision or handle it differently

            child = self.children[attr_val]
            # uncomment to get log information of code execution
            print("Testing ", self.split_feat_name, "->", attr_val)
            return child.predict(sample)

    def fit(self, X, y):
        """
        The function accepts a training dataset, from which it builds the tree
        structure to make decisions or to make children nodes (tree branches)
        to do further inquiries
        :param X: [n * p] n observed data samples of p attributes
        :param y: [n] target values
        """
        if self.default_decision is None:
            self.default_decision = y.mode()[0]

        print(self.name, "received", len(X), "samples")
        if len(X) < self.min_sample_num:
            # If the data is empty when this node is arrived,
            # we just make an arbitrary decision
            if len(X) == 0:
                self.decision = self.default_decision
                print("DECISION", self.decision)
            else:
                self.decision = y.mode()[0]
                print("DECISION", self.decision)
            return
        else:
            unique_values = y.unique()
            if len(unique_values) == 1:
                self.decision = unique_values[0]
                print("DECISION", self.decision)
                return
            else:
                info_gain_max = 0
                valid_features = [a for a in X.keys() if
                                  len(X[a].unique()) > 1]  # Filter out features with only one unique value
                if not valid_features:
                    # Handle the case when there are no valid features to split on
                    print("No valid features to split on.")
                    return
                for a in valid_features:  # Examine each valid attribute
                    aig = compute_info_gain(X, a, y)
                    print(a, aig)
                    if aig > info_gain_max:
                        info_gain_max = aig
                        self.split_feat_name = a
                print(f"Split by {self.split_feat_name}, IG: {info_gain_max:.2f}")
                self.children = {}
                for v in X[self.split_feat_name].unique():
                    index = X[self.split_feat_name] == v
                    self.children[v] = ID3(
                        node_name=self.name + ":" + self.split_feat_name + "==" + str(v),
                        min_sample_num=self.min_sample_num,
                        default_decision=self.default_decision)
                    self.children[v].fit(X[index], y[index])

    def predict_all(self, X, y):
        predictions = []
        err_fp = 0
        err_fn = 0
        for (_, ct), tgt in zip(X.iterrows(), y):
            a = self.predict(ct)
            predictions.append(a)
            if a and not tgt:
                err_fp += 1
            elif not a and tgt:
                err_fn += 1
        return predictions, err_fp, err_fn

    # TO DO ??? (DODANE NA SZYBKO)
    def eval(self, X_test, y_test):
        acc, f1 = self.scores(X_test, y_test)
        print('Accuracy:', acc)
        print('F1 score: ', f1)
        return acc, f1

    def scores(self, X_test, y_test):
        X_test = np.array(X_test)
        y_pred = self.predict(X_test)
        y_pred = np.array(y_pred, dtype=str)
        y_test = np.array(y_test, dtype=str)
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
        return acc, f1