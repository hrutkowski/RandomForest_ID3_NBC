#https://www.kaggle.com/code/hinsontsui/iris-prediction-id3-decision-tree

import numpy as np


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


class TreeNode:
    """
    A recursively defined data structure to store a tree.
    Each node can contain other nodes as its children
    """

    def __init__(self, node_name="", min_sample_num=10, default_decision=None):
        self.children = {}  # Sub nodes --
        # recursive, those elements of the same type (TreeNode)
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
            #print("Decision:", self.decision)
            return self.decision
        else:
            # this node is an internal one, further queries about an attribute
            # of the data is needed.
            attr_val = sample[self.split_feat_name]
            child = self.children.get(attr_val, None)

            if child is None:
                # Handle the case where the attribute value is not in self.children
             #   print(f"Warning: Attribute value {attr_val} not found in children.")
                return self.default_decision

            # uncomment to get log information of code execution
           # print("Testing ", self.split_feat_name, "->", attr_val)
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

        #print(self.name, "received", len(X), "samples")
        if len(X) < self.min_sample_num:
            # If the data is empty when this node is arrived,
            # we just make an arbitrary decision
            if len(X) == 0:
                self.decision = self.default_decision
         #       print("DECISION", self.decision)
            else:
                self.decision = y.mode()[0]
         #       print("DECISION", self.decision)
            return
        else:
            unique_values = y.unique()
            if len(unique_values) == 1:
                self.decision = unique_values[0]
          #      print("DECISION", self.decision)
                return
            else:
                info_gain_max = 0
                for a in X.keys():  # Examine each attribute
                    aig = compute_info_gain(X, a, y)
                    if aig > info_gain_max:
                        info_gain_max = aig
                        self.split_feat_name = a
           #     print(f"Split by {self.split_feat_name}, IG: {info_gain_max:.2f}")
                self.children = {}
                for v in X[self.split_feat_name].unique():
                    print(f"Processing value: {v}")
                    if v is not None:
                        index = X[self.split_feat_name] == v
                        if v not in self.children:
                            print(f"Creating child for value: {v}")
                            self.children[v] = TreeNode(
                                node_name=self.name + ":" + self.split_feat_name + "==" + str(v),
                                min_sample_num=self.min_sample_num,
                                default_decision=self.default_decision
                            )
                        #print(X[index], y[index])
                        self.children[v].fit(X[index], y[index])
                    else:
                        print("Warning: Unexpected value 'None' in", self.split_feat_name)

