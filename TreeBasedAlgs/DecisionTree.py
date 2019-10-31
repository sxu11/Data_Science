

import pandas as pd

class Node:
    """
    Node of the tree. Contains:

    """
    def __init__(self):
        self._true = None
        self._false = None
        pass

    def build(self, df):

        pass

class DecisionTree:
    def __init__(self, df):
        self._root = Node()
        self._df = df

    def build(self):
        self._root.build(self._df)

def test():
    df = pd.DataFrame({'Color': ['Green', 'Yellow', 'Red', 'Red', 'Yellow'],
                       'Diam': [3, 3, 1, 1, 3],
                       'Label': ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon']})

    decisionTree = DecisionTree(df)
    print decisionTree._root._trueBranch

if __name__ == '__main__':
    test()