

import pandas as pd

class Node:
    """
    Node of the tree. Contains:

    """
    def __init__(self):
        self._true = None
        self._false = None
        pass

    def split(self, df):
        """
        Steps:
        Find the best question to ask, based on current df
            (1) best feature
            (2) best split
        Then under a question, split the node into two branches
        """

        pass

class DecisionTree:
    def __init__(self, df):
        self._root = Node()
        self._df = df

    def build(self):
        self._root.split(self._df)





"""
Input: two numbers (test if a>=b); two object (test if a==b) 
Output: bool True/False
"""
def ask(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return ask_numeric(a, b)
    elif isinstance(a, str) and isinstance(b, str):
        return ask_categorical(a, b)
    else:
        raise Exception("ask not implemented!")

def ask_numeric(a, b):
    return a >= b

def ask_categorical(a, b):
    return a == b


def test_question():
    print question("a", "a")


def test():
    df = pd.DataFrame({'Color': ['Green', 'Yellow', 'Red', 'Red', 'Yellow'],
                       'Diam': [3, 3, 1, 1, 3],
                       'Label': ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon']})

    decisionTree = DecisionTree(df)
    print decisionTree._root._trueBranch

if __name__ == '__main__':
    test_question()