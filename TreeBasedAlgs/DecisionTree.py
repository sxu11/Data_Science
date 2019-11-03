
import numpy as np
import pandas as pd

def calcSquaredError(predicts, actual):
    """

    :param predicts: Numpy array
    :param actual: number
    :return:
    """
    numerify = [0 if x == actual else 1 for x in predicts]

    return sum(numerify)

class Node:
    """
    Node of the tree. Contains:

    """
    def __init__(self, df=None):
        self._trueNode = None
        self._falseNode = None
        self.df = df
        pass

    def getSplits(self, feature):
        if self.df[feature].dtype == int:
            vector = np.sort(self.df[feature].unique())
            return ((vector[1:] + vector[:-1])/2.).tolist()

        elif self.df[feature].dtype == object:
            return self.df[feature].unique().tolist()

        else:
            raise Exception("%s not implemented for getSplits()!" % self.df[feature].dtype)

    def getBranches(self, feature, split):
        # if self.df[feature].dtype == int:
        #     pass
        # elif self.df[feature].dtype == object:
        #     pass
        trueMasks = self.df[feature].apply(lambda x: ask(split, x))
        trueBranch = self.df[trueMasks]
        falseBranch = self.df[~trueMasks]
        return trueBranch, falseBranch


    def split(self):
        """
        Steps:
        Find the best question to ask, based on current df
            (1) best feature
            (2) best split
        Then under a question, split the node into two branches
        """
        features = self.df.columns[:-1].values.tolist()
        target = self.df.columns[-1]

        minSquaredError = 100000 # TODO
        bestSplitDfs = []
        for feature in features:
            splits = self.getSplits(feature)
            for split in splits:
                trueDf, falseDf = self.getBranches(feature, split)

                truePredict = trueDf[target].value_counts().idxmax()
                falsePredict = falseDf[target].value_counts().idxmax()
                trueSquaredError = calcSquaredError(trueDf[target].values, truePredict)
                falseSquaredError = calcSquaredError(falseDf[target].values, falsePredict)
                curSquaredError = trueSquaredError + falseSquaredError
                if curSquaredError < minSquaredError:
                    minSquaredError = curSquaredError
                    bestSplitDfs = [trueDf, falseDf]

        self._trueNode = Node(bestSplitDfs[0])
        self._falseNode = Node(bestSplitDfs[1])


class DecisionTree:
    def __init__(self, df):
        self._root = Node(df)

    def build(self):
        self._root.split()





"""
Input: two numbers (test if a>=b); two object (test if a==b) 
Output: bool True/False
"""
def ask(a, b):
    if isinstance(a, (np.int64, int, float)) and isinstance(b, (np.int64, int, float)):
        return ask_numeric(a, b)
    elif isinstance(a, str) and isinstance(b, str):
        return ask_categorical(a, b)
    else:
        raise Exception("ask not implemented for %s and %s!" % (type(a), type(b)))

def ask_numeric(a, b):
    return a >= b

def ask_categorical(a, b):
    return a == b


def test_question():
    # print question("a", "a")
    pass


def test():
    df = pd.DataFrame({'Color': ['Green', 'Yellow', 'Red', 'Red', 'Yellow'],
                       'Diam': [3, 3, 1, 1, 3],
                       'Label': ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon']})

    decisionTree = DecisionTree(df)
    decisionTree.build()
    # print(df['Color'].value_counts().idxmax())

    print(decisionTree._root._trueNode.df)
    print(decisionTree._root._falseNode.df)

if __name__ == '__main__':
    test()