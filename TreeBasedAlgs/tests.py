
from sklearn.datasets import load_breast_cancer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from classes import DecisionTree

import numpy as np

import matplotlib.pyplot as plt

def test():
    df = pd.DataFrame({'Color': ['Green', 'Yellow', 'Red', 'Red', 'Yellow'],
                       'Diam': [3, 3, 1, 1, 3],
                       'Label': ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon']})

    decisionTree = DecisionTree(df)
    decisionTree.build()

    print(decisionTree._root._trueNode.df)
    print(decisionTree._root._falseNode.df)

def loadXy():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.33,
                                                        random_state = 42)
    return X_train, X_test, y_train, y_test

def getAuc(clf, verbose=False):
    X_train, X_test, y_train, y_test = loadXy()

    clf.fit(X_train, y_train)

    if verbose:
        print(clf)

    y_pred = clf.predict(X_test)
    accuracy = roc_auc_score(y_test, y_pred)
    return (accuracy)

def getAucsWithNSeeds(clf, n, params=None):
    ress = []
    for i in range(n):
        if params is None:
            params = {"random_state": i}
        else:
            params["random_state"] = i

        res = getAuc(clf(**params))
        ress.append(res)
    return ress

def plotHist(dataVect, xlabel=""):
    plt.hist(dataVect)
    plt.xlabel(xlabel)
    plt.show()


def main():
    aucsDt = getAucsWithNSeeds(DecisionTree, n=1)
    print(aucsDt)
    quit()
    plotHist(aucsDt, np.mean(aucsDt))

    aucsSklearnDt = getAucsWithNSeeds(DecisionTreeClassifier, n=100)
    plotHist(aucsSklearnDt, np.mean(aucsSklearnDt))

    params = {"n_estimators": 1, "max_features": None}
    aucsSklearnRf = getAucsWithNSeeds(RandomForestClassifier, n=100, params=params)
    plotHist(aucsSklearnRf, np.mean(aucsSklearnRf))




if __name__ == '__main__':
    main()