
from sklearn.datasets import load_breast_cancer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def test():
    df = pd.DataFrame({'Color': ['Green', 'Yellow', 'Red', 'Red', 'Yellow'],
                       'Diam': [3, 3, 1, 1, 3],
                       'Label': ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon']})

    decisionTree = DecisionTree(df)
    decisionTree.build()
    # print(df['Color'].value_counts().idxmax())

    print(decisionTree._root._trueNode.df)
    print(decisionTree._root._falseNode.df)

def loadXy():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    return X,y

def main():
    loadXy()

if __name__ == '__main__':
    main()