

def test():
    df = pd.DataFrame({'Color': ['Green', 'Yellow', 'Red', 'Red', 'Yellow'],
                       'Diam': [3, 3, 1, 1, 3],
                       'Label': ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon']})

    decisionTree = DecisionTree(df)
    decisionTree.build()
    # print(df['Color'].value_counts().idxmax())

    print(decisionTree._root._trueNode.df)
    print(decisionTree._root._falseNode.df)