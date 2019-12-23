

class DecisionTree:
    def __init__(self, df):
        self._root = Node(df.values, df.dtypes)

    def buildDecisionTree(self):
        self._root.grow()


class Node:
    """
    Node of the tree. Contains:

    """
    def __init__(self, matrix, colTypes):
        self._trueNode = None
        self._falseNode = None
        self.matrix = matrix
        self.colTypes = colTypes
        self.rowNum, self.colNum = matrix.shape

        self.MIN_GAIN_TO_SPLIT = 10 # TODO: scale?

    def grow(self):
        maxGainAllCols = 0
        for col in range(self.colNum):
            splits = self.getSplits(col)
            maxGainCurCol = max([self.getGainFromSplit(split) for split in splits])
            maxGainAllCols = max(maxGainCurCol, maxGainAllCols)

        if maxGainAllCols > self.MIN_GAIN_TO_SPLIT:
            self._trueNode.grow()
            self._falseNode.grow()


    def getGainFromSplit(self, split):
        gain = 0
        return gain



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


