
import numpy as np

class DecisionTree:
    def __init__(self, random_state=0):
        # self._root = Node(df.values, df.dtypes)
        self._random_state = random_state

    def fit(self, X_train, y_train):
        self._root = Node(X_train, y_train) # TODO: for now, only handle numeric!
        self._root.grow(random_state = self._random_state)

    def predict(self, X_test):
        ress = []
        for i in range(X_test.shape[0]):
            res = self._root.predict(X_test[i, :])
            ress.append(res)
        return ress


class Node:
    """
    Node of the tree. Contains:

    """
    def __init__(self, X_train, y_train):
        self._trueNode = None
        self._falseNode = None
        # self.matrix = matrix
        # self.colTypes = colTypes
        self._X_train = X_train
        self._y_train = y_train

        self._prob = y_train.sum()/float(len(y_train))

        self.rowNum, self.colNum = X_train.shape

        self._splitFeat = None
        self._splitVal = None

        self.MIN_GAIN_TO_SPLIT = 0 # TODO: scale?

    def predict(self, testVec):
        if self._splitFeat is None:
            return self._prob

        if testVec[self._splitFeat] < self._splitVal:
            return self._trueNode.predict(testVec)
        else:
            return self._falseNode.predict(testVec)

    def grow(self, random_state=0): # TODO
        # maxGainAllCols = 0
        # maxGainArgColAllCols = None
        # maxGainArgSplitAllCols = None
        maxGainDict = {"gain":0}

        for col in range(self.colNum):
            splits = self.getSplits(col)

            # maxGainCurCol, maxGainArgSplitCurCol = self.getMaxGainAndSplit(col, splits)
            curColMaxGainDict = self.getMaxGainAndSplit(col, splits)
            # maxGainCurCol = max([self.getGainFromSplit(split) for split in splits])
            if curColMaxGainDict["gain"] > maxGainDict["gain"]:
                # maxGainAllCols = max(maxGainCurCol, maxGainAllCols)
                # maxGainAllCols, maxGainArgSplitAllCols = maxGainCurCol, maxGainArgSplitCurCol
                # maxGainArgColAllCols = col
                maxGainDict = curColMaxGainDict

        if maxGainDict["gain"] > self.MIN_GAIN_TO_SPLIT:
            self._growCol = maxGainArgColAllCols # TODO
            self._growSplit = maxGainArgSplitAllCols

            self._trueNode = Node(maxGainDict["xtrainTrue"], maxGainDict["ytrainTrue"])
            self._trueNode.grow()
            self._falseNode = Node(maxGainDict["xtrainFalse"], maxGainDict["ytrainFalse"])
            self._falseNode.grow()

    def calcGiniImpurity(self, aVec):
        return 1 - np.square(aVec).sum()

    def getSplitDictBySplitVal(self, splitFeat, splitVal):
        # TODO: numeric for now
        # Question: val < split ?
        fullVec = self._X_train[:, splitFeat]

        splitDict = {}
        trueMasks = fullVec < splitVal
        falseMasks = fullVec >= splitVal

        splitDict["ytrainTrue"] = self._y_train[trueMasks] # TODO: Calc it here, or pass the mask?
        splitDict["xtrainTrue"] = self._X_train[trueMasks]

        splitDict["ytrainFalse"] = self._y_train[falseMasks]
        splitDict["xtrainFalse"] = self._X_train[falseMasks]

        return splitDict

    def getGainFromSplit(self, col, split):
        # fullVec = self._X_train[:, col]
        splitDict = self.getSplitDictBySplitVal(col, split)

        fullVec = self._y_train

        trueY, falseY = splitDict["ytrainTrue"], splitDict["ytrainFalse"]

        curGini = self.calcGiniImpurity(fullVec)

        trueGini = self.calcGiniImpurity(trueY)
        trueWt = len(trueY)/float(len(fullVec))

        falseGini = self.calcGiniImpurity(falseY)
        falseWt = len(falseY)/float(len(fullVec))

        gain = curGini - trueWt * trueGini - falseWt * falseGini

        splitDict["gain"] = gain

        return splitDict

    def getMaxGainAndSplit(self, col, splits):
        # maxGain = 0
        # maxGainSplit = None
        maxGainDict = {"gain":0}

        for split in splits:
            curSplitDict = self.getGainFromSplit(col, split)

            if curSplitDict["gain"] > maxGainDict["gain"]:
                # maxGain = curGain
                # maxGainSplit = split
                maxGainDict = curSplitDict

        return maxGainDict #maxGain, maxGainSplit

    def getSplits(self, feature):
        if True: #self.df[feature].dtype == int:
            vector = np.sort(list(set(self._X_train[:, feature])))
            return ((vector[1:] + vector[:-1])/2.).tolist()

        elif self.df[feature].dtype == object:
            return self.df[feature].unique().tolist()

        else:
            raise Exception("%s not implemented for getSplits()!" % self.df[feature].dtype)


    # def getBranches(self, feature, split):
    #     # if self.df[feature].dtype == int:
    #     #     pass
    #     # elif self.df[feature].dtype == object:
    #     #     pass
    #     trueMasks = self.df[feature].apply(lambda x: ask(split, x))
    #     trueBranch = self.df[trueMasks]
    #     falseBranch = self.df[~trueMasks]
    #     return trueBranch, falseBranch


