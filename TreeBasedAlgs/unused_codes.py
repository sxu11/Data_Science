


# def split(self):
#     """
#     Steps:
#     Find the best question to ask, based on current df
#         (1) best feature
#         (2) best split
#     Then under a question, split the node into two branches
#     """
#     features = self.df.columns[:-1].values.tolist()
#     target = self.df.columns[-1]
#
#     minSquaredError = 100000 # TODO
#     bestSplitDfs = []
#     for feature in features:
#         splits = self.getSplits(feature)
#         for split in splits:
#             trueDf, falseDf = self.getBranches(feature, split)
#
#             truePredict = trueDf[target].value_counts().idxmax()
#             falsePredict = falseDf[target].value_counts().idxmax()
#             trueSquaredError = calcSquaredError(trueDf[target].values, truePredict)
#             falseSquaredError = calcSquaredError(falseDf[target].values, falsePredict)
#             curSquaredError = trueSquaredError + falseSquaredError
#             if curSquaredError < minSquaredError:
#                 minSquaredError = curSquaredError
#                 bestSplitDfs = [trueDf, falseDf]
#
#     self._trueNode = Node(bestSplitDfs[0])
#     self._falseNode = Node(bestSplitDfs[1])