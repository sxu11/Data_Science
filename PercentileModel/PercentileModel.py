import utils_PercentileModel as utils
import os

class PercentileModel():
    def __init__(self, dataFolderPath=None, maxn=50000, thres=98, percentileMatrix=None):
        # Model.__init__(self)
        self.maxN = int(maxn)
        self.thres = int(thres)

        filterType = "0.4"

        self.featureDict = {"totalTargetCnt": "totalCnt",
                            "regressTargetCnt_%s" % filterType: "regressCnt",
                            "rolloutLabel": "rolloutLabel"}
        self.features = list(self.featureDict.keys())

        if dataFolderPath is not None:

            self.artifactsFilePath = os.path.join(dataFolderPath, "percentileMatrix.csv") #, "artifacts"
            self.dataFolderPath = dataFolderPath

            self.matrix = None
        else:
            self.matrix = percentileMatrix
        pass


    def fit(self, dataDf):
        df = dataDf[self.features].rename(columns=self.featureDict)

        labelToInterpolatedTimeSeries = utils.interpolateTimeSeries(df, N=self.maxN)  # I-by-(10000, 10000)
        totalCntToRegressCnts = utils.getRegressSortedCnts(labelToInterpolatedTimeSeries, self.dataFolderPath, N=self.maxN)
        percentileMatrix = utils.getPercentileMatrix(totalCntToRegressCnts, N=self.maxN)
        utils.savePercentileMatrix(percentileMatrix, self.artifactsFilePath)

        self.matrix = percentileMatrix

    def predict(self, dataDf): 
        df = dataDf[self.features].rename(columns=self.featureDict)
        df["thresReg"] = df["totalCnt"].apply(lambda x: utils.getReg(x, self.thres, self.matrix))
        return df["regressCnt"] >= df["thresReg"]


    def predictProbaRow(self, row):
        return utils.getPercentile(row["totalCnt"], row["regressCnt"], self.matrix) / 100.

    def predict_proba(self, dataDf):
        df = dataDf[self.features].rename(columns=self.featureDict)
        return df.apply(self.predictProbaRow, axis=1)