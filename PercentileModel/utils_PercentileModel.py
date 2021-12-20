import matplotlib.pyplot as plt
import numpy as np
import os

import bisect
def getPercentile(tot, reg, modelMatrix):
    if tot >= modelMatrix.shape[0]: # slient beyond capability
        return 0
    curPercentiles = modelMatrix[tot, :]
    curPercentile = bisect.bisect_left(curPercentiles, reg)
    return curPercentile

def getInterpolatedCnt(ns, ms, n):
    import bisect
    # ns = cachedTotalNodes[rolloutLabel]
    # ms = cachedRegressNodes[rolloutLabel]
    ind = bisect.bisect_left(ns, n)
    if ind == len(ns):
        m = ms[-1]
        n = ns[-1]
    else:
        if ns[ind] == n:
            m = ms[ind]
        else:
            if ind == 0:
                nLeft = 0
                mLeft = 0
            else:
                nLeft = ns[ind-1]
                mLeft = ms[ind - 1]
            nRight = ns[ind]
            mRight = ms[ind]
            mInterpolated = (n-nLeft)/(nRight-nLeft) * (mRight-mLeft) + mLeft
            m = mInterpolated
    if n == 0:
        return 0
    else:
        return m #m/n

def interpolateTimeSeries(dataDf, N=20000):
    """
    No longer extrapolate naively
    key: label
    value:
    totalCnts: [0,1,2,...,min(N, maxTotalCnt)]
    regressCnts: [...]
    """
    labels = dataDf["rolloutLabel"].unique()
    labelToInterpolatedTimeSeries = dict()
    for label in labels:
        labelToInterpolatedTimeSeries[label] = {"totalCnts":[], "regressCnts":[]}

        curDf = dataDf[dataDf["rolloutLabel"]==label]

        totalCnts = curDf["totalCnt"].values
        regressCnts = curDf["regressCnt"].values

        if len(totalCnts) == 0:  # records are empty
            continue
        maxTotalCnt = totalCnts[-1]

        for totalCnt in range(min(N, maxTotalCnt)+1):
            rInterpolated = getInterpolatedCnt(totalCnts, regressCnts, totalCnt)

            labelToInterpolatedTimeSeries[label]["totalCnts"].append(totalCnt)
            labelToInterpolatedTimeSeries[label]["regressCnts"].append(rInterpolated)

    return labelToInterpolatedTimeSeries


def getRegressSortedCnts(labelToInterpolatedTimeSeries, dataFolderPath, N = 20000):
    """
    input:
    key: label
    val: totalCnts, regCnts

    output:
    key: totalCnt
    val: sorted regCnts

    byproduct:
    interpolation quality report
    """

    totalCntToRegressCnts = [[] for _ in range((N+1))] # N by I

    for label, labelVals in labelToInterpolatedTimeSeries.items():
        totalCnts = labelVals["totalCnts"]
        regressCnts = labelVals["regressCnts"]
        for j in range(len(totalCnts)):
            n = totalCnts[j]
            r = regressCnts[j]
            totalCntToRegressCnts[n].append(r)
    minNumRegress = 99999
    minTotalCntAtMinRegress = -1
    numRegresses = []
    ns = list(range(N+1))
    for n in ns:
        totalCntToRegressCnts[n] = sorted(totalCntToRegressCnts[n])
        regNum = len(totalCntToRegressCnts[n])
        numRegresses.append(regNum)

        if regNum < minNumRegress:
            minNumRegress = regNum
            minTotalCntAtMinRegress = n

    print("minNumRegress %d at minTotalCntAtMinRegress %d" % (minNumRegress, minTotalCntAtMinRegress))
    plt.plot(ns, numRegresses)
    plt.xlabel("totalCnt")
    plt.ylabel("numRegresses")
    # plt.savefig(os.path.join(dataFolderPath, "interpolationQuality.png"))
    plt.show()

    return totalCntToRegressCnts


def getPercentileMatrix(totalCntToRegressCnts, N = 20000):
    P = 100


    percentileMatrix = np.zeros((N+1, P+1)) #[[] for _ in range((N+1))]
    """ r[n, p] is a specific number r """
    for n in range(N+1):
        regressCnts = totalCntToRegressCnts[n]
        regressNum = len(regressCnts)

        thress = [0]
        regrss = [0]
        for regressInd in range(1, regressNum+1):
            thresIn = int(float(regressInd)/regressNum * 100)
            thress.append(thresIn)
            regrss.append(regressCnts[regressInd-1])
        # print(thress)
        # print(regrss)

        for j in range(len(thress)-1):
            percentileMatrix[n][thress[j]:thress[j+1]] = regrss[j+1]
        percentileMatrix[n][-1] = percentileMatrix[n][-2] + 0.001 # last one
    return percentileMatrix

def savePercentileMatrix(percentileMatrix, filePath):
    np.savetxt(filePath, percentileMatrix, delimiter=",")


def getReg(tot, percentile, modelMatrix):
    if tot >= modelMatrix.shape[0]:
        return np.NaN
    return modelMatrix[tot, percentile]