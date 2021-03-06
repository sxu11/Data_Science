

import pandas as pd
import numpy as np

def readDataDf():
    df = pd.read_csv("/Users/songxu/PycharmProjects/Data_Science/Kaggle/santa-workshop-tour-2019/data/family_data.csv")
    return df

familyDataNp = readDataDf().values

def getGift(ind, familySize):
    gifts = {
        0: 0,
        1: 50,
        2: 50 + 9*familySize,
        3: 100 + 9*familySize,
        4: 200 + 9*familySize,
        5: 200 + 18*familySize,
        6: 300 + 18*familySize,
        7: 300 + 36*familySize,
        8: 400 + 36*familySize,
        9: 500 + 36*familySize + 199*familySize,
        -1: 500 + 36*familySize + 398*familySize
    }
    return gifts[ind]

def getTotalPenalty(familyAssign, verbose=False): # length 100
    """
    Summing up the costs of Christmas Eve (d=1), to d=100

    :param NdList:
    :return:
    """
    NdList = getNdList(familyAssign)
    if False: #verbose:
        # print("NdList[:10] inside getTotalPenalty():", NdList[:10])
        print(NdList)
        # np.savetxt("intermediate/familyAssign.txt", familyAssign)

    totPenalty = 0
    for d in range(1,101):
        i = d - 1 # On Christmas Eve, i = 0, i + 1 = 1
        Nd = NdList[i]
        if d < 100:
            Nd1 = NdList[i+1]
        else: # boundary condition
            Nd1 = NdList[i]

        curPenalty = (Nd-125.)/400. * Nd**(0.5+abs(Nd-Nd1)/50.)
        # print('exponent:', 0.5+abs(Nd-Nd1)/50.)
        totPenalty += curPenalty

    return totPenalty

def getCurGift(assignDay, familyPreferencesList, familySize):
    # print(assignDay)
    # print(familyPreferencesList)
    if assignDay in familyPreferencesList:
        perferInd = familyPreferencesList.index(assignDay)
    else:
        perferInd = -1

    # print("perferInd:", perferInd)
    return getGift(perferInd, familySize)

def getFamilySizes():


    row, col = familyDataNp.shape
    familySizes = []
    for i in range(row):
        familySize = familyDataNp[i, -1]
        familySizes.append(familySize)
    return familySizes

def getTotalGift(familyAssign, verbose=False):

    # NdList = getNdList(familyAssign)
    familyDataNp = readDataDf().values

    row, col = familyDataNp.shape
    totalGift = 0
    for i in range(row): # the i-th family
        familyPreferencesList = familyDataNp[i, 1:-1].tolist()
        familySize = familyDataNp[i, -1]

        # print("the %d-th family: " % i)
        curGift = getCurGift(familyAssign[i], familyPreferencesList, familySize)
        totalGift += curGift
    return totalGift

def getNdList(familyAssign): # familyAssigment: from 1 to 100
    NdList = [0] * 100
    familySizes = getFamilySizes()
    # print(len(familyAssign))
    for i in range(len(familyAssign)): # i ranges from 0 to 4999
        # if (familyAssign[i] != int(round(familyAssign[i]))):
        #     print("wtf")
        day = int(round(familyAssign[i]))
        NdList[day-1] += familySizes[i]
    return NdList

def getTotalScore(familyAssign, verbose=False): # list of assignment


    totClean = getTotalPenalty(familyAssign, verbose)

    totGift = getTotalGift(familyAssign, verbose)

    if verbose:
        print("totGift, totClean:", totGift, totClean)
    return totClean + totGift

def readSampleAssign():
    df = pd.read_csv("data/sample_submission.csv")
    return df["assigned_day"].tolist()

if __name__ == '__main__':
    # sampleAssign = readSampleAssign()
    # print(getTotalScore(sampleAssign))
    print(getFamilySizes())
    # print(readDataDf())