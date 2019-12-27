

import pandas as pd


def readDataDf():
    df = pd.read_csv("data/family_data.csv")
    return df

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
        9: 500 + 36*familySize,
        -1: 500 + 36*familySize + 398*familySize
    }
    return gifts[ind]

def getTotalPenalty(familyAssign): # length 100
    """
    Summing up the costs of Christmas Eve (d=1), to d=100

    :param NdList:
    :return:
    """
    NdList = getNdList(familyAssign)

    totPenalty = 0
    for d in range(1,101):
        i = d - 1 # On Christmas Even, i = 0, i + 1 = 1
        Nd = NdList[i]
        if d < 100:
            Nd1 = NdList[i+1]
        else: # boundary condition
            Nd1 = NdList[i]

        curPenalty = (Nd-125)/400. * Nd**(0.5+abs(Nd-Nd1)/50.)
        totPenalty += curPenalty

    return totPenalty

def getCurGift(assignDay, familyPreferencesList, familySize):
    if assignDay in familyPreferencesList:
        perferInd = familyPreferencesList.index(assignDay)
    else:
        perferInd = -1
    return getGift(perferInd, familySize)

def getTotalGift(familyAssign):

    NdList = getNdList(familyAssign)
    familyDataNp = readDataDf().values

    row, col = familyDataNp.shape
    totalGift = 0
    for i in range(row): # the i-th family
        familyPreferencesList = familyDataNp[i, :-1].tolist()
        familySize = familyDataNp[i, -1]

        curGift = getCurGift(familyAssign[i], familyPreferencesList, familySize)
        totalGift += curGift
    return totalGift

def getNdList(familyAssign):
    NdList = [0] * 100
    for i in range(len(familyAssign)):
        NdList[familyAssign[i]-1] += 1
    return NdList

def getTotalScore(familyAssign): # list of assignment


    totPenalty = getTotalPenalty(familyAssign)

    totGift = getTotalGift(familyAssign)

    return totPenalty + totGift

def readSampleAssign():
    df = pd.read_csv("data/sample_submission.csv")
    return df["assigned_day"].tolist()

if __name__ == '__main__':
    sampleAssign = readSampleAssign()
    print(getTotalScore(sampleAssign))