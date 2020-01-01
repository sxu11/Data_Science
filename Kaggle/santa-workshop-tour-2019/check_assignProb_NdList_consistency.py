

import numpy as np

assignProbNp = np.loadtxt("/Users/songxu/PycharmProjects/Data_Science/Kaggle/santa-workshop-tour-2019/intermediate/assignProbNp.txt")
NdListNp = np.loadtxt("/Users/songxu/PycharmProjects/Data_Science/Kaggle/santa-workshop-tour-2019/intermediate/NdListNp.txt")
NdTensorNp = np.loadtxt("/Users/songxu/PycharmProjects/Data_Science/Kaggle/santa-workshop-tour-2019/intermediate/NdTensorNp.txt")
familyAssign = np.loadtxt("/Users/songxu/PycharmProjects/Data_Science/Kaggle/santa-workshop-tour-2019/intermediate/familyAssign.txt")

# print(assignProbNp.shape, NdListNp.shape)
#
import matplotlib.pyplot as plt

import utils
import torch

def getGiftCostsNp():
    row, col = 5000, 100
    giftCostsNp = np.zeros((row, col))
    familySizesNp = np.zeros((row, 1))

    familyDataNp = utils.readDataDf().values

    for i in range(row):
        familySizesNp[i] = familyDataNp[i, -1]
        for j in range(col):
            giftCostsNp[i, j] = utils.getCurGift(assignDay=j+1,
                                                 familyPreferencesList=familyDataNp[i, 1:-1].tolist(),
                                                 familySize=familySizesNp[i])
    return giftCostsNp

def checkProbVsInt(assignProbNp):
    row, col = assignProbNp.shape
    maxs = []
    for i in range(row):
        print(assignProbNp[i,:].sum())
        maxs.append(assignProbNp[i,:].max())
    plt.hist(maxs)
    plt.show()

# checkProbVsInt(assignProbNp)

def integerize(assignProbNp):
    assignIntNp = np.zeros(assignProbNp.shape)
    row, col = assignProbNp.shape
    for i in range(row):
        assignIntNp[i, assignProbNp[i,:].argmax()] = 1
    return assignIntNp

# assignIntNp = integerize(assignProbNp)

def getAssignFromDummy(assignProbNp):
    bestDays = []
    for i in range(assignProbNp.shape[0]):
        bestDays.append(assignProbNp[i,:].argmax()+1)
    return bestDays

familySizes = np.array(utils.getFamilySizes())
# print((assignProbNp[:,0]*familySizes).sum())

some = 0
for i in range(5000):
    if assignProbNp[i,:].argmax()==0:
        some += familySizes[i]
print("some:", some)

assign = getAssignFromDummy(assignProbNp)
# print(assign)
# assSum = 0
# for ass in assign:
#     if ass == 1:
#         assSum += familySizes[0]
# print("assSum", assSum)

NdList = utils.getNdList(assign)
print(NdList)
quit()

assignProbTc = torch.from_numpy(assignProbNp).type(torch.float32)
assignIntTc = torch.from_numpy(assignIntNp).type(torch.int16)

giftCostsNp = getGiftCostsNp()
giftCostsTc = torch.from_numpy(giftCostsNp).type(torch.float32)


# print(assignProbTc)
giftCostProbTotalTc = (giftCostsTc * assignProbTc).sum()
print(giftCostProbTotalTc)

giftCostIntTotalTc = (giftCostsTc * assignIntTc).sum()
print(giftCostIntTotalTc)

assignIntBackNp = assignIntTc.data.numpy()
assignIntBack = []
for i in range(assignIntBackNp.shape[0]):
    for j in range(assignIntBackNp.shape[1]):
        if assignIntBackNp[i,j] == 1:
            assignIntBack.append(j+1)
print(familyAssign.shape)



totalGiftPy = utils.getTotalGift(assignIntBack)
print(totalGiftPy)

totalGiftPy = utils.getTotalGift(familyAssign)
print(totalGiftPy)

def testMisc():
    print(utils.getNdList(familyAssign))

    epoch = 100000

    maxProbs = []
    for i in range(5000):
        argMax = assignProbNp[i, :].argmax()
        print(argMax, familyAssign[i])
        curMaxProb = assignProbNp[i, :].max()
        maxProbs.append(curMaxProb)
    plt.hist(maxProbs)
    plt.xlim([0,1])
    plt.xlabel("After %d epoch" % epoch)
    plt.ylabel("Hist of maxProbs")
    plt.show()

    quit()
    plt.plot(NdListNp)
    plt.plot(NdTensorNp)
    plt.show()