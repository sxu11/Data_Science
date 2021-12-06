
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# folderPath = r"/Users/songxu/Downloads/datathon"
#
# fpFolderPath = os.path.join(folderPath, "fp")
# for fpFileName in os.listdir(fpFolderPath):
#     fpFilePath = os.path.join(fpFolderPath, fpFileName)
#     dfFp = pd.read_csv(fpFilePath)
#     print(dfFp.head(3))



azdFilePath="/Users/songxu/Downloads/Azd_Sept_data.csv"
azdDf = pd.read_csv(azdFilePath)

hsAll = azdDf[azdDf["Source"]=="AzDeployHealth"]
hsAllStopLabels = dict(zip(hsAll["Query"], hsAll["TIMESTAMP"]))

hsInterest = hsAll[(hsAll["TIMESTAMP"]>="2021-09-01") & (hsAll["TIMESTAMP"]<"2021-10-01")]

hsTN = azdDf[(azdDf["Source"].isin(["AzureWatson","AzureComputeInsights"]))
              & ~(azdDf["Query"].isin(hsAllStopLabels))
              & (azdDf["IsFalsePositive"]==0)]
hsTNInterest = hsTN[(hsTN["TIMESTAMP"]>="2021-09-01") & (hsTN["TIMESTAMP"]<"2021-10-01")]
# print(hsTNInterest)



azdDf["hsStopTime"] = azdDf["Query"].apply(lambda x: hsAllStopLabels.get(x, "9999-99-99"))
hsSlower=azdDf[(azdDf["Query"].isin(hsAllStopLabels))  #
      & (azdDf["TIMESTAMP"]<azdDf["hsStopTime"]) #faster than hs
      & (azdDf["TIMESTAMP"]>="2021-09-01") & (azdDf["TIMESTAMP"]<"2021-10-01")
      & (azdDf["IsFalsePositive"]==0)
]
# print(hsSlower[["Query","Source","TIMESTAMP","hsStopTime"]])

hsFaster=azdDf[(azdDf["Query"].isin(hsAllStopLabels))  #
      & (azdDf["TIMESTAMP"]>azdDf["hsStopTime"]) #faster than hs
      & (azdDf["TIMESTAMP"]>="2021-09-01") & (azdDf["TIMESTAMP"]<"2021-10-01")
      & (azdDf["IsFalsePositive"]==0)
]
print(hsFaster[["Query","Source","TIMESTAMP","hsStopTime"]])

hsFasterByModel = hsFaster[hsFaster["Source"]=="AzureComputeInsights"]
diff=pd.to_datetime(hsFasterByModel["TIMESTAMP"])-\
     pd.to_datetime(hsFasterByModel["hsStopTime"])
# print()
# print(diff.mean())