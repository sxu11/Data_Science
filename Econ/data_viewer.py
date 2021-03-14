
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


sp_data = pd.read_csv("manual_data/SP500.csv")
"""2011-03-14 to 2021-03-12"""

cpi_data = pd.read_csv("manual_data/CPALTT01USM657N.csv")
m1_data = pd.read_csv("manual_data/WM1NS.csv")
baa_data = pd.read_csv("manual_data/BAA10Y.csv")
gdp_data = pd.read_csv("manual_data/GDP.csv")

# print(sp_data.shape[0], cpi_data.shape[0], m1_data.shape[0]*7, baa_data.shape[0]*1, gdp_data.shape[0]*90)

# print(sp_data.head(3))
# print(cpi_data.head(3))
# print(m1_data.head(3))
# print(baa_data.head(3))
# print(gdp_data.head(3))

idx = pd.date_range("2016-03-14", "2021-01-01")

filenames = ["SP500","CPALTT01USM657N", "WM1NS", "BAA10Y", "GDP"]

for i,df in enumerate([sp_data, cpi_data, m1_data, baa_data, gdp_data]):
    df["DATE"] = pd.to_datetime(df["DATE"], format='%Y-%m-%d')
    df[filenames[i]] = df[filenames[i]].apply(lambda x: np.NaN if x == "." else float(x))



firstVals = []
"""
what is the latest available value before "2016-03-14"?
"""
# TODO: should not probably do this for missing sp_data (y)
for df in [cpi_data, m1_data, baa_data, gdp_data]:
    prev_val = None
    for i in range(df.shape[0]):
        dateVal = df.iloc[i].values
        if dateVal[0] > idx[0]:
            firstVals.append(float(prev_val))
            break
        prev_val = dateVal[1]
# print(firstVals)

sp_series = sp_data.set_index("DATE").squeeze().reindex(idx)
cpi_series = cpi_data.set_index("DATE").squeeze().reindex(idx)
m1_series = m1_data.set_index("DATE").squeeze().reindex(idx)
baa_series = baa_data.set_index("DATE").squeeze().reindex(idx)
gdp_series = gdp_data.set_index("DATE").squeeze().reindex(idx)



print("After squeeze:")

print(sp_series.head(5))
print(cpi_series.head(5))
print(m1_series.head(5))
print(baa_series.head(5))
print(gdp_series.head(5))
# quit()

def fillInHistoryValue(ds, firstVal):
    curVal = firstVal
    for i in range(ds.shape[0]):
        if ds.iloc[i] != ds.iloc[i]:
            ds.iloc[i] = curVal
        else:
            curVal = ds.iloc[i]
    return ds

for i, ds in enumerate([cpi_series, m1_series, baa_series, gdp_series]):
    # df["DATE"] = pd.to_datetime(df["DATE"], format='%Y-%m-%d')
    fillInHistoryValue(ds, firstVals[i])

print("After fillInHistoryValue:")

print(sp_series.head(5))
print(cpi_series.head(5))
print(m1_series.head(5))
print(baa_series.head(5))
print(gdp_series.head(5))
"""data seems only 7 year"""
# plt.plot(sp_data["0"])
# plt.ylim([0,3500])
# plt.show()



intermediate_folder = "intermediate_data"
for i, ds in enumerate([sp_series, cpi_series, m1_series, baa_series, gdp_series]):
    ds.to_csv(os.path.join(intermediate_folder, filenames[i]+".csv"), index=False)

