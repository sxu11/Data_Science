

import pandas as pd

ha = [0]*3
ha[0] = 1
print ha

import matplotlib.pyplot as plt

ha = list('abca')
print pd.Series(ha)
print pd.get_dummies(ha)