
"""
https://pypi.org/project/fredapi/

https://seekingalpha.com/article/4395421-modeling-fair-value-of-s-and-p-500

https://mortada.net/python-api-for-fred.html

https://fred.stlouisfed.org/searchresults/?st=CPI
"""

import pandas as pd
from fredapi import Fred
fred = Fred(api_key="39046de0f8665857c262677588e22e46")

observation_start='2020-12-14'
observation_end='2021-03-14'
datestr = observation_start + "__" + observation_end

"""
daily
"""
filenames = ["SP500","CPALTT01USM657N", "WM1NS", "BAA10Y", "GDP"]
for index in filenames:
    s = fred.get_series(index, observation_start=observation_start, observation_end=observation_end)
    print(index)
    print(s.tail())
    print()
    s.to_csv("data/%s_%s.csv" % (index, datestr), index=False)

quit()

