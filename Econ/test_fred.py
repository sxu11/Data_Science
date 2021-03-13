
"""
https://pypi.org/project/fredapi/

https://seekingalpha.com/article/4395421-modeling-fair-value-of-s-and-p-500

https://mortada.net/python-api-for-fred.html

https://fred.stlouisfed.org/searchresults/?st=CPI
"""

import pandas as pd
from fredapi import Fred
fred = Fred(api_key="39046de0f8665857c262677588e22e46")

observation_start='1960-01-01'
observation_end='2021-01-01'

"""
daily
"""
s = fred.get_series('SP500', observation_start=observation_start, observation_end=observation_end)
print(s.tail())
s.to_csv("data/sp500_19600101_2021_0101.csv", index=False)


"""
3-month
"""
g = fred.get_series('GDP', observation_start=observation_start, observation_end=observation_end)
print(g.tail())
g.to_csv("data/gdp_19600101_2021_0101.csv", index=False)

"""
weekly
"""
g = fred.get_series('M1', observation_start=observation_start, observation_end=observation_end)
print(g.tail())
g.to_csv("data/m1_19600101_2021_0101.csv", index=False)

"""
daily
"""
g = fred.get_series('BAA10Y', observation_start=observation_start, observation_end=observation_end)
print(g.tail())
g.to_csv("data/baa10y_19600101_2021_0101.csv", index=False)

"""
monthly
"""
g = fred.get_series('CPALTT01USM657N', observation_start=observation_start, observation_end=observation_end)
print(g.tail())
g.to_csv("data/cpi_19600101_2021_0101.csv", index=False)