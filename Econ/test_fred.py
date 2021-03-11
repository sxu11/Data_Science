
"""
https://pypi.org/project/fredapi/
"""

import pandas as pd
from fredapi import Fred
fred = Fred(api_key="39046de0f8665857c262677588e22e46")

s = fred.get_series('SP500', observation_start='2014-09-02', observation_end='2014-09-05')
print(s.tail())

g = fred.get_series('GDP', observation_start='2014-03-02', observation_end='2014-09-05')
print(g.tail())