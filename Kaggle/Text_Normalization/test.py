
import pandas as pd
import operator
import numpy as np

print [set() for i in range(3)]
quit()

from collections import Counter
a = [2,1,2]
b = [1,2,2]
print(Counter(a)==Counter(b))
quit()

def get_year(year_str):
    res = ''
    if int(year_str[:2]) == '19':
        res += 'nineteen'
    elif int(year_str[:2]) == '20':
        res += 'two thousand'
    res += ' '

    if int(year_str[2:]) == '11':
        res += 'eleven'
    elif int(year_str[2:]) == '12':
        res += ''


import math
print math.floor(0.5)
quit()

a = ['a','b','c']
b = ['c','d','e']

if set(a)&set(b):
    print 'ha'
quit()

my_queue = []
my_queue.insert(0,'a')
print my_queue
my_queue.insert(0,'b')
print my_queue
my_queue.pop()
print my_queue
quit()

a = [0,1,2,None,40]
print sum(a)
quit()

import matplotlib.pyplot as plt
plt.hist(a)
plt.show()