


import zipfile

with zipfile.ZipFile('train.zip', 'r') as zin:
    zin.extractall('train/')

with zipfile.ZipFile('test.zip', 'r') as zin:
    zin.extractall('test/')