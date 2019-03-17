

import pandas as pd

df = pd.read_csv("all/train_ver2.csv")

row, col = df.shape

a_dict = {}
for i in range(row):
    user_id = df.ix[i, 'ncodpers']
    if not a_dict.has_key(user_id):
        a_dict[user_id] = [i]
    else:
        a_dict[user_id].append(i)


a_keys_pd = pd.DataFrame(a_dict.keys())
a_vals_pd = pd.DataFrame(a_dict.values())
a_keys_pd.to_csv('a_keys_pd.csv')
a_vals_pd.to_csv('a_vals_pd.csv')