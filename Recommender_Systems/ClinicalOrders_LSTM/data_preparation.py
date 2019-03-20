
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/patient_items_50000_sample.csv')
print df.shape



'''
Remapping patient and item ids
'''
old_pat_ids = df['patient_id'].drop_duplicates().values.tolist()
num_unique_pats = len(old_pat_ids)
new_pats_ids = range(1,num_unique_pats+1)
pat_mapping = dict(zip(old_pat_ids, new_pats_ids))
df['patient_id_old'] = df['patient_id']
df['patient_id'] = df['patient_id'].apply(lambda x: pat_mapping[x])

old_item_ids = df['clinical_item_id'].drop_duplicates().values.tolist()
num_unique_items = len(old_item_ids)
new_item_ids = range(1,num_unique_items+1)
item_mapping = dict(zip(old_item_ids, new_item_ids))
df['clinical_item_id_old'] = df['clinical_item_id']
df['clinical_item_id'] = df['clinical_item_id'].apply(lambda x: item_mapping[x])

pats_train, pats_test = train_test_split(new_pats_ids)

df_train = df[df['patient_id'].isin(pats_train)]
df_test = df[df['patient_id'].isin(pats_test)]

df_train.to_csv('data/patient_items_50000_sample_train.csv', index=False)
df_test.to_csv('data/patient_items_50000_sample_test.csv', index=False)
