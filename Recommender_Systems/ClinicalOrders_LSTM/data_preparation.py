
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/patient_items_50000_sample.csv')
print df.shape

all_pats = df['patient_id'].drop_duplicates().values

pats_train, pats_test = train_test_split(all_pats)

df_train = df[df['patient_id'].isin(pats_train)]
df_test = df[df['patient_id'].isin(pats_test)]

df_train.to_csv('data/patient_items_50000_sample_train.csv', index=False)
df_test.to_csv('data/patient_items_50000_sample_test.csv', index=False)
