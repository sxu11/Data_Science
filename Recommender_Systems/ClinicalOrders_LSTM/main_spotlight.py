
'''
https://towardsdatascience.com/introduction-to-recommender-system-part-2-adoption-of-neural-network-831972c4cbf7
https://maciejkula.github.io/spotlight/sequence/implicit.html
'''

import pandas as pd
import numpy as np

df = pd.read_csv('data/patient_items_5000_sample.csv')
print 'df.shape:', df.shape

# pat_ids = df['patient_id'].drop_duplicates().values.tolist()
# grouped_pat_items = df.groupby('patient_id')['clinical_item_id'].apply(list).reset_index()

from spotlight.interactions import Interactions
from spotlight.sequence.implicit import ImplicitSequenceModel

# users = grouped_pat_items['patient_id'].values
# list_of_nplists = grouped_pat_items['clinical_item_id'].values
# items = [list(x) for x in list_of_nplists]

users = df['patient_id'].values
items = df['clinical_item_id'].values
times = df['item_date'].values

print df['clinical_item_id'].shape
print df['clinical_item_id'].drop_duplicates().shape
#TODO: re-number all pat_ids and item_ids!

implicit_interactions = Interactions(users, items, timestamps=times)
print implicit_interactions.num_items
quit()


sequential_interaction = implicit_interactions.to_sequence()
implicit_sequence_model = ImplicitSequenceModel()

implicit_sequence_model.fit(sequential_interaction)
print implicit_sequence_model._num_items

print len(implicit_sequence_model.predict(5))