
import pandas as pd
import numpy as np

from spotlight.interactions import Interactions
from spotlight.sequence.implicit import ImplicitSequenceModel

df_train = pd.read_csv('data/patient_items_50000_sample_train.csv')
print 'df_train.shape:', df_train.shape

pat_ids_train = df_train['patient_id'].drop_duplicates().values.tolist()
print 'number of train patients: ', len(pat_ids_train)


'''
Remapping patient and item ids
'''
old_pat_ids = df_train['patient_id'].drop_duplicates().values.tolist()
num_unique_pats = len(old_pat_ids)
pat_mapping = dict(zip(old_pat_ids, range(1,num_unique_pats+1)))
df_train['patient_id'] = df_train['patient_id'].apply(lambda x: pat_mapping[x])

old_item_ids = df_train['clinical_item_id'].drop_duplicates().values.tolist()
num_unique_items = len(old_item_ids)
item_mapping = dict(zip(old_item_ids, range(1,num_unique_items+1)))
df_train['clinical_item_id'] = df_train['clinical_item_id'].apply(lambda x: item_mapping[x])


'''
I'm really amazed by this pandas trick, tho it is not useful:

grouped_pat_items = df.groupby('patient_id')['clinical_item_id'].apply(list).reset_index()
'''


'''
Prepare training data
'''
users_train = df_train['patient_id'].values
items_train = df_train['clinical_item_id'].values
times_train = df_train['item_date'].values


'''
Feed training data into model for training
'''
implicit_interactions = Interactions(users_train,
                                     items_train,
                                     timestamps=times_train)
sequential_interaction = implicit_interactions.to_sequence()
implicit_sequence_model = ImplicitSequenceModel()
implicit_sequence_model.fit(sequential_interaction)


'''
Testing
'''
df_test = pd.read_csv('data/patient_items_50000_sample_test.csv')
print 'df_test.shape:', df_test.shape
df_test = df_test.sort_values(['patient_id', 'item_date'])

pat_ids_test = df_test['patient_id'].drop_duplicates().values.tolist()
print 'number of train patients: ', len(pat_ids_test)

def get_top_k(scores, k=10):
    return [x+1 for x in np.argsort(scores)[-k:]]

def precision_at_k(actuals, predicts):
    num_relevant = 0
    for predict in predicts:
        if predict in actuals:
            num_relevant += 1
    return float(num_relevant)/len(predicts)

for pat_id in pat_ids_test:
    cur_rows = df_test[df_test['patient_id']==pat_id]
    cur_orders = cur_rows['clinical_item_id'].values.tolist()

    '''
    Needs optimization!
    '''
    query_set = [cur_orders[0]]
    query_set = [item_mapping.get(x,0) for x in query_set]

    valid_set = cur_orders[1:]
    valid_set = [item_mapping.get(x,0) for x in valid_set]

    all_recommends = implicit_sequence_model.predict(query_set)
    top_k_recommends = get_top_k(all_recommends, k=10)

    print 'precision_at_k:', precision_at_k(actuals=valid_set, predicts=top_k_recommends)
    # print len()
