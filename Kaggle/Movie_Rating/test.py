
import pandas as pd
import os


data_folder = 'predict-movie-ratings-v22'
result_df = pd.read_csv(os.path.join('results', 'submission.csv'))

no_IDs = result_df[result_df['rating'].isnull()]['ID'].values.tolist()

test_df = pd.read_csv(os.path.join(data_folder, 'test.csv'))
no_movies = test_df[test_df['ID'].isin(no_IDs)]['movie']

train_df = pd.read_csv(os.path.join(data_folder, 'train.csv'))
print train_df[train_df['movie'].isin(no_movies)]
quit()





#

result_df['rating'] = result_df['rating'].fillna(3.51241660044)
result_df.to_csv(os.path.join('results', 'submission_baseline.csv'), float_format='%.2f', index=False)
