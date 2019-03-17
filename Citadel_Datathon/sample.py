
import pandas as pd
pd.set_option('display.width', 300)

data_folder = 'raw_data/'
# chemicals_pd = pd.read_csv(data_folder + '/' + 'chemicals.csv')
earnings_pd = pd.read_csv(data_folder + '/' + 'earnings.csv')
print earnings_pd
# print pd.get_dummies(chemicals_pd['contaminant_level'])

# all_contaminant_levels = chemicals_pd['contaminant_level'].values.tolist()
# print set(all_contaminant_levels)
# An MCL is the highest level of a contaminant that is allowed in drinking water.
# print all_contaminant_levels.count('Greater than MCL')
# print all_contaminant_levels.count('Less than or equal MCL')
# print all_contaminant_levels.count('Non Detect')