

from medinfo.db import DBUtil
import pandas as pd

columns = ('patient_id', 'clinical_item_id', 'item_date')

query_str = "select %s, %s, %s " \
            "from patient_item " \
            "where item_date >= '2016-01-01' " \
            "and item_date < '2017-01-01'" \
            "limit 5000"


db_cursor = DBUtil.connection().cursor()
db_cursor.execute(query_str % columns)

all_rows = db_cursor.fetchall()

df = pd.DataFrame(all_rows, columns=columns)
print df
df.to_csv('patient_items_5000_sample.csv', index=False)
quit()

cols = ()

for one_row in all_rows:
    print one_row