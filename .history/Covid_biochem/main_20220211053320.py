import pandas as pd
import os
dirname = os.path.dirname(__file__)

csv_file=os.join("pytorch_tabular-main/data/clinical_data.csv"
df = pd.read_csv(csv_file)
cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols=list(set(cols) - set(num_cols))
print(num_cols)
print(cat_cols)

