import pandas as pd

csv_file="pytorch_tabular-main/data/clinical_data.csv"
df = pd.read_csv(csv_file)
cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols=list(set(cols) - set(num_cols))
prin()

