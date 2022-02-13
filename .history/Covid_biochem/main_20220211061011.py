import pandas as pd
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb

dirname = os.path.dirname(__file__)
csv_file = os.path.join(dirname, "pytorch_tabular-main/data/clinical_data.csv")
df = pd.read_csv(csv_file)
num_cols = list(df.select_dtypes("number").columns)
cat_cols = list(df.select_dtypes(exclude=["number"]).columns)
target_col = "last.status"
train, test = train_test_split(df, test_size=0.2)
