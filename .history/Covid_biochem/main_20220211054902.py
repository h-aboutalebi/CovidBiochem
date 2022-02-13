import pandas as pd
import os
dirname = os.path.dirname(__file__)
from sklearn.model_selection import train_test_split

csv_file=os.path.join(dirname,"pytorch_tabular-main/data/clinical_data.csv")
df = pd.read_csv(csv_file)
num_cols = list(df.select_dtypes("number").columns)
cat_cols=list(df.select_dtypes(exclude=["number"]).columns)
print(num_cols)
print(cat_cols)
print("yes")

