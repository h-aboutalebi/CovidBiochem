import pandas as pd
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from utility.utils import seed_everything
from data_preprocess.csv_handler import CSVHandler

seed = 111
seed_everything(seed)

dirname = os.path.dirname(__file__)
csv_file = os.path.join(dirname, "pytorch_tabular-main/data/clinical_data.csv")
target_col = "therapeuticexnoxBoolean" #If your col name  has special character other than "_", remove them in the name
csv_handle = CSVHandler(
    csv_file,
    useless_cols_list=["to_patient_id"],
    target_col=target_col)
train, test = train_test_split(csv_handle.df, test_size=0.2,random_state=42)

clf = lgb.LGBMClassifier(random_state=42)
clf.fit(
    train.drop(
        columns=target_col),
    train[target_col],
    categorical_feature=csv_handle.cat_cols)
test_pred = clf.predict(test.drop(columns=target_col))


def print_metrics(y_true, y_pred, tag):
    if isinstance(
            y_true,
            pd.DataFrame) or isinstance(
            y_true,
            pd.Series):
        y_true = y_true.values
    if isinstance(
            y_pred,
            pd.DataFrame) or isinstance(
            y_pred,
            pd.Series):
        y_pred = y_pred.values
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    val_acc = accuracy_score(y_true, y_pred)
    # val_f1 = f1_score(y_true, y_pred)
    print(f"{tag} Acc: {val_acc}")


print_metrics(test[target_col], test_pred, "Holdout")
