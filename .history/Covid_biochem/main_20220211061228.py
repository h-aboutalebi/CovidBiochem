import pandas as pd
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score

dirname = os.path.dirname(__file__)
csv_file = os.path.join(dirname, "pytorch_tabular-main/data/clinical_data.csv")
df = pd.read_csv(csv_file)
num_cols = list(df.select_dtypes("number").columns)
cat_cols = list(df.select_dtypes(exclude=["number"]).columns)
target_col = "last.status"
train, test = train_test_split(df, test_size=0.2)
clf = lgb.LGBMClassifier(random_state=42)
clf.fit(train.drop(columns='target'), train['target'], categorical_feature=cat_cols)
test_pred = clf.predict(test.drop(columns='target'))


def print_metrics(y_true, y_pred, tag):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim>1:
        y_true=y_true.ravel()
    if y_pred.ndim>1:
        y_pred=y_pred.ravel()
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred)
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")
    
print_metrics(test['target'], test_pred, "Holdout")
