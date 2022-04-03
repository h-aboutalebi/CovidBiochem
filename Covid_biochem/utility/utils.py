import random
import numpy as np
import os
import re
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def seed_everything(seed=42):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pytorch_accuracy(list_labels, list_outputs):
    correct = 0
    total = 0
    for i in len(list_labels):
        correct += (list_outputs[i] == list_labels[i]).float().sum()
        total += len(list_labels)
    return correct / total


def print_metrics(y_true, y_pred):
    try:
        y_true = list(y_true)
        y_pred = list(y_pred)
    except Exception as e:
        print("error occured: {}".format(e))
    conf_matrix = confusion_matrix(y_true, y_pred)
    precisions = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    results = {
        "confusion matrix": conf_matrix,
        "Accuracy": accuracy,
        "precision": precisions,
        "recall": recall,
        "F1 score": f1
    }
    for key in results:
        print("{}: {}".format(key, results[key]))

def correct_col_names(df):
     return df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

def identify_d_type(df):
    num_cols = list(df.select_dtypes("number").columns)
    cat_cols = list(df.select_dtypes(exclude=["number"]).columns)
    return cat_cols, num_cols
