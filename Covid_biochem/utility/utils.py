import random
import numpy as np
import os
import torch

from sklearn.metrics import accuracy_score, f1_score

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
