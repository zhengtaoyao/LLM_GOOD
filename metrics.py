import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, average_precision_score

def get_acc(y_true, y_pred, mask):
    n_correct = (y_true[mask] == y_pred[mask]).sum()
    n_total = mask.sum()

    return n_correct / n_total