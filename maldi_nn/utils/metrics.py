import numpy as np
from sklearn.metrics import roc_auc_score, recall_score


def ic_roc_auc(preds, trues, locs, drugs):
    locs = locs.astype(bytes)
    t = []
    for l in np.unique(locs):
        sub = locs == l
        if len(np.unique(trues[sub])) > 1:
            t.append(roc_auc_score(trues[sub], preds[sub]))

    return np.mean(t), np.array(t)


def macro_roc_auc(preds, trues, locs, drugs):
    drugs = drugs.astype(bytes)
    t = []
    for l in np.unique(drugs):
        sub = drugs == l
        if len(np.unique(trues[sub])) > 1:
            t.append(roc_auc_score(trues[sub], preds[sub]))

    return np.mean(t), np.array(t)


def prec_at_1_neg(preds, trues, locs, drugs):
    r, c = np.unique(locs, return_counts=True)
    pos = 0
    total = 0
    for r_ in r[c > 1]:
        trues_sub = trues[locs == r_]
        if len(np.unique(trues[locs == r_])) > 1:
            preds_sub = preds[locs == r_]
            pos += trues_sub[np.argmin(preds_sub)]
            total += 1
    return (total - pos) / total


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ic_sensitivity(preds, trues, locs, drugs, threshold=0.5):
    locs = locs.astype(bytes)
    t = []
    for l in np.unique(locs):
        sub = locs == l
        if not (trues[sub] == 0).all():
            t.append(recall_score(trues[sub], sigmoid(preds[sub]) >= threshold))

    return np.mean(t), np.array(t)


def ic_specificity(preds, trues, locs, drugs, threshold=0.5):
    locs = locs.astype(bytes)
    t = []
    for l in np.unique(locs):
        sub = locs == l
        if not (trues[sub] == 1).all():
            t.append(recall_score(1 - trues[sub], sigmoid(preds[sub]) < threshold))

    return np.mean(t), np.array(t)
