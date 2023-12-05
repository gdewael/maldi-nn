import os

os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=1

import torch
import numpy as np
from maldi_nn.utils.data import SpeciesClfDataModule
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchmetrics.functional.classification import multiclass_accuracy
from sklearn.model_selection import ParameterGrid
import argparse

class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass

def read_data(path):
    dm = SpeciesClfDataModule(
        path,
        batch_size=128,
        n_workers=2,
        preprocessor=None,
        in_memory=True,
    )
    dm.setup(None)


    X_train = torch.stack([b["intensity"] for b in dm.train]).numpy()
    y_train = np.array([b["species"] for b in dm.train])
    X_val = torch.stack([b["intensity"] for b in dm.val]).numpy()
    y_val = np.array([b["species"] for b in dm.val])
    X_test = torch.stack([b["intensity"] for b in dm.test]).numpy()
    y_test = np.array([b["species"] for b in dm.test])
    test_locs = np.array([b["0/loc"] for b in dm.test])
    return X_train, X_val, X_test, y_train, y_val, y_test, test_locs

def fit_knn(X_train, y_train, params):
    if params["norm"] == "standardscaler":
        model = Pipeline(
            steps=[
                ("norm", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=params["n_neighbors"],
                    ),
                ),
            ]
        )
    else:
        model = KNeighborsClassifier(
                    n_neighbors=params["n_neighbors"],
                )
        
    model.fit(X_train, y_train)
    return model

def fit_lr(X_train, y_train, params):
    if params["norm"] == "standardscaler":
        model = Pipeline(
            steps=[
                ("norm", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=500,
                        C=params["C"],
                    ),
                ),
            ]
        )
    else:
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=params["C"],
        )
    model.fit(X_train, y_train)
    return model

def fit_rf(X_train, y_train, params):
    model = RandomForestClassifier(**params, n_estimators = 200, n_jobs = 12)
    model.fit(X_train, y_train)
    return model


def main_knn(args):
    X_train, X_val, X_test, y_train, y_val, y_test, test_locs = read_data(args.path)

    knn_grid = ParameterGrid(
        {
            "norm": ["passthrough", "standardscaler"],
            "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25],
        }
    )

    scores = []
    for params in knn_grid:
        model = fit_knn(X_train, y_train, params)
        preds = torch.tensor(model.predict_proba(X_val))

        print(params, multiclass_accuracy(preds, torch.tensor(y_val), num_classes = preds.shape[1], average="micro"), flush = True)
        scores.append(
            [params, multiclass_accuracy(preds, torch.tensor(y_val), num_classes = preds.shape[1], average="micro")]
        )

    max_index = np.argmax([j[-1] for j in scores])
    winning_params = scores[max_index][0]

    winning_model = fit_knn(X_train, y_train, winning_params)

    np.savez(
        args.outputs,
        **{
            "trues": y_test,
            "preds": winning_model.predict_proba(X_test),
            "locs": test_locs,
        }
    )

def main_lr(args):
    X_train, X_val, X_test, y_train, y_val, y_test, test_locs = read_data(args.path)
    lr_grid = ParameterGrid(
        {
            "norm": ["passthrough", "standardscaler"],
            "C": 10.0 ** np.arange(-3, 4),
        }
    )
    scores = []
    for params in lr_grid:
        model = fit_lr(X_train, y_train, params)
        preds = torch.tensor(model.predict_proba(X_val))

        print(params, multiclass_accuracy(preds, torch.tensor(y_val), num_classes = preds.shape[1], average="micro"), flush = True)
        scores.append(
            [params, multiclass_accuracy(preds, torch.tensor(y_val), num_classes = preds.shape[1], average="micro")]
        )

    max_index = np.argmax([j[-1] for j in scores])
    winning_params = scores[max_index][0]
    winning_model = fit_lr(X_train, y_train, winning_params)

    np.savez(
        outputs,
        **{
            "trues": y_test,
            "preds": winning_model.predict_proba(X_test),
            "locs": test_locs,
        }
    )

def main_rf(args):
    X_train, X_val, X_test, y_train, y_val, y_test, test_locs = read_data(args.path)

    rf_grid = ParameterGrid(
        {
            "max_depth" : [25, 50, 75, 100],
            "min_samples_split": [2, 5, 10],
            "max_features": [10, 25, 50, 100],
        }
    )

    scores = []
    for params in rf_grid:
        model = fit_rf(X_train, y_train, params)
        preds = torch.tensor(model.predict_proba(X_val))

        print(params, multiclass_accuracy(preds, torch.tensor(y_val), num_classes = preds.shape[1], average="micro"), flush = True)
        scores.append(
            [params, multiclass_accuracy(preds, torch.tensor(y_val), num_classes = preds.shape[1], average="micro")]
        )

    max_index = np.argmax([j[-1] for j in scores])
    winning_params = scores[max_index][0]

    winning_model = fit_rf(X_train, y_train, winning_params)

    np.savez(
        args.outputs,
        **{
            "trues": y_test,
            "preds": winning_model.predict_proba(X_test),
            "locs": test_locs,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for species identification baselines. Returns an npz file with predictions for the test set.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("path", type=str, metavar="path", help="path to h5torch file.")
    parser.add_argument(
        "outputs",
        type=str,
        metavar="outputs.npz",
        help="numpy .npz file to write (test) predictions into.",
    )
    parser.add_argument(
        "modeltype",
        type=str,
        metavar="modeltype",
        choices=["knn", "lr", "rf"],
        help="Which modeltype to use as baseline, choices: {%(choices)s}",
    )

    args = parser.parse_args()

    if args.modeltype == "knn":
        main_knn(args)
    elif args.modeltype == "lr":
        main_lr(args)
    elif args.modeltype == "rf":
        main_rf(args)
    