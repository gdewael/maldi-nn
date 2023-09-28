import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=1

from sklearn.model_selection import ParameterGrid
import numpy as np
import h5torch
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import argparse


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Training script for non-recommender logistic regression baselines.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("path", type=str, metavar="path", help="path to h5torch file.")
    parser.add_argument(
        "outputs",
        type=str,
        metavar="outputs.npz",
        help="numpy .npz file to write (test) predictions into.",
    )
    args = parser.parse_args()

    f = h5torch.File(args.path, "r")

    dr = f["central/indices"][1]
    sp = f["0/species"][:][f["central/indices"][0]]
    comb, cnt = np.unique(
        (pd.Series(sp.astype(str)) + "_" + pd.Series(dr.astype(str))).values[
            (f["unstructured/split"][:] == b"A_train")
        ],
        return_counts=True,
    )
    asrt = np.argsort(cnt)

    locs = []
    drug_names = []
    preds = []
    trues = []

    for t in comb[asrt][-300:][::-1]:
        print(t)
        s, d = t.split("_")
        col_of_drug = int(d)
        rows_with_species = np.where(f["0/species"][:] == int(s))[0]

        indices = np.logical_and(
            f["central/indices"][:][1] == col_of_drug,
            np.isin(f["central/indices"][:][0], rows_with_species),
        )

        train = np.logical_and(indices, f["unstructured/split"][:] == b"A_train")
        val = np.logical_and(indices, f["unstructured/split"][:] == b"A_val")
        test = np.logical_and(indices, f["unstructured/split"][:] == b"A_test")

        X_train = f["0/intensity"][f["central/indices"][:][0][train]]
        y_train = (f["central/data"][:][train] != b"S").astype(int)

        X_val = f["0/intensity"][f["central/indices"][:][0][val]]
        y_val = (f["central/data"][:][val] != b"S").astype(int)

        X_test = f["0/intensity"][f["central/indices"][:][0][test]]
        y_test = (f["central/data"][:][test] != b"S").astype(int)
        if (
            (len(np.unique(y_test)) <= 1)
            or (len(np.unique(y_val)) <= 1)
            or (len(np.unique(y_train)) <= 1)
        ):
            continue

        # tune
        lr_grid = ParameterGrid(
            {
                "norm": ["passthrough", "standardscaler"],
                "penalty": ["l2"],
                "C": 10.0 ** np.arange(-3, 4),
            }
        )

        scores = []
        for params in lr_grid:
            if params["norm"] == "standardscaler":
                model = Pipeline(
                    steps=[
                        ("norm", StandardScaler()),
                        (
                            "lr",
                            LogisticRegression(
                                solver="lbfgs",
                                max_iter=500,
                                penalty=params["penalty"],
                                C=params["C"],
                            ),
                        ),
                    ]
                )
            else:
                model = LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    penalty=params["penalty"],
                    C=params["C"],
                )
            model.fit(X_train, y_train)
            print(params, roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
            scores.append(
                [params, "lr", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])]
            )

        # final model
        max_index = np.argmax([j[-1] for j in scores])

        if scores[max_index][0]["norm"] == "standardscaler":
            m = Pipeline(
                steps=[
                    ("norm", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=500,
                            penalty=scores[max_index][0]["penalty"],
                            C=scores[max_index][0]["C"],
                        ),
                    ),
                ]
            )
        else:
            m = LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                penalty=scores[max_index][0]["penalty"],
                C=scores[max_index][0]["C"],
            )

        m.fit(X_train, y_train)

        loc_ = f["0/loc"][:][f["central/indices"][:][0][test]]
        drug_name = f["1/drug_names"][:][f["central/indices"][:][1][test]]

        preds.append(m.predict_proba(X_test)[:, 1])
        trues.append(y_test)
        locs.append(loc_)
        drug_names.append(drug_name)

    np.savez(
        args.outputs,
        **{
            "trues": np.concatenate(trues),
            "preds": np.concatenate(preds),
            "locs": np.concatenate(locs),
            "drug_names": np.concatenate(drug_names),
        }
    )


if __name__ == "__main__":
    main()
