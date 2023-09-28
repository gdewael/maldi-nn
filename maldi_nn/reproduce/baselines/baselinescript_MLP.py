import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=1

import numpy as np
import h5torch
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from lightning.pytorch import Trainer
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import maldi_nn.nn as maldinn
from maldi_nn.models import MaldiLightningModule
from torchmetrics.classification import BinaryAUROC
import argparse
import ast


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass


class BaselineModel(MaldiLightningModule):
    def __init__(
        self,
        spectrum_kwargs={},
        weight_decay=0,
        lr=1e-4,
        lr_decay_factor=1.00,
        warmup_steps=250,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_factor=lr_decay_factor,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()

        self.spectrum_embedder = maldinn.MLP(**spectrum_kwargs)
        self.auroc = BinaryAUROC()

    def forward(self, batch):
        return self.spectrum_embedder(batch).squeeze(-1)

    def training_step(self, batch, batch_idx):
        X, y = batch
        batch = {"intensity": X.to(self.dtype)}

        logits = self(batch)

        loss = F.binary_cross_entropy_with_logits(logits, y)

        self.log("train_loss", loss, batch_size=len(logits))
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        batch = {"intensity": X.to(self.dtype)}

        logits = self(batch)

        loss = F.binary_cross_entropy_with_logits(logits, y)

        self.log("val_loss", loss, batch_size=len(logits))

        self.auroc(logits, y)
        self.log(
            "val_auroc",
            self.auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(logits),
        )


dict_ = {
    "S": [256, 128],
    "M": [512, 256, 128],
    "L": [1024, 512, 256, 128],
    "XL": [2048, 1024, 512, 256, 128],
}


def main():
    parser = argparse.ArgumentParser(
        description="Training script for non-recommender MLP baselines.",
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
        "size",
        type=str,
        metavar="size",
        choices=["S", "M", "L", "XL", "Linear"],
        help="Which size spectrum embedder to use, choices: {%(choices)s}",
    )
    parser.add_argument(
        "--devices",
        type=ast.literal_eval,
        default=1,
        help="devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus.",
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

        kwargs = {
            "n_inputs": 6000,
            "n_outputs": 1,
            "layer_dims": dict_[args.size],
            "layer_or_batchnorm": "layer",
            "dropout": 0.2,
        }

        valscores_all = []
        preds_test_all = []
        for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
            model = BaselineModel(
                spectrum_kwargs=kwargs,
                lr=lr,
                weight_decay=0,
                lr_decay_factor=1.00,
                warmup_steps=250,
            )

            train_set = TensorDataset(
                torch.tensor(X_train).float(), torch.tensor(y_train).float()
            )
            val_set = TensorDataset(
                torch.tensor(X_val).float(), torch.tensor(y_val).float()
            )
            test_set = TensorDataset(
                torch.tensor(X_test).float(), torch.tensor(y_test).float()
            )
            train_dl = DataLoader(
                train_set, batch_size=128, shuffle=True, pin_memory=True
            )
            val_dl = DataLoader(val_set, batch_size=128, shuffle=False, pin_memory=True)
            test_dl = DataLoader(
                test_set, batch_size=128, shuffle=False, pin_memory=True
            )

            val_ckpt = ModelCheckpoint(monitor="val_auroc", mode="max")
            callbacks = [val_ckpt]
            callbacks += [EarlyStopping(monitor="val_auroc", patience=10, mode="max")]

            trainer = Trainer(
                accelerator="gpu",
                devices=args.devices,
                max_epochs=250,
                callbacks=callbacks,
            )
            trainer.fit(model, train_dl, val_dl)

            model = BaselineModel.load_from_checkpoint(val_ckpt.best_model_path)
            model = model.eval().cpu()
            with torch.no_grad():
                preds_val = []
                for i in np.arange(0, len(val_set), 128):
                    X, _ = val_set[i : i + 128]
                    batch = {
                        "intensity": X.to(model.dtype),
                        "mz": (torch.arange(2000, 20000, 3) + 1.5)
                        .expand(X.shape[0], -1)
                        .to(model.dtype),
                    }
                    preds_val.append(model(batch).numpy())

                valscores_all.append(roc_auc_score(y_val, np.concatenate(preds_val)))

                preds_test = []
                for i in np.arange(0, len(test_set), 128):
                    X, _ = test_set[i : i + 128]
                    batch = {
                        "intensity": X.to(model.dtype),
                        "mz": (torch.arange(2000, 20000, 3) + 1.5)
                        .expand(X.shape[0], -1)
                        .to(model.dtype),
                    }
                    preds_test.append(model(batch).numpy())
                preds_test_all.append(np.concatenate(preds_test))

        # final model
        max_index = np.argmax(valscores_all)

        loc_ = f["0/loc"][:][f["central/indices"][:][0][test]]
        drug_name = f["1/drug_names"][:][f["central/indices"][:][1][test]]

        preds.append(preds_test_all[max_index])
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
