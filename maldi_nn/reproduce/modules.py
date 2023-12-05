import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maldi_nn.models import MaldiLightningModule
from torchmetrics.classification import BinaryAUROC, MulticlassAccuracy
import maldi_nn.nn as maldinn
from maldi_nn.utils.data import DRIAMSSpectrumDataModule
from maldi_nn.spectrum import (
    Binner,
    SpectrumObject,
    UniformPeakShifter,
    SequentialPreprocessor,
)


class MaldiTransformerOnlyClf(MaldiLightningModule):
    def __init__(
        self,
        depth,
        dim,
        n_classes=64,
        n_heads=8,
        dropout=0.2,
        lr=0.0005,
        weight_decay=0,
        lr_decay_factor=1,
        warmup_steps=2500,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_factor=lr_decay_factor,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()
        self.transformer = maldinn.Transformer(
            depth,
            dim,
            n_heads=n_heads,
            dropout=dropout,
            cls=True,
            reduce="none",
            output_head_dim=n_classes,
        )

        self.n_species = n_classes

        self.auroc = BinaryAUROC()
        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.top5_accuracy = MulticlassAccuracy(
            num_classes=n_classes, top_k=5, average="micro"
        )

    def forward(self, batch):
        z = self.transformer(batch)
        clf_logits = self.transformer.output_head(z[:, 0, :])
        return clf_logits

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        clf_logits = self(batch)

        clf_loss = F.cross_entropy(clf_logits, batch["species"])

        self.log("train_loss", clf_loss, batch_size=len(batch["species"]))
        return clf_loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        clf_logits = self(batch)

        clf_loss = F.cross_entropy(clf_logits, batch["species"])

        self.log(
            "val_clfloss",
            clf_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
            sync_dist=True,
        )
        self.accuracy(clf_logits, batch["species"])
        self.log(
            "val_clfacc",
            self.accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
            sync_dist=True,
        )
        self.top5_accuracy(clf_logits, batch["species"])
        self.log(
            "val_clftop5_acc",
            self.top5_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
            sync_dist=True,
        )

    def predict_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        clf_logits = self(batch)

        return (
            batch["loc"],
            clf_logits,
            batch["species"],
        )


class MaskingPlugin:
    def __init__(self, prob=0.15, unchanged=0.2):
        self.p = prob
        self.u = unchanged

    def __call__(self, spectrum):
        spectrum["intensity_true"] = spectrum["intensity"].clone()
        train_indices = torch.empty(spectrum["intensity"].shape).uniform_() < self.p

        unchanged_indices = torch.empty(spectrum["intensity"].shape).uniform_() < self.u

        spectrum["intensity"].masked_fill_(
            train_indices & ~unchanged_indices, torch.nan
        )
        spectrum["train_indices"] = train_indices
        return spectrum


class NegativeSamplerPlugin:
    def __init__(
        self, p_x, p_y_x, train_on=15, n_to_sample=15, range=0.5, binningstep=1
    ):
        self.p_x = p_x
        self.p_y_x = p_y_x
        self.train_on = train_on
        self.n = n_to_sample
        self.binner = SequentialPreprocessor(
            Binner(step=binningstep), UniformPeakShifter(range=range)
        )
        self.range_ = torch.arange(self.n)

    def __call__(self, spectrum):
        s = SpectrumObject(
            mz=spectrum["mz"].numpy(), intensity=spectrum["intensity"].numpy()
        )
        s = self.binner(s)

        bg_signal = self.p_x.copy()

        neg_locs = np.random.choice(len(bg_signal), size=(self.n), p=bg_signal)
        xs = s.mz[neg_locs]

        quantiles = self.p_y_x[neg_locs]

        q = torch.randint(100, (self.n,))
        ys = (
            torch.rand(self.n, dtype=spectrum["intensity"].dtype)
            * (quantiles[self.range_, q + 1] - quantiles[self.range_, q])
            + quantiles[self.range_, q]
        )

        train_indices = torch.empty(spectrum["intensity"].shape).uniform_() < (
            self.train_on / len(spectrum["intensity"])
        )

        spectrum["mz"] = torch.cat(
            [spectrum["mz"], torch.tensor(xs, dtype=spectrum["mz"].dtype)]
        )
        spectrum["intensity_true"] = torch.cat(
            [
                torch.ones_like(spectrum["intensity"]).long(),
                torch.zeros((len(neg_locs)), dtype=spectrum["intensity"].dtype),
            ]
        )
        spectrum["intensity"] = torch.cat([spectrum["intensity"], ys])

        spectrum["train_indices"] = torch.cat(
            [train_indices, torch.ones(len(neg_locs), dtype=bool)]
        )
        return spectrum


class DRIAMSSpectrumDataModuleWithNoiser(DRIAMSSpectrumDataModule):
    def __init__(
        self,
        path,
        batch_size=512,
        n_workers=4,
        preprocessor=None,
        min_spectrum_len=None,
        in_memory=True,
        exclude_nans=False,
        noiser=None,
    ):
        super().__init__(
            path=path,
            batch_size=batch_size,
            n_workers=n_workers,
            preprocessor=preprocessor,
            min_spectrum_len=min_spectrum_len,
            in_memory=in_memory,
            exclude_nans=exclude_nans,
        )
        self.noiser = noiser

    def processor(self, f, sample):
        spectrum = SpectrumObject(
            mz=(sample["0/mz"] if "0/mz" in sample else f["unstructured/mz"][:]),
            intensity=sample["0/intensity"],
        )

        if self.preprocessor is not None:
            spectrum = self.preprocessor(spectrum)

        spectrum = {
            "intensity": torch.tensor(spectrum.intensity).float(),
            "mz": torch.tensor(spectrum.mz),
        }

        if self.noiser is not None:
            spectrum = self.noiser(spectrum)

        return (
            spectrum
            | {
                "species": (
                    sample["central"]
                    if self.species_mapping is None
                    else self.species_mapping[sample["central"]]
                )
            }
            | {"loc": sample["0/loc"].astype(str)}
        )


class MaldiTransformerMaskMSE(MaldiLightningModule):
    def __init__(
        self,
        depth,
        dim,
        n_classes=64,
        n_heads=8,
        dropout=0.2,
        clf=True,
        clf_train_p=1 / 100,
        lr=0.0005,
        weight_decay=0,
        lr_decay_factor=1,
        warmup_steps=2500,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_factor=lr_decay_factor,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()
        self.transformer = maldinn.Transformer(
            depth,
            dim,
            n_heads=n_heads,
            dropout=dropout,
            cls=True,
            reduce="none",
            output_head_dim=n_classes,
        )

        self.output_head = nn.Linear(dim, 1)

        self.n_species = n_classes
        self.clf = clf
        self.clf_train_p = clf_train_p

        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.top5_accuracy = MulticlassAccuracy(
            num_classes=n_classes, top_k=5, average="micro"
        )

    def forward(self, batch):
        z = self.transformer(batch)
        mlm_logits = self.output_head(z[:, 1:]).squeeze(-1)
        if self.clf:
            clf_logits = self.transformer.output_head(z[:, 0, :])
            return mlm_logits, clf_logits
        else:
            return mlm_logits

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        outputs = self(batch)
        if self.clf:
            mlm_logits, clf_logits = outputs
        else:
            mlm_logits = outputs

        mlm_logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mse_loss = F.mse_loss(mlm_logits_train, trues_train.to(self.dtype))

        if self.clf:
            indexer = batch["species"] < self.n_species
            clf_loss = F.cross_entropy(clf_logits[indexer], batch["species"][indexer])

            self.log(
                "train_loss", mse_loss + clf_loss, batch_size=len(batch["intensity"])
            )
            return mse_loss + clf_loss * (
                1 if (torch.rand(1) < self.clf_train_p).item() else 0
            )
        else:
            self.log("train_loss", mse_loss, batch_size=len(batch["intensity"]))
            return mse_loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        outputs = self(batch)
        if self.clf:
            mlm_logits, clf_logits = outputs
        else:
            mlm_logits = outputs

        mlm_logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mse_loss = F.mse_loss(mlm_logits_train, trues_train.to(self.dtype))

        self.log(
            "val_mseloss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(mlm_logits_train),
            sync_dist=True,
        )

        if self.clf:
            indexer = batch["species"] < self.n_species
            if indexer.sum() > 0:
                clf_loss = F.cross_entropy(
                    clf_logits[indexer], batch["species"][indexer]
                )

                self.log(
                    "val_clfloss",
                    clf_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )
                self.accuracy(clf_logits[indexer], batch["species"][indexer])
                self.log(
                    "val_clfacc",
                    self.accuracy,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )
                self.top5_accuracy(clf_logits[indexer], batch["species"][indexer])
                self.log(
                    "val_clftop5_acc",
                    self.top5_accuracy,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )


class MaldiTransformerNegSampler(MaldiLightningModule):
    def __init__(
        self,
        depth,
        dim,
        n_classes=64,
        n_heads=8,
        dropout=0.2,
        clf=True,
        clf_train_p=1 / 100,
        lr=0.0005,
        weight_decay=0,
        lr_decay_factor=1,
        warmup_steps=2500,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_factor=lr_decay_factor,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()
        self.transformer = maldinn.Transformer(
            depth,
            dim,
            n_heads=n_heads,
            dropout=dropout,
            cls=True,
            reduce="none",
            output_head_dim=n_classes,
        )

        self.output_head = nn.Linear(dim, 1)

        self.n_species = n_classes
        self.clf = clf
        self.clf_train_p = clf_train_p

        self.auroc = BinaryAUROC()
        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.top5_accuracy = MulticlassAccuracy(
            num_classes=n_classes, top_k=5, average="micro"
        )

    def forward(self, batch):
        z = self.transformer(batch)
        mlm_logits = self.output_head(z[:, 1:]).squeeze(-1)
        if self.clf:
            clf_logits = self.transformer.output_head(z[:, 0, :])
            return mlm_logits, clf_logits
        else:
            return mlm_logits

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        outputs = self(batch)
        if self.clf:
            mlm_logits, clf_logits = outputs
        else:
            mlm_logits = outputs

        mlm_logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mlm_loss = F.binary_cross_entropy_with_logits(
            mlm_logits_train, trues_train.to(self.dtype)
        )

        if self.clf:
            indexer = batch["species"] < self.n_species
            clf_loss = F.cross_entropy(clf_logits[indexer], batch["species"][indexer])

            self.log(
                "train_loss", mlm_loss + clf_loss, batch_size=len(batch["intensity"])
            )
            return mlm_loss + clf_loss * (
                1 if (torch.rand(1) < self.clf_train_p).item() else 0
            )
        else:
            self.log("train_loss", mlm_loss, batch_size=len(batch["intensity"]))
            return mlm_loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        outputs = self(batch)
        if self.clf:
            mlm_logits, clf_logits = outputs
        else:
            mlm_logits = outputs

        mlm_logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mlm_loss = F.binary_cross_entropy_with_logits(
            mlm_logits_train, trues_train.to(self.dtype)
        )

        self.log(
            "val_mlmloss",
            mlm_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(mlm_logits_train),
            sync_dist=True,
        )

        if self.clf:
            indexer = batch["species"] < self.n_species
            if indexer.sum() > 0:
                clf_loss = F.cross_entropy(
                    clf_logits[indexer], batch["species"][indexer]
                )

                self.log(
                    "val_clfloss",
                    clf_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )
                self.accuracy(clf_logits[indexer], batch["species"][indexer])
                self.log(
                    "val_clfacc",
                    self.accuracy,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )
                self.top5_accuracy(clf_logits[indexer], batch["species"][indexer])
                self.log(
                    "val_clftop5_acc",
                    self.top5_accuracy,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )
