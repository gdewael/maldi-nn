import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lightning import LightningModule
from torchmetrics.classification import BinaryAUROC, MulticlassAccuracy
from maldi_nn.utils.drug import (
    DrugOneHotEmbedding,
    DrugMLP,
    DrugGRU,
    DrugCNN,
    DrugTransformer,
    DrugImageCNN,
)
import maldi_nn.nn as maldinn


class MaldiLightningModule(LightningModule):
    def __init__(self, lr=1e-4, weight_decay=0, lr_decay_factor=0.95, warmup_steps=500):
        """
        A wrapper around pl.LightningModule with some standard stuff applicable to all models.
        """
        super().__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambd
        )
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [("total", total_params)]
        return params_per_layer

    def train_indices_select(self, in_, train_indices):
        if train_indices.dtype == torch.long:
            return torch.gather(in_, 1, train_indices)
        elif train_indices.dtype == torch.bool:
            return in_[train_indices]
        else:
            raise ValueError("train_indices should be either .long or .bool")


class AMRModel(MaldiLightningModule):
    def __init__(
        self,
        spectrum_embedder="mlp",
        drug_embedder="ecfp",
        spectrum_kwargs={},
        drug_kwargs={},
        lr=1e-4,
        weight_decay=0,
        lr_decay_factor=1.00,
        warmup_steps=250,
        scaled_dot_product=False,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_factor=lr_decay_factor,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()

        if drug_embedder == "onehot":
            self.drug_embedder = DrugOneHotEmbedding(**drug_kwargs)
        elif drug_embedder == "ecfp":
            self.drug_embedder = DrugMLP(**drug_kwargs)
        elif drug_embedder == "gru":
            self.drug_embedder = DrugGRU(**drug_kwargs)
        elif drug_embedder == "cnn":
            self.drug_embedder = DrugCNN(**drug_kwargs)
        elif drug_embedder == "trf":
            self.drug_embedder = DrugTransformer(**drug_kwargs)
        elif drug_embedder == "img":
            self.drug_embedder = DrugImageCNN(**drug_kwargs)
        elif drug_embedder == "kernel":
            self.drug_embedder = DrugMLP(**drug_kwargs)

        if spectrum_embedder == "mlp":
            self.spectrum_embedder = maldinn.MLP(**spectrum_kwargs)
        elif spectrum_embedder == "trf":
            self.spectrum_embedder = maldinn.Transformer(**spectrum_kwargs)

        self.scale = scaled_dot_product

        self.auroc = BinaryAUROC()

    def forward(self, batch):
        drug_embedding = self.drug_embedder(batch["drug"])
        spectrum_embedding = self.spectrum_embedder(batch)
        norm = 1 if not self.scale else spectrum_embedding.shape[-1] ** 0.5
        return (drug_embedding * spectrum_embedding).sum(-1) / norm

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)
        batch["drug"] = batch["drug"].to(self.dtype)

        logits = self(batch)

        loss = F.binary_cross_entropy_with_logits(
            logits, batch["label"].to(logits.dtype)
        )

        self.log("train_loss", loss, batch_size=len(batch["label"]))
        return loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)
        batch["drug"] = batch["drug"].to(self.dtype)

        logits = self(batch)

        loss = F.binary_cross_entropy_with_logits(
            logits, batch["label"].to(logits.dtype)
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["label"]),
        )

        self.auroc(logits, batch["label"])
        self.log(
            "val_micro_auc",
            self.auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["label"]),
        )

    def predict_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)
        batch["drug"] = batch["drug"].to(self.dtype)

        logits = self(batch)

        return (
            torch.stack([batch["label"].to(logits), logits], -1),
            batch["loc"],
            batch["drug_name"],
        )


class SpeciesClassifier(MaldiLightningModule):
    def __init__(
        self,
        spectrum_embedder="mlp",
        spectrum_kwargs={},
        lr=1e-4,
        weight_decay=0,
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

        if spectrum_embedder == "mlp":
            self.spectrum_embedder = maldinn.MLP(**spectrum_kwargs)
            n_classes = spectrum_kwargs["n_outputs"]
        elif spectrum_embedder == "trf":
            self.spectrum_embedder = maldinn.Transformer(**spectrum_kwargs)
            n_classes = spectrum_kwargs["output_head_dim"]

        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.top5_accuracy = MulticlassAccuracy(
            num_classes=n_classes, top_k=5, average="micro"
        )

    def forward(self, batch):
        return self.spectrum_embedder(batch)

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)

        loss = F.cross_entropy(logits, batch["central"])

        self.log("train_loss", loss, batch_size=len(batch["central"]))
        return loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)

        loss = F.cross_entropy(logits, batch["central"])

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["central"]),
        )

        self.accuracy(logits, batch["central"])
        self.log(
            "val_acc",
            self.accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["central"]),
        )
        self.top5_accuracy(logits, batch["central"])
        self.log(
            "val_top5_acc",
            self.top5_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["central"]),
        )

    def predict_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)

        return (
            torch.stack([batch["central"].to(logits).unsqueeze(-1), logits], -1),
            batch["0/loc"],
        )


class MaldiTransformer(MaldiLightningModule):
    def __init__(
        self,
        depth,
        dim,
        n_classes=64,
        n_heads=8,
        dropout=0.2,
        p=0.125,
        clf=False,
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
        self.p = p

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

        batch = self.shuffler(batch)

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

        batch = self.shuffler(batch)

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

        self.auroc(mlm_logits_train, trues_train)
        self.log(
            "val_mlmmicro_auc",
            self.auroc,
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

    def predict_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        batch = self.shuffler(batch)

        outputs = self(batch)
        if self.clf:
            mlm_logits, clf_logits = outputs
        else:
            mlm_logits = outputs

        logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mzs_train = self.train_indices_select(batch["mz"], batch["train_indices"])
        intensities_train = self.train_indices_select(
            batch["intensity"], batch["train_indices"]
        )

        if self.clf:
            return (
                np.array(batch["loc"])[
                    torch.where(batch["train_indices"])[0].cpu().numpy()
                ],
                batch["species"][torch.where(batch["train_indices"])[0]],
                mzs_train,
                intensities_train,
                logits_train,
                trues_train,
                clf_logits,
                batch["species"],
            )
        else:
            return (
                np.array(batch["loc"])[
                    torch.where(batch["train_indices"])[0].cpu().numpy()
                ],
                batch["species"][torch.where(batch["train_indices"])[0]],
                mzs_train,
                intensities_train,
                logits_train,
                trues_train,
            )

    def shuffler(self, batch):
        mz = batch["mz"]
        intensity = batch["intensity"]
        n = len(mz) // 2

        all_indices = torch.stack(torch.where(mz)).T

        shuff, pos = torch.chunk(
            torch.randperm(len(all_indices), device=all_indices.device)[
                : int(len(all_indices) * self.p)
            ],
            2,
        )
        shuffled_shuff = shuff[torch.randperm(len(shuff), device=shuff.device)]

        indexer = (all_indices[shuff] != all_indices[shuffled_shuff])[:, 0]

        shuff = shuff[indexer]
        shuffled_shuff = shuffled_shuff[indexer]
        pos = pos[indexer]

        train_indices = torch.zeros_like(mz.bool())
        train_indices[all_indices[pos][:, 0], all_indices[pos][:, 1]] = True
        train_indices[all_indices[shuff][:, 0], all_indices[shuff][:, 1]] = True

        intensity_true = torch.ones_like(mz).long()
        intensity_true[all_indices[shuff][:, 0], all_indices[shuff][:, 1]] = 0

        all_indices[shuff] = all_indices[shuffled_shuff]

        batch["mz"] = mz[all_indices[:, 0], all_indices[:, 1]].view_as(mz)
        batch["intensity"] = intensity[all_indices[:, 0], all_indices[:, 1]].view_as(
            intensity
        )
        batch["intensity_true"] = intensity_true
        batch["train_indices"] = train_indices
        return batch
