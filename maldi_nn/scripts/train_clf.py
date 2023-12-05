import os
import argparse
import numpy as np
import pandas as pd
import torch
from maldi_nn.models import SpeciesClassifier, MaldiTransformer
from maldi_nn.utils.data import SpeciesClfDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from maldi_nn.spectrum import PeakFilter
import ast
from maldi_nn.reproduce.modules import (
    MaldiTransformerNegSampler,
    MaldiTransformerOnlyClf,
    MaldiTransformerMaskMSE
)

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass


    parser = argparse.ArgumentParser(
        description="Training script for species identification.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("path", type=str, metavar="path", help="path to h5torch file.")
    parser.add_argument("logs_path", type=str, metavar="logs_path", help="path to logs.")

    parser.add_argument(
        "spectrum_embedder",
        type=str,
        metavar="spectrum_embedder",
        choices=["mlp", "trf"],
        help="Which spectrum embedder to use, choices: {%(choices)s}",
    )

    parser.add_argument(
        "size",
        type=str,
        metavar="size",
        choices=["Linear", "S", "M", "L", "XL"] ,
        help="Model size, choices: {%(choices)s}. Linear is only available for mlp.",
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")

    parser.add_argument(
        "--logging_file",
        type=str,
        default="res.txt",
        help="Which file to write final performances to.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in dataloader. Reduce to alleviate CPU.",
    )

    parser.add_argument(
        "--devices",
        type=ast.literal_eval,
        default=1,
        help="devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus.",
    )

    parser.add_argument(
        "--trf_n_peaks",
        type=int,
        default=200,
        help="Number of peaks",
    )

    parser.add_argument(
        "--trf_ckpt_path",
        type=str,
        default="None",
        help="Checkpoint path of malditransformer",
    )

    parser.add_argument(
        "--trf_ckpt_modeltype",
        type=str,
        default="vanilla",
        choices=["vanilla", "negpeaksampler", "intensitymlm", "onlyclf", "onlyshf"],
        help="Maldi Transformer pre-trained modeltype. choices: {%(choices)s}",
    )

    parser.add_argument(
        "--trf_transfer_output_head",
        type=boolean,
        default=False,
        help="Whether to transfer the clf output head of the Maldi Transformer. Can be set to true if domain adaptation was adopted.",
    )


    args = parser.parse_args()

    dm = SpeciesClfDataModule(
        args.path,
        batch_size=128,
        n_workers=args.num_workers,
        preprocessor=(
            None if args.spectrum_embedder == "mlp" else PeakFilter(max_number=args.trf_n_peaks)
        ),
        in_memory=True,
    )
    dm.setup(None)

    genus = pd.Series(dm.train.f["unstructured/species_labels"][:].view(np.ndarray)).astype(str).str.split(" ", expand=True)[0].values
    mapper = {k : v for v, k in enumerate(np.unique(genus))}
    genus_labels = torch.tensor([mapper[k] for k in genus])

    size_to_layer_dims = {
        "mlp": {
            "S": [256, 128],
            "M": [512, 256, 128],
            "L": [1024, 512, 256, 128],
            "XL": [2048, 1024, 512, 256, 128],
            "Linear": [],
        },
        "trf": {
            "S": [160, 4],
            "M": [184, 6],
            "L": [232, 8],
            "XL": [304, 10],
        },
    }

    if args.spectrum_embedder == "mlp":
        kwargs = {
            "n_inputs": 6000,
            "n_outputs": dm.n_species,
            "layer_dims": size_to_layer_dims["mlp"][args.size],
            "dropout": 0.2,
        }

    elif args.spectrum_embedder == "trf":
        kwargs = {
            "dim": size_to_layer_dims["trf"][args.size][0],
            "depth": size_to_layer_dims["trf"][args.size][1],
            "output_head_dim": dm.n_species,
            "reduce": "cls",
            "dropout": 0.2,
        }
        if args.trf_ckpt_path != "None":
            modeltype = MaldiTransformer
            if args.trf_ckpt_modeltype == "intensitymlm":
                modeltype = MaldiTransformerMaskMSE
            elif args.trf_ckpt_modeltype == "negpeaksampler":
                modeltype = MaldiTransformerNegSampler
            elif args.trf_ckpt_modeltype == "onlyclf":
                modeltype = MaldiTransformerOnlyClf

            malditransformer = modeltype.load_from_checkpoint(args.trf_ckpt_path)
            kwargs["depth"] = malditransformer.hparams.depth
            kwargs["dim"] = malditransformer.hparams.dim


    model = SpeciesClassifier(
        spectrum_embedder=args.spectrum_embedder,
        spectrum_kwargs=kwargs,
        lr=args.lr,
        weight_decay=0,
        lr_decay_factor=1,
        warmup_steps=250,
        genus_labels = genus_labels
    )

    if args.trf_ckpt_path != "None":
        pretrained_dict = malditransformer.transformer.state_dict()
        model_state_dict = model.spectrum_embedder.state_dict()
        if not args.trf_transfer_output_head:
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if not k.startswith("output_head")
            }
        else:
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
            }
        model_state_dict.update(pretrained_dict)
        model.spectrum_embedder.load_state_dict(model_state_dict)


    val_ckpt = ModelCheckpoint(monitor="val_acc", mode="max")
    callbacks = [val_ckpt, EarlyStopping(monitor="val_acc", patience=10, mode="max")]
    logger = TensorBoardLogger(
        args.logs_path,
        name="clf_%s_%s_%s_%s" % (args.spectrum_embedder, args.size, args.lr, args.trf_n_peaks),
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="auto",
        max_epochs=250,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # validation
    p = trainer.validate(
        model, dataloaders=dm.val_dataloader(), ckpt_path=val_ckpt.best_model_path
    )[0]

    with open(os.path.join(args.logs_path, args.logging_file), "a") as f:
        f.write(
            "%s\tval\t%.5f\t%.5f\n"
            % (val_ckpt.best_model_path, p["val_acc"], p["val_genus_acc"])
        )

    # test
    p = trainer.validate(
        model, dataloaders=dm.test_dataloader(), ckpt_path=val_ckpt.best_model_path
    )[0]

    with open(os.path.join(args.logs_path, args.logging_file), "a") as f:
        f.write(
            "%s\ttest\t%.5f\t%.5f\n"
            % (val_ckpt.best_model_path, p["val_acc"], p["val_genus_acc"])
        )

if __name__ == "__main__":
    main()