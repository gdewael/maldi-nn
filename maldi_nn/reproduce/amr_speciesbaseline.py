import pandas as pd
import numpy as np
import os
import h5torch
import ast
import argparse
from maldi_nn.utils.metrics import *
from maldi_nn.spectrum import PeakFilter
from maldi_nn.models import AMRModel, MaldiTransformer
from lightning.pytorch import Trainer
from maldi_nn.utils.data import DRIAMSAMRDataModule
from maldi_nn.reproduce.modules import (
    MaldiTransformerNegSampler,
    MaldiTransformerOnlyClf,
    MaldiTransformerMaskMSE
)
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass

size_to_layer_dims ={
    "mlp" : {
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
    }
}

drug_encoder_args = {
    "onehot": {},
    "kernel": {},
    "ecfp": {"bits": 512, "diameter": 4},
    "cnn": {"alphabet": "deepsmiles"},
    "trf": {"alphabet": "deepsmiles"},
    "gru": {"alphabet": "deepsmiles"},
    "img": {"size": (128, 128), "normalize": True, "inverse": True},
}

def main():
    parser = argparse.ArgumentParser(
        description="Training script for dual-branch AMR recommender.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("path", type=str, metavar="path", help="path to h5torch file.")
    parser.add_argument(
        "logs_path", type=str, metavar="logs_path", help="path to logs."
    )
    parser.add_argument(
        "drug_embedder",
        type=str,
        metavar="drug_embedder",
        choices=["ecfp", "onehot", "gru", "cnn", "trf", "img", "kernel"],
        help="Which drug embedder to use, choices: {%(choices)s}",
    )

    parser.add_argument(
        "spectrum_embedder",
        type=str,
        metavar="spectrum_embedder",
        choices=["trf", "mlp"],
        help="Which spectrum embedder to use, choices: {%(choices)s}",
    )

    parser.add_argument(
        "spectrum_embedder_size",
        type=str,
        metavar="spectrum_embedder_size",
        choices=["S", "M", "L", "XL", "Linear"],
        help="Which size to use for spectrum embedder, choices: {%(choices)s}. Linear is only available for mlp.",
    )

    parser.add_argument(
        "--logging_file",
        type=str,
        default="res.npz",
        help="Which file to write final predictions to.",
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
        help="Number of peaks for transformer-peak-based models",
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
    args = parser.parse_args()

    f = h5torch.File(args.path).to_dict()

    s, c = np.unique(f["0/species"][np.unique(f["central/indices"][0][f["unstructured/split"][:] == b"A_train"])], return_counts = True)
    set_ = s[np.argsort(c)[::-1]][:25].view(np.ndarray)


    locs_final = []
    drug_names_final = []
    preds_final = []
    trues_final = []
    print(set_, len(set_))
    for s in set_:
        print(s, "-" * 100)
        rows_with_species = np.where(f["0/species"][:] == int(s))[0]

        train = np.logical_and(np.isin(f["central/indices"][:][0], rows_with_species), f["unstructured/split"][:] == b"A_train")
        val = np.logical_and(np.isin(f["central/indices"][:][0], rows_with_species), f["unstructured/split"][:] == b"A_val")
        test = np.logical_and(np.isin(f["central/indices"][:][0], rows_with_species), f["unstructured/split"][:] == b"A_test")


        train_indices = np.where(train)[0]
        val_indices = np.where(val)[0]
        test_indices = np.where(test)[0]


        dm = DRIAMSAMRDataModule(
            f,
            drug_encoder=args.drug_embedder,
            drug_encoder_args=drug_encoder_args[args.drug_embedder],
            batch_size=128,
            n_workers=args.num_workers,
            preprocessor=(None if args.spectrum_embedder == "mlp" else PeakFilter(max_number=args.trf_n_peaks)),
            min_spectrum_len=None,
            in_memory=True,
            train_indices = train_indices,
            val_indices = val_indices,
            test_indices = test_indices,
        )
        dm.setup(None)

        drug_embedder_kwargs = {
            "onehot": {
                "num_drugs": len(dm.drug_encoder.vocab),
                "dim": 64,
            },
            "kernel": {
                "n_inputs": len(dm.drug_encoder.embeds),
                "n_outputs": 64,
                "layer_dims": [],
                "dropout": 0.2,
            },
            "ecfp": {"n_inputs": 512, "n_outputs": 64, "layer_dims": [], "dropout": 0.2},
            "cnn": {
                "vocab_size": len(dm.drug_encoder.alphabet_dict),
                "dim": 64,
                "depth": 4,
                "kernel_size": 5,
            },
            "trf": {
                "vocab_size": len(dm.drug_encoder.alphabet_dict),
                "dim": 64,
                "depth": 4,
                "n_head": 8,
            },
            "gru": {"vocab_size": len(dm.drug_encoder.alphabet_dict), "hidden_dim": 64},
            "img": {"stem_downsample": 2, "kernel_size": 5, "hidden_dim": 32, "depth": 2},
        }

        if args.spectrum_embedder == "mlp":
            spectrum_kwargs = {
                "n_inputs": 6000,
                "n_outputs": 64,
                "layer_dims": size_to_layer_dims["mlp"][args.spectrum_embedder_size],
                "dropout": 0.2,
            }
        elif args.spectrum_embedder == "trf":
            spectrum_kwargs = {
                "depth": size_to_layer_dims["trf"][args.spectrum_embedder_size][1],
                "dim": size_to_layer_dims["trf"][args.spectrum_embedder_size][0],
                "output_head_dim": 64,
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
                spectrum_kwargs["depth"] = malditransformer.hparams.depth
                spectrum_kwargs["dim"] = malditransformer.hparams.dim

        valscores_all = []
        preds_test_per_lr_ = []
        trues_test_per_lr_ = []
        locs_test_per_lr_ = []
        drugs_test_per_lr_ = []
        for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
            model = AMRModel(
                spectrum_embedder=args.spectrum_embedder,
                drug_embedder=args.drug_embedder,
                spectrum_kwargs=spectrum_kwargs,
                drug_kwargs=drug_embedder_kwargs[args.drug_embedder],
                lr=lr,
                weight_decay=0,
                lr_decay_factor=1,
                warmup_steps=250,
                scaled_dot_product=True,
            )

            if (args.trf_ckpt_path != "None") and (args.spectrum_embedder == "trf"):
                pretrained_dict = malditransformer.transformer.state_dict()
                model_state_dict = model.spectrum_embedder.state_dict()
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if not k.startswith("output_head")
                }
                model_state_dict.update(pretrained_dict)
                model.spectrum_embedder.load_state_dict(model_state_dict)


            val_ckpt = ModelCheckpoint(monitor="val_roc_auc", mode="max")
            callbacks = [
                val_ckpt,
                EarlyStopping(monitor="val_roc_auc", patience=10, mode="max"),
            ]
            logger = TensorBoardLogger(
                os.path.join(args.logs_path, str(s)),
                name="amr%s_%s_%s_%s" % (args.spectrum_embedder, args.drug_embedder, args.spectrum_embedder_size, lr),
            )

            trainer = Trainer(
                accelerator="gpu",
                devices=args.devices,
                strategy="auto",
                val_check_interval=0.1,
                max_epochs=50,
                callbacks=callbacks,
                logger=logger,
            )

            trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

            # validation
            p = trainer.predict(
                model, dataloaders=dm.val_dataloader(), ckpt_path=val_ckpt.best_model_path
            )

            trues = torch.cat([v[1] for v in p]).numpy()
            preds = torch.cat([v[0] for v in p]).numpy()
            locs = np.concatenate([v[2] for v in p])
            drugs = np.concatenate([v[3] for v in p])
            valscores_all.append(ic_roc_auc(preds, trues, locs, drugs)[0])
            

            # test
            p = trainer.predict(
                model, dataloaders=dm.test_dataloader(), ckpt_path=val_ckpt.best_model_path
            )

            trues = torch.cat([v[1] for v in p]).numpy()
            preds = torch.cat([v[0] for v in p]).numpy()
            locs = np.concatenate([v[2] for v in p])
            drugs = np.concatenate([v[3] for v in p])

            preds_test_per_lr_.append(preds)
            trues_test_per_lr_.append(trues)
            locs_test_per_lr_.append(locs)
            drugs_test_per_lr_.append(drugs)

        max_index = np.argmax(valscores_all)
        locs_final.append(locs_test_per_lr_[max_index])
        drug_names_final.append(drugs_test_per_lr_[max_index])
        preds_final.append(preds_test_per_lr_[max_index])
        trues_final.append(trues_test_per_lr_[max_index])

        
    np.savez(
        os.path.join(args.logs_path, args.logging_file),
        **{
            "trues": np.concatenate(trues_final),
            "preds": np.concatenate(preds_final),
            "locs": np.concatenate(locs_final),
            "drug_names": np.concatenate(drug_names_final),
        }
    )

if __name__ == "__main__":
    main()