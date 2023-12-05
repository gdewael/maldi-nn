import argparse
import ast
from maldi_nn.models import MaldiTransformer
from maldi_nn.utils.data import DRIAMSSpectrumDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from maldi_nn.spectrum import PeakFilter
from maldi_nn.reproduce.modules import (
    NegativeSamplerPlugin,
    MaskingPlugin,
    DRIAMSSpectrumDataModuleWithNoiser,
    MaldiTransformerNegSampler,
    MaldiTransformerOnlyClf,
    MaldiTransformerMaskMSE
)
import h5torch

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
        description="Training script for Maldi Transformer.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("path", type=str, metavar="path", help="path to h5torch file.")
    parser.add_argument("logs_path", type=str, metavar="logs_path", help="path to logs.")
    parser.add_argument(
        "spectrum_embedder",
        type=str,
        metavar="spectrum_embedder",
        choices=["S", "M", "L", "XL"],
        help="Which size spectrum embedder to use",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="vanilla",
        choices=["vanilla", "negpeaksampler", "intensitymlm", "onlyclf", "onlyshf"],
        help="Maldi Transformer training mode, choices: {%(choices)s} Note that negpeaksampler requires to run reproduce.estimate_peak_distr first.",
        )

    parser.add_argument(
        "--n_peaks",
        type=int,
        default=200,
        help="Number of peaks",
    )

    parser.add_argument(
        "--p",
        type=float,
        default=0.15,
        help="shuffle freq",
    )

    parser.add_argument(
        "--lmbda",
        type=float,
        default=1 / 100,
        help="Lambda. This is the probability with which to apply the spec id loss per step. ",
    )

    parser.add_argument(
        "--lmbda2",
        type=float,
        default=1.,
        help="Additionally, fixed multiplier to apply to the spec id loss.",
    )

    parser.add_argument(
        "--steps",
        type=float,
        default=500_000,
        help="steps",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="batch size per gpu, effective batch size is this value times the number of gpus. (pytorch ddp).",
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")

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

    args = parser.parse_args()

    size_to_layer_dims = {
        "S": [160, 4],
        "M": [184, 6],
        "L": [232, 8],
        "XL": [304, 10],
    }

    if args.mode == "negpeaksampler":
        dm = DRIAMSSpectrumDataModuleWithNoiser(
            args.path,
            batch_size=args.batch_size,
            n_workers=args.num_workers,
            preprocessor=PeakFilter(max_number=args.n_peaks),
            min_spectrum_len=128,
            in_memory=True,
            exclude_nans=False,
            noiser=NegativeSamplerPlugin(
                p_x=h5torch.File(args.path)["unstructured/p_x_200"][:],
                p_y_x=h5torch.File(args.path)["unstructured/p_y_x_200"][:],
                train_on=int(args.n_peaks * args.p / 2),
                n_to_sample=int(args.n_peaks * args.p / 2),
                range=0.5,
                binningstep=1,
            ),
        )
    elif args.mode == "intensitymlm":
        dm = DRIAMSSpectrumDataModuleWithNoiser(
            args.path,
            batch_size=args.batch_size,
            n_workers=args.num_workers,
            preprocessor=PeakFilter(max_number=args.n_peaks),
            min_spectrum_len=128,
            in_memory=True,
            exclude_nans=False,
            noiser=MaskingPlugin(prob=args.p, unchanged=0.2),
        )
    else:
        dm = DRIAMSSpectrumDataModule(
            args.path,
            batch_size=args.batch_size,
            n_workers=args.num_workers,
            preprocessor=PeakFilter(max_number=args.n_peaks),
            min_spectrum_len=128,
            in_memory=True,
            exclude_nans=(True if args.mode == "onlyclf" else False),
        )


    modeltype = MaldiTransformer
    model_kwargs = {
        "n_classes" : dm.n_species - 1,
        "n_heads" : 8,
        "dropout" : 0.2,
        "p" : args.p,
        "clf" : True,
        "clf_train_p" : args.lmbda,
        "lmbda" : args.lmbda2,
        "lr" : args.lr,
        "weight_decay" : 0,
        "lr_decay_factor" : 1,
        "warmup_steps" : 2500,
    }

    if args.mode == "onlyclf":
        modeltype = MaldiTransformerOnlyClf
        model_kwargs["n_classes"] = dm.n_species
        del model_kwargs["clf"]
        del model_kwargs["p"]
        del model_kwargs["clf_train_p"]
        del model_kwargs["lmbda"]
    elif args.mode == "negpeaksampler":
        modeltype = MaldiTransformerNegSampler
        del model_kwargs["lmbda"]
        del model_kwargs["p"]
    elif args.mode == "intensitymlm":
        modeltype = MaldiTransformerMaskMSE
        del model_kwargs["lmbda"]
        del model_kwargs["p"]
    elif args.mode == "onlyshf":
        model_kwargs["clf"] = False


    model = modeltype(
        size_to_layer_dims[args.spectrum_embedder][1],
        size_to_layer_dims[args.spectrum_embedder][0],
        **model_kwargs,
    )


    if args.mode == "onlyclf":
        val_ckpt = ModelCheckpoint(monitor="val_clfacc", mode="max")
        callbacks = [val_ckpt, EarlyStopping(monitor="val_clfacc", patience=10, mode="max")]
        logger = TensorBoardLogger(
            args.logs_path,
            name="malditrf%s_%s_%s_%s"
            % (args.mode, args.spectrum_embedder, args.n_peaks, args.lr),
        )
        trainer = Trainer(
            accelerator="gpu",
            devices=args.devices,
            strategy="auto",
            max_epochs=250,
            callbacks=callbacks,
            logger=logger,
            precision="bf16-mixed",
        )

    else:
        logger = TensorBoardLogger(
            args.logs_path,
            name="malditrf%s_%s_%s_%s_%s_%s"
            % (args.mode, args.spectrum_embedder, args.n_peaks, args.p, args.lmbda, args.lr),
        )
        callbacks = [
            ModelCheckpoint(every_n_train_steps=10_000),
        ]

        trainer = Trainer(
            accelerator="gpu",
            devices=args.devices,
            strategy="auto",
            gradient_clip_val=1,
            max_steps=args.steps,
            max_epochs=-1,
            val_check_interval=5_000,
            check_val_every_n_epoch=None,
            callbacks=callbacks,
            logger=logger,
            precision="bf16-mixed",
        )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()