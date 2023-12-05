import argparse
import ast
from maldi_nn.models import MaldiTransformer
from maldi_nn.utils.data import SpeciesClfDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from maldi_nn.spectrum import PeakFilter


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
        description="Training script for domain adaptation of a Maldi Transformer.",
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
        "--ckpt_path",
        type=str,
        default="None",
        help="Ckpt path",
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
        default=0.5,
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
        default=20_000,
        help="steps",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="batch size",
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in dataloader. Reduce to alleviate CPU.",
    )
    parser.add_argument("--devices", type=ast.literal_eval)

    args = parser.parse_args()

    size_to_layer_dims = {
        "S": [160, 4],
        "M": [184, 6],
        "L": [232, 8],
        "XL": [304, 10],
    }

    dm = SpeciesClfDataModule(
        args.path,
        batch_size=args.batch_size,
        n_workers=args.num_workers,
        preprocessor=PeakFilter(max_number=args.n_peaks),
        in_memory=True,
    )
    dm.setup(None)


    model = MaldiTransformer(
        size_to_layer_dims[args.spectrum_embedder][1],
        size_to_layer_dims[args.spectrum_embedder][0],
        n_classes=dm.n_species,
        n_heads=8,
        dropout=0.2,
        p=args.p,
        clf=True,
        clf_train_p=args.lmbda,
        lmbda = args.lmbda2,
        lr=args.lr,
        weight_decay=0,
        lr_decay_factor=1,
        warmup_steps=2500,
        )

    if args.ckpt_path != "None":
        malditransformer = MaldiTransformer.load_from_checkpoint(args.ckpt_path)
        pretrained_dict = malditransformer.transformer.state_dict()
        model_state_dict = model.transformer.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if not k.startswith("output_head")
        }
        model_state_dict.update(pretrained_dict)
        model.transformer.load_state_dict(model_state_dict)
        
        model.output_head.load_state_dict(malditransformer.output_head.state_dict())

    callbacks = [
        ModelCheckpoint(every_n_train_steps=5_000),
    ]
    logger = TensorBoardLogger(
        args.logs_path,
        name="domainadapt_%s_%s_%s_%s_%s"
        % (args.spectrum_embedder, args.n_peaks, args.p, args.lmbda, args.lr),
    )

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