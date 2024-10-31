import h5torch
from maldi_nn.utils.dna import rRNADataModule, rRNAVAE
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

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

    parser.add_argument("silva_path", type=str, metavar="path", help="path to h5torch file.")
    parser.add_argument("zsl_file_path", type=str, metavar="path", help="path to h5torch file.")
    parser.add_argument(
        "logs_path", type=str, metavar="logs_path", help="path to logs."
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=1e-3, help="beta")

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
        "--data_in_memory",
        type=boolean,
        default=True,
        help="Whether to load the data in memory. Loading in memory typically results in faster training at the cost of increasing memory load.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="max epochs",
    )

    args = parser.parse_args()

    dm = rRNADataModule(args.silva_path, batch_size = 64, n_workers=args.num_workers, in_memory=args.data_in_memory)
    dm.setup(None)

    model = rRNAVAE(
        input_hsize = 16,
        C = 128, 
        n_blocks = 12,
        kernel_size = 5,
        bottleneck_size = 64,
        beta = args.beta,
        lr = args.lr,
        weight_decay=0,
        lr_decay_factor=1.00,
        warmup_steps=250,
    )

    val_ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
    callbacks = [val_ckpt, EarlyStopping(monitor="val_loss", patience=5, mode="min")]
    logger = TensorBoardLogger(
        args.logs_path,
        name="vae_%s_%s" % (args.lr, args.beta),
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="auto",
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        precision="bf16-true",
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    preds = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")

    all_embeds = torch.cat([p[0]for p in preds], 0).float().numpy()
    taxons = np.concatenate([p[1] for p in preds])

    f = h5torch.File(args.zsl_file_path)
    t = f["1/strain_names"][:]
    ss, cc = np.unique([tt.astype(str).split(";")[0] for tt in t], return_counts=True)
    ten_most_common_families = ss[np.argsort(cc)[::-1]][:10]

    color = np.zeros_like(taxons, dtype="int") -1
    for ix, fam in enumerate(ten_most_common_families):
        color[np.array([fam in t for t in taxons])] = ix

    all_embeds = all_embeds[color != -1]
    color = color[color != -1]

    sim = cosine_similarity(all_embeds, all_embeds)
    nn = np.argsort(sim, 1)[:, -2]
    nn_acc = float((color == color[nn]).sum() / len(color))

    with open(os.path.join(args.logs_path, args.logging_file), "a") as f:
            f.write(
                "%s\t%.5f\n"
                % (val_ckpt.best_model_path, nn_acc)
            )

if __name__ == "__main__":
    main()