from maldi_nn.utils.data import DRIAMSSpectrumDataModule
import torch
from maldi_nn.spectrum import *
import h5torch
import sys
import argparse

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

def main():
    parser = argparse.ArgumentParser(
        description="Estimate distribution over negative peaks, and add it to the h5torch file. Necessary to run before training Maldiformer using sampled negative peaks.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("path", type=str, metavar="path", help="path to h5torch file.")
    parser.add_argument(
        "--n_peaks",
        type=int,
        default=200,
        help="Number of peaks",
    )
    args = parser.parse_args()
    
    dm = DRIAMSSpectrumDataModule(
        args.path,
        batch_size=128,
        n_workers=8,
        preprocessor=SequentialPreprocessor(
            PeakFilter(args.n_peaks), Binner(start=2000, stop=20000, step=1)
        ),
        min_spectrum_len=128,
        exclude_nans=False,
        in_memory=False,
    )
    f = h5torch.File(dm.path)
    if dm.min_spectrum_len is not None:
        lens = np.array([len(tt) for tt in f["0/intensity"][:][dm.train_indices]])
        dm.train_indices = dm.train_indices[lens > dm.min_spectrum_len]

        lens = np.array([len(tt) for tt in f["0/intensity"][:][dm.val_indices]])
        dm.val_indices = dm.val_indices[lens > dm.min_spectrum_len]

        lens = np.array([len(tt) for tt in f["0/intensity"][:][dm.test_indices]])
        dm.test_indices = dm.test_indices[lens > dm.min_spectrum_len]


    f2 = f.to_dict()
    f.close()

    dm.train = h5torch.Dataset(f2, subset=dm.train_indices, sample_processor=dm.processor)

    p_x = torch.zeros(dm.train[0]["intensity"].shape)
    p_y_x = [torch.tensor([]) for _ in range(len(p_x))]
    dl = torch.utils.data.DataLoader(
        dm.train,
        num_workers=8,
        batch_size=128,
        shuffle=True,
        pin_memory=False,
    )

    len_ = len(dl)
    print(len_)
    for ix, s in enumerate(iter(dl)):
        p_x += (s["intensity"] != 0).sum(0)

        a = torch.unique(torch.nonzero(s["intensity"])[:, 1])
        for aa in a:
            sub = s["intensity"][:, aa]
            p_y_x[aa] = torch.cat([p_y_x[aa], sub[sub != 0]])
        if (ix % 25) == 0:
            print(ix)

    p_x = (p_x / p_x.sum()).numpy()

    p_y_x_q = []
    for p in p_y_x:
        if len(p) > 0:
            p_y_x_q.append(torch.quantile(p, torch.linspace(0, 1, 101)))
        else:
            p_y_x_q.append(torch.zeros(101))


    p_y_x = torch.stack(p_y_x_q).numpy()

    f = h5torch.File(args.path, "a")
    f.create_dataset("unstructured/p_x_200", data=p_x)
    f.create_dataset("unstructured/p_y_x_200", data=p_y_x)
    f.close()


if __name__ == "__main__":
    main()
