# Reproducing study results

This repository/package contains all code to reproduce the main results in our papers [(1) "An antimicrobial drug recommender system using MALDI-TOF MS and dual-branch neural networks"](https://www.biorxiv.org/content/10.1101/2023.09.28.559916v3). and [(2) "Pre-trained Maldi Transformers improve MALDI-TOF MS-based predictions"](google.com).
The following will assume the `maldi-nn` python package has been installed.

## (1) An antimicrobial drug recommender system using MALDI-TOF MS and dual-branch neural networks

### Data

Our scripts require DRIAMS to be downloaded and processed to HDF5 format using [h5torch](https://github.com/gdewael/h5torch).
If you don't want to run our code and are just after the training-validation-test splits we used, they can be found [here](https://github.com/gdewael/maldi-nn/blob/main/maldi_nn/utils/driams_split.json)

To download DRIAMS: in a separate folder (which we will call `DRIAMS_ROOT`), download DRIAMS-A, -B, -C, and -D from its [download page](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q).
Unpack (`tar -zvxf ...`) all folders in `DRIAMS_ROOT` such that it looks like this:

<details><summary>DRIAMS folder structure</summary>

```
DRIAMS_ROOT/
├── DRIAMS-A
│   ├── id
│   │   ├── 2015
│   │   ├── 2016
│   │   ├── 2017
│   │   └── 2018
│   ├── raw
│   │   ├── 2015
│   │   ├── 2016
│   │   ├── 2017
│   │   └── 2018
│   ├─
...
├── DRIAMS-B
│   ├── binned_6000
│   │   └── 2018
│   ├── id
│   │   └── 2018
│   ├── preprocessed
│   │   └── 2018
│   └── raw
│       └── 2018
├── DRIAMS-C
│   ├─
...
└── DRIAMS-D
│   ├─
...

```

</details>

Then, in the terminal, run: `process_DRIAMS /path/to/DRIAMS_ROOT/ /path/to/DRIAMS_ROOT/amrraw.h5 /path/to/DRIAMS_ROOT/spectraraw.h5 /path/to/DRIAMS_ROOT/amrbin.h5 /path/to/DRIAMS_ROOT/spectrabin.h5 /path/to/DRIAMS_ROOT/amrpks.h5 /path/to/DRIAMS_ROOT/spectrapks.h5`

This command will create 6 new h5torch files. 3 files containing all spectra in DRIAMS, and 3 files containing the spectra with drug resistance information.
Each type of file will be created in 3 formats: Raw, binned and peaks. "Raw" are the pure spectra as profiled by the MALDI-TOF machine. "Binned" is preprocessed as in the original DRIAMS publication and ours and then binned in 6000-dimensional feature vectors. Finally, "peaks" is preprocessed the same way, but instead of binning running the topological peak filtering algorithm on the preprocessed data.

### Train AMR models

We provide convenient access to AMR predictions models with the `train_amr` terminal command.

```bash
train_amr /path/to/DRIAMS_ROOT/amrbin.h5 logs/ onehot mlp M --lr 0.0005 --devices [0]
```

Upon completion of the script, tensorboard logs will be present in the `logs/` folder. Additionally, two lines will be written in a `logs/res.txt` file. The first line will contain the validation performance metrics (micro ROC-AUC, macro ROC-AUC, instance-wise ROC-AUC and Precision@1 of the negative class, in that order), and the second line will contain the test metrics.
Note that this script will only always train, validate, and test on the DRIAMS-A splits we have used.

<details><summary>train_AMR flags</summary>

```
train_amr --help

usage: train_amr [-h] [--lr float] [--logging_file str] [--num_workers int] [--devices literal_eval] [--trf_n_peaks int] [--trf_ckpt_path str]
                 [--trf_ckpt_modeltype {vanilla,negpeaksampler,intensitymlm,onlyclf,onlyshf}]
                 path logs_path drug_embedder spectrum_embedder spectrum_embedder_size

Training script for dual-branch AMR recommender.

positional arguments:
  path                  path to h5torch file.
  logs_path             path to logs.
  drug_embedder         Which drug embedder to use, choices: {ecfp, onehot, gru, cnn, trf, img, kernel}
  spectrum_embedder     Which spectrum embedder to use, choices: {trf, mlp}
  spectrum_embedder_size
                        Which size to use for spectrum embedder, choices: {S, M, L, XL, Linear}. Linear is only available for mlp.

options:
  -h, --help            show this help message and exit
  --lr float            Learning rate. (default: 0.0005)
  --logging_file str    Which file to write final performances to. (default: res.txt)
  --num_workers int     Number of workers in dataloader. Reduce to alleviate CPU. (default: 4)
  --devices literal_eval
                        devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus.
                        (default: 1)
  --trf_n_peaks int     Number of peaks for transformer-peak-based models (default: 200)
  --trf_ckpt_path str   Checkpoint path of malditransformer (default: None)
  --trf_ckpt_modeltype {vanilla,negpeaksampler,intensitymlm,onlyclf,onlyshf}
                        Maldi Transformer pre-trained modeltype. choices: {vanilla, negpeaksampler, intensitymlm, onlyclf, onlyshf} (default:
                        vanilla)
```
</details>

### Fine-tune on new hospitals

To reproduce Section 4.3 from our paper, `maldi-nn` comes with a `reproduce_maldi_finetune_hosp` terminal command, e.g.:

```bash
reproduce_maldi_finetune_hosp /path/to/DRIAMS_ROOT/amrbin.h5 logs/ ecfp M B 0.10 --ckpt_path logs/.../model.ckpt
```

This takes as input a pre-trained AMR prediction model obtained via `train_amr`, and fine-tunes it for prediction on hospital B, C or D.
The `--ckpt_path` flag is optional, which means it can also train a model on these hospitals from scratch.

<details><summary>reproduce_maldi_finetune_hosp flags</summary>

```
reproduce_maldi_finetune_hosp --help

usage: reproduce_maldi_finetune_hosp [-h] [--ckpt_path str] [--lr float] [--logging_file str] [--num_workers int] [--devices literal_eval]
                                     path logs_path drug_embedder spectrum_embedder hospital percent

Training script for dual-branch AMR recommender fine-tuning on other DRIAMS hospitals.

positional arguments:
  path                  path to h5torch file.
  logs_path             path to logs.
  drug_embedder         Which drug embedder to use, choices: {ecfp, onehot, gru, cnn, trf, img, kernel} Ignored if a ckpt_path is given.
  spectrum_embedder     Which size spectrum embedder to use, choices: {S, M, L, XL, Linear} Ignored if a ckpt_path is given
  hospital              Which DRIAMS hospital to fine-tune on, choices: {B, C, D}
  percent               Percentage of training data to use (100percent means 1000 samples)

options:
  -h, --help            show this help message and exit
  --ckpt_path str       Checkpoint from which to start training (default: None)
  --lr float            Learning rate, Ignored if a ckpt_path is given. (default: 0.0005)
  --logging_file str    Which file to write final performances to. (default: res.txt)
  --num_workers int     Number of workers in dataloader. Reduce to alleviate CPU. (default: 4)
  --devices literal_eval
                        devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus.
                        (default: 1)
```

</details>

### Non-recommender AMR baselines

To train the non-recommender AMR baselines from Section 4.2 in our paper. Use the `reproduce_amr_baseline` terminal command

<details><summary>reproduce_amr_baseline flags</summary>

```
usage: reproduce_amr_baseline [-h] [--mlp_size {S,M,L,XL,Linear}] [--mlp_devices literal_eval] path outputs.npz modeltype

Training script for non-recommender AMR baselines.

positional arguments:
  path                  path to h5torch file.
  outputs.npz           numpy .npz file to write (test) predictions into.
  modeltype             Which modeltype to use as baseline, choices: {MLP, lr, xgb}

options:
  -h, --help            show this help message and exit
  --mlp_size {S,M,L,XL,Linear}
                        Which size spectrum embedder to use for MLP, choices: {S, M, L, XL, Linear} (default: ['M'])
  --mlp_devices literal_eval
                        devices to use for MLP. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus.
                        (default: 1)
```

</details>

**Warning**, if MLP is chosen as modeltype, the script will create a huge dump of models inside `./lightning_logs/`. After running the scripts, this folder can be safely removed.

## (2) Pre-trained Maldi Transformers improve MALDI-TOF MS-based predictions

### Data

- The results in this study similarly use the DRIAMS database. Refer to the Data section of the previous study in this README for instructions on how to download DRIAMS.
- RKI is publicly available, see below for downloading instructions.
- LM-UGent is a private database of the Laboratory of Microbiology at Ghent University.

For the **RKI** database: in a separate folder (which we will call `RKI_ROOT`), download RKI from its [download page](https://zenodo.org/records/7702375) (the zip file).
Unzip the file in `RKI_ROOT` such that it looks like this:


<details><summary>RKI_ROOT folder structure</summary>

```
RKI_ROOT/
├── Achromobacter
│   └── Achromobacter xylosoxidans
│       └── ...
├── Acinetobacter
│   ├── Acinetobacter baumannii
│       └── ...
│   ├── Acinetobacter lwoffii
│       └── ...
│   └── Achromobacter pittii
│       └── ...
```

</details>

Then, in the terminal, run: `process_RKI /path/to/RKI_ROOT/ /path/to/DRIAMS_ROOT/RKIraw.h5 /path/to/DRIAMS_ROOT/RKIbin.h5 /path/to/DRIAMS_ROOT/RKIpeaks.h5`

This command will create 3 new h5torch files, with different version of the same data: Raw, binned and peaks. "Raw" are the pure spectra as profiled by the MALDI-TOF machine. "Binned" is preprocessed as described in our study and then binned in 6000-dimensional feature vectors. Finally, "peaks" is preprocessed the same way, but instead of binning running the topological peak filtering algorithm on the preprocessed data.

### Train Maldi Transformer

To train Maldi Transformer: `train_malditransformer /path/to/DRIAMS_ROOT/spectrapks.h5 logs/ M --devices [0,1]`

<details><summary>train_malditransformer flags</summary>

```
usage: train_malditransformer [-h] [--mode {vanilla,negpeaksampler,intensitymlm,onlyclf,onlyshf}] [--n_peaks int] [--p float] [--lmbda float]
                              [--lmbda2 float] [--steps float] [--batch_size int] [--lr float] [--num_workers int] [--devices literal_eval]
                              path logs_path spectrum_embedder

Training script for Maldi Transformer.

positional arguments:
  path                  path to h5torch file.
  logs_path             path to logs.
  spectrum_embedder     Which size spectrum embedder to use

options:
  -h, --help            show this help message and exit
  --mode {vanilla,negpeaksampler,intensitymlm,onlyclf,onlyshf}
                        Maldi Transformer training mode, choices: {vanilla, negpeaksampler, intensitymlm, onlyclf, onlyshf} Note that
                        negpeaksampler requires to run reproduce.estimate_peak_distr first. (default: vanilla)
  --n_peaks int         Number of peaks (default: 200)
  --p float             shuffle freq (default: 0.15)
  --lmbda float         Lambda. This is the probability with which to apply the spec id loss per step. (default: 0.01)
  --lmbda2 float        Additionally, fixed multiplier to apply to the spec id loss. (default: 1.0)
  --steps float         steps (default: 500000)
  --batch_size int      batch size per gpu, effective batch size is this value times the number of gpus. (pytorch ddp). (default: 512)
  --lr float            Learning rate. (default: 0.0005)
  --num_workers int     Number of workers in dataloader. Reduce to alleviate CPU. (default: 4)
  --devices literal_eval
                        devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus.
                        (default: 1)
```

</details>

Note that this script also provides the functionality to train the ablation models in our study.

To fine-tune a Maldi Transformer model on AMR prediction, refer to the `train_amr` script explained above.

### Species identification

To train MLP or Transformer models for species identification, use `train_clf`.

For MLP models, e.g.: `train_clf /path/to/RKI_ROOT/RKIbin.h5 logs/ mlp M --devices [0] --lr 0.0005`

For Transformer models, e.g.: `train_clf /path/to/RKI_ROOT/RKIpks.h5 logs/ trf M --devices [0] --lr 0.0005`

Or for a pre-trained Maldi Transformer model, e.g.: `train_clf /path/to/RKI_ROOT/RKIpks.h5 logs/ trf M --devices [0] --lr 0.0005 --trf_ckpt_path /path/to/model.ckpt`

<details><summary>train_clf flags</summary>

```
usage: train_clf [-h] [--lr float] [--logging_file str] [--num_workers int] [--devices literal_eval] [--trf_n_peaks int] [--trf_ckpt_path str]
                 [--trf_ckpt_modeltype {vanilla,negpeaksampler,intensitymlm,onlyclf,onlyshf}] [--trf_transfer_output_head boolean]
                 path logs_path spectrum_embedder size

Training script for species identification.

positional arguments:
  path                  path to h5torch file.
  logs_path             path to logs.
  spectrum_embedder     Which spectrum embedder to use, choices: {mlp, trf}
  size                  Model size, choices: {Linear, S, M, L, XL}. Linear is only available for mlp.

options:
  -h, --help            show this help message and exit
  --lr float            Learning rate. (default: 0.0005)
  --logging_file str    Which file to write final performances to. (default: res.txt)
  --num_workers int     Number of workers in dataloader. Reduce to alleviate CPU. (default: 4)
  --devices literal_eval
                        devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus.
                        (default: 1)
  --trf_n_peaks int     Number of peaks (default: 200)
  --trf_ckpt_path str   Checkpoint path of malditransformer (default: None)
  --trf_ckpt_modeltype {vanilla,negpeaksampler,intensitymlm,onlyclf,onlyshf}
                        Maldi Transformer pre-trained modeltype. choices: {vanilla, negpeaksampler, intensitymlm, onlyclf, onlyshf} (default:
                        vanilla)
  --trf_transfer_output_head boolean
                        Whether to transfer the clf output head of the Maldi Transformer. Can be set to true if domain adaptation was adopted.
                        (default: False)
```

</details>

### Species identification baselines

In Section 3.1 of our study, we compare the performance of Maldi Transformer against several non-neural network baselines. You can re-run these analyses via the command `reproduce_clf_baseline`,

E.g.: `reproduce_clf_baseline /path/to/RKI_ROOT/RKIbin.h5 knn_outputs.npz knn`

<details><summary>reproduce_clf_baseline flags</summary>

```
usage: reproduce_clf_baseline [-h] path outputs.npz modeltype

Training script for species identification baselines. Returns an npz file with predictions for the test set.

positional arguments:
  path         path to h5torch file.
  outputs.npz  numpy .npz file to write (test) predictions into.
  modeltype    Which modeltype to use as baseline, choices: {knn, lr, rf}

options:
  -h, --help   show this help message and exit
```

</details>

### Domain adaptation

Maldi Transformers are pre-trained on e.g. DRIAMS. When the target (fine-tuning) data domain consists of wildly different species, the performance is hindered. For this reason, we can additionally pre-train on the training set of the fine-tuning task before supervised fine-tuning. For this reason, we include the terminal command `malditransformer_domain_adapt`:

<details><summary>malditransformer_domain_adapt flags</summary>

```
usage: malditransformer_domain_adapt [-h] [--ckpt_path str] [--n_peaks int] [--p float] [--lmbda float] [--lmbda2 float] [--steps float]
                                     [--batch_size int] [--lr float] [--num_workers int] [--devices literal_eval]
                                     path logs_path spectrum_embedder

Training script for domain adaptation of a Maldi Transformer.

positional arguments:
  path                  path to h5torch file.
  logs_path             path to logs.
  spectrum_embedder     Which size spectrum embedder to use

options:
  -h, --help            show this help message and exit
  --ckpt_path str       Ckpt path (default: None)
  --n_peaks int         Number of peaks (default: 200)
  --p float             shuffle freq (default: 0.5)
  --lmbda float         Lambda. This is the probability with which to apply the spec id loss per step. (default: 0.01)
  --lmbda2 float        Additionally, fixed multiplier to apply to the spec id loss. (default: 1.0)
  --steps float         steps (default: 20000)
  --batch_size int      batch size (default: 512)
  --lr float            Learning rate. (default: 0.0005)
  --num_workers int     Number of workers in dataloader. Reduce to alleviate CPU. (default: 4)
  --devices literal_eval
```

</details>

