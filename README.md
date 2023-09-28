# maldi-nn
Deep learning tools and models for MALDI-TOF spectra analysis.

Features:
- Reading and preprocessing functions for MALDI-TOF spectra.
- Model definitions to process SMILES strings with state-of-the-art techniques.
- Model definitions and scripts to train AMR models on the DRIAMS database.

## Install

`maldi-nn` is distributed on PyPI.
```bash
pip install maldi-nn
```

You may need to [install PyTorch](https://pytorch.org/get-started/locally/) before running this command in order to ensure the right CUDA kernels for your system are installed

## Academic Reproducibility

This package contains all code to reproduce the results in ["Paper name"](www.google.com).
The following will assume the `maldi-nn` python package has been installed.

### DRIAMS data download instructions

To run any of the following, we expect the DRIAMS dataset to be downloaded and processed to HDF5 format using [h5torch](https://github.com/gdewael/h5torch).
If you don't want to run our code and are just after the training-validation-test splits we used, they can be found [here](https://github.com/gdewael/maldi-nn/blob/main/maldi_nn/utils/driams_split.json)

To download DRIAMS: in a separate folder (which we will call DRIAMS_ROOT), download DRIAMS-A, -B, -C, and -D from its [download page](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q).
Unpack (`tar -zvxf ...`) all folders in DRIAMS_ROOT such that it looks like this:
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
Then, in the terminal, run: `process_DRIAMS /path/to/DRIAMS_ROOT/ /path/to/DRIAMS_ROOT/amrraw.h5 /path/to/DRIAMS_ROOT/spectraraw.h5 /path/to/DRIAMS_ROOT/amrbin.h5 /path/to/DRIAMS_ROOT/spectrabin.h5 /path/to/DRIAMS_ROOT/amrpks.h5 /path/to/DRIAMS_ROOT/spectrapks.h5`

This command will create 6 new h5torch files. 3 files containing all spectra in DRIAMS, and 3 files containing the spectra with drug resistance information.
Each type of file will be created in 3 formats: Raw, binned and peaks. Raw are the pure spectra as profiled by the MALDI-TOF machine. Binned is preprocessed as in the original DRIAMS publication and ours and then binned in 6000-dimensional feature vectors. Finally, peaks is preprocessed the same way, but instead of binning running the topological peak filtering algorithm on the preprocessed data.

### Train AMR models

We provide convenient access to our models with the train_amr function, which is available as a terminal command after installation of maldi-nn.

```bash
train_amr /path/to/DRIAMS_ROOT/amrbin.h5 logs/ onehot M --lr 0.0005 --devices [0]
```

Upon completion of the script, tensorboard logs will be present in the `logs/` folder. Additionally, two lines will be written in a `logs/res.txt` file. The first line will contain the validation performance metrics (micro ROC-AUC, macro ROC-AUC, instance-wise ROC-AUC and Precision@1 of the negative class, in that order), and the second line will contain the test metrics.
Note that this script will only always train, validate, and test on the DRIAMS-A splits we have used.


<details><summary>train_AMR flags</summary>

```
train_amr --help

usage: train_amr [-h] [--lr float] [--logging_file str] [--num_workers int] [--devices literal_eval] path logs_path drug_embedder spectrum_embedder

Training script for dual-branch AMR recommender.

positional arguments:
  path                  path to h5torch file.
  logs_path             path to logs.
  drug_embedder         Which drug embedder to use, choices {ecfp, onehot, gru, cnn, trf, img, kernel}
  spectrum_embedder     Which size spectrum embedder to use, choices {S, M, L, XL, Linear}

options:
  -h, --help            show this help message and exit
  --lr float            Learning rate. (default: 0.0005)
  --logging_file str    Which file to write final performances to. (default: res.txt)
  --num_workers int     Number of workers in dataloader. Reduce to alleviate CPU. (default: 4)
  --devices literal_eval
                        devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus. (default: 1)
```
</details>

### Fine-tune on new hospitals

`maldi-nn` comes with a `reproduce_maldi_finetune_hosp` terminal command, e.g.:

```bash
reproduce_maldi_finetune_hosp /path/to/DRIAMS_ROOT/amrbin.h5 logs/ ecfp M B 0.10 --ckpt_path logs/.../model.ckpt
```

<details><summary>reproduce_maldi_finetune_hosp flags</summary>

```
reproduce_maldi_finetune_hosp --help

usage: reproduce_maldi_finetune_hosp [-h] [--ckpt_path str] [--lr float] [--logging_file str] [--num_workers int] [--devices literal_eval] path logs_path drug_embedder spectrum_embedder hospital percent

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
                        devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus. (default: 1)
```
</details>

### Non-recommender baselines

`maldi-nn` comes with a commands for each of the tested non-recommender baselines, e.g.:

```bash
reproduce_maldi_baselines_lr /path/to/DRIAMS_ROOT/amrbin.h5 predictions_lr.npz
reproduce_maldi_baselines_xgb /path/to/DRIAMS_ROOT/amrbin.h5 predictions_xgb.npz
reproduce_maldi_baselines_MLP /path/to/DRIAMS_ROOT/amrbin.h5 predictions_S.npz S --devices [0]
reproduce_maldi_baselines_MLP /path/to/DRIAMS_ROOT/amrbin.h5 predictions_M.npz M --devices [0]
reproduce_maldi_baselines_MLP /path/to/DRIAMS_ROOT/amrbin.h5 predictions_L.npz L --devices [0]
reproduce_maldi_baselines_MLP /path/to/DRIAMS_ROOT/amrbin.h5 predictions_XL.npz XL --devices [0]
```

<details><summary>reproduce_maldi_baselines_lr flags</summary>

```
reproduce_maldi_baselines_lr --help

usage: reproduce_maldi_baselines_lr [-h] path outputs.npz

Training script for non-recommender logistic regression baselines.

positional arguments:
  path         path to h5torch file.
  outputs.npz  numpy .npz file to write (test) predictions into.

options:
  -h, --help   show this help message and exit
```
</details>

<details><summary>reproduce_maldi_baselines_xgb flags</summary>

```
reproduce_maldi_baselines_xgb --help

usage: reproduce_maldi_baselines_xgb [-h] path outputs.npz

Training script for non-recommender xgboost baselines.

positional arguments:
  path         path to h5torch file.
  outputs.npz  numpy .npz file to write (test) predictions into.

options:
  -h, --help   show this help message and exit
```
</details>

<details><summary>reproduce_maldi_baselines_MLP flags</summary>

```
reproduce_maldi_baselines_MLP --help

usage: reproduce_maldi_baselines_MLP [-h] [--devices literal_eval] path outputs.npz size

Training script for non-recommender MLP baselines.

positional arguments:
  path                  path to h5torch file.
  outputs.npz           numpy .npz file to write (test) predictions into.
  size                  Which size spectrum embedder to use, choices: {S, M, L, XL, Linear}

options:
  -h, --help            show this help message and exit
  --devices literal_eval
                        devices to use. Input an integer to specify a number of gpus or a list e.g. [1] or [0,1,3] to specify which gpus. (default: 1)
```
</details>

**Warning**, the baselinescript_MLP.py scripts will create a huge dump of models inside `./lightning_logs/`. After running the scripts, this folder can be safely removed.


## Credits
- Implementations of many MALDI reading and processing functions were based on the R package [MaldiQuant](https://github.com/sgibb/MALDIquant).
- Topological Peak Filtering was taken from the [Topf package](https://github.com/BorgwardtLab/Topf).