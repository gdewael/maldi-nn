# maldi-nn
Deep learning tools and models for MALDI-TOF mass spectra analysis.

Package features:
- Reading and preprocessing functions for MALDI-TOF MS spectra.
- Model definitions to process SMILES strings with state-of-the-art techniques (for feature-based AMR prediction).
- Model definitions to pre-train state-of-the-art Transformer networks on MALDI-TOF MS data
- Model definitions and scripts to train AMR models on the [DRIAMS database](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q).
- Model definitions and scripts to train species identification models.

#### Minor note:
As of August 2025, the [DRIAMS database](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q) has updated some file names in DRIAMS-C, breaking some functionalities of this repository.
When using `maldi-nn` with DRIAMS, please use the DRIAMS release dating from November 11, 2021.

## Install

`maldi-nn` is distributed on PyPI.
```bash
pip install maldi-nn
```

In case this package loses backward-compatibility with more-recent versions of PyTorch and PyTorch Lightning: the code has been tested with `torch==2.0.1` and `pytorch-lightning==2.0.9`. If you encounter errors with these packages, try running the code using these versions.

You may need to [install PyTorch](https://pytorch.org/get-started/locally/) before running this command in order to ensure the right CUDA kernels for your system are installed


## Quick start

We have uploaded some example spectra from the RKI database and our pre-trained models in the `assets` folder.

To quickly start playing around with our models, follow:

In bash:
```bash
pip install maldi-nn
git clone https://github.com/gdewael/maldi-nn.git
```

In Python:
```python
from maldi_nn.spectrum import *
from maldi_nn.models import MaldiTransformer
import torch
spectrum = SpectrumObject.from_bruker(
    "./maldi-nn/assets/RKI_example/Bacillus_anthracis/acqu",
    "./maldi-nn/assets/RKI_example/Bacillus_anthracis/fid"
    )

preprocessor = SequentialPreprocessor(
    VarStabilizer(method="sqrt"),
    Smoother(halfwindow=10),
    BaselineCorrecter(method="SNIP", snip_n_iter=20),
    Trimmer(),
    PersistenceTransformer(extract_nonzero=True),
    Normalizer(sum=1),
    PeakFilter(max_number=200),
)

spectrum_preprocessed = preprocessor(spectrum)
spectrum_tensors = spectrum_preprocessed.torch()

model = MaldiTransformer.load_from_checkpoint("../../maldi-nn/assets/MaldiTransformerM.ckpt").eval().cpu()

mlm_logits, spectrum_embedding = model(spectrum_tensors)

prob_noise_peak = torch.sigmoid(mlm_logits)
```

Other Maldi Transformer model sizes are available at https://huggingface.co/gdewael/MaldiTransformer/tree/main.

## Academic Reproducibility

This package contains all code and scripts to reproduce: ["An antimicrobial drug recommender system using MALDI-TOF MS and dual-branch neural networks"](https://doi.org/10.7554/eLife.93242.1), and ["Pre-trained Maldi Transformers improve MALDI-TOF MS-based prediction"](https://www.biorxiv.org/content/10.1101/2024.01.18.576189v1).
All information regarding reproducing our results can be found in the [reproduce folder README](https://github.com/gdewael/maldi-nn/tree/main/maldi_nn/reproduce)

## Credits
- Implementations of many MALDI reading and processing functions were based on the R package [MaldiQuant](https://github.com/sgibb/MALDIquant).
- Topological Peak Filtering was taken from the [Topf package](https://github.com/BorgwardtLab/Topf).

## Citation

Antimicrobial drug recommenders:

```
@article{dewaele2024antimicrobial,
  title={An antimicrobial drug recommender system using MALDI-TOF MS and dual-branch neural networks},
  author={De Waele, Gaetan and Menschaert, Gerben and Waegeman, Willem},
  journal={eLife},
  volume={13},
  pages={RP93242},
  year={2024},
  publisher={eLife Sciences Publications Limited}
}


```

Maldi Transformers:

```
@article{dewaele2025pre,
  title={Pre-trained Maldi Transformers improve MALDI-TOF MS-based prediction},
  author={De Waele, Gaetan and Menschaert, Gerben and Vandamme, Peter and Waegeman, Willem},
  journal={Computers in Biology and Medicine},
  volume={186},
  pages={109695},
  year={2025},
  publisher={Elsevier}
}
```
