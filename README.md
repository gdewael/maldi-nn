# maldi-nn
Deep learning tools and models for MALDI-TOF spectra analysis.
Contains all code and scripts to reproduce: ["An antimicrobial drug recommender system using MALDI-TOF MS and dual-branch neural networks"](https://www.biorxiv.org/content/10.1101/2023.09.28.559916v3), and "Pre-trained Maldi Transformers improve MALDI-TOF MS-based prediction" (in draft).

Package features:
- Reading and preprocessing functions for MALDI-TOF MS spectra.
- Model definitions to process SMILES strings with state-of-the-art techniques (for feature-based AMR prediction).
- Model definitions to pre-train state-of-the-art Transformer networks on MALDI-TOF MS data
- Model definitions and scripts to train AMR models on the [DRIAMS database](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q).
- Model definitions and scripts to train species identification models.

## Install

`maldi-nn` is distributed on PyPI.
```bash
pip install maldi-nn
```

You may need to [install PyTorch](https://pytorch.org/get-started/locally/) before running this command in order to ensure the right CUDA kernels for your system are installed

## Academic Reproducibility

All information regarding reproducing our results can be found in the [reproduce folder README](https://github.com/gdewael/maldi-nn/tree/main/maldi_nn/reproduce)

## Credits
- Implementations of many MALDI reading and processing functions were based on the R package [MaldiQuant](https://github.com/sgibb/MALDIquant).
- Topological Peak Filtering was taken from the [Topf package](https://github.com/BorgwardtLab/Topf).

## Citation
```
@article{de2023antimicrobial,
  title={An antimicrobial drug recommender system using MALDI-TOF MS and dual-branch neural networks},
  author={De Waele, Gaetan and Menschaert, Gerben and Waegeman, Willem},
  journal={bioRxiv},
  pages={2023--09},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
