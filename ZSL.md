## ZSL plans

- Datasets: LM-UGent and RKI

- 10x independent split with
  - Strain hold out
  - Species hold out
  - Genus hold out
  - Family hold out

- Multi-level accuracy evaluation
  - Species-lvl
  - Genus-lvl
  - Family-lvl

- Multi-class model:
  - MLPs from Maldi Transformer (baseline: linear)

- ZSL Model: Dual branch
  - Spectrum side: MLPs from Maldi Transformer (baseline: linear)
  - DNA side: ConvNet (tune a bit using default experiment set)
  - Eval/inference: using all possible strains as candidate prediction set

- DNA side information data prep
  - SILVA
  - Algnmt?

- Extra experiments:
  - Impact of pre-trained Maldi Transformer
  - Impact of VAE pre-training DNA side
  - Loss function based on evolutionary dist. prediction?

- Questions / sections:
  - 1. What is the ideal model size?
  - 2. What is the zero-shot performance on multiple levels?
  - 3. A novel loss based on evolutionary dist. prediction?
  - 4. Impact of pre-training (Maldi Transformer & DNA VAE)
  - 5. Model analyses (Difficult clades? / Embedding spaces? / Noise data?)