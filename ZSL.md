## ZSL plans

- Datasets: LM-UGent and RKI

- Different models for different scenarios, train separately:
  - 10x Strain hold out
  - 10x Species hold out
  - 10x Genus hold out

- Multi-level accuracy evaluation
  - Species-lvl (optimize for this one in all cases.)
  - Genus-lvl

- Multi-class model:s
  - MLPs from Maldi Transformer (baseline: linear)

- ZSL Model: Dual branch
  - Spectrum side: MLPs from Maldi Transformer (baseline: linear)
  - DNA side: ConvNet (tune a bit using default experiment set)
  - Eval/inference: using all possible strains in dataset as candidate prediction set

- DNA side information data prep
  - SILVA aligned 16s rRNA, 50_000 length sequences.

- Extra experiments:
  - Impact of pre-trained Maldi Transformer
  - Impact of VAE pre-training DNA side
  - Loss function based on evolutionary dist. prediction?

- Open questions:
  - Is there a way for the model to indicate - per spectrum - that it is not sure on e.g. species-lvl, but is certain for genus-lvl?
  - Should one feed all possible side-information classes or can one sample some negative classes during training?

- Questions / sections:
  - 1. What is the ideal model size?
  - 2. What is the zero-shot performance on multiple levels?
  - 3. A novel loss based on evolutionary dist. prediction?
  - 4. Impact of pre-training (Maldi Transformer & DNA VAE)
  - 5. Model analyses (Difficult clades? / Embedding spaces? / Noise data? / Confidence of hold outs on diff lvls?)