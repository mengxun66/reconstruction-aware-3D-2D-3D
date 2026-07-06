# Evaluation

This folder contains the evaluation scripts and derived results used in the manuscript.

The evaluation is organized into three parts:

## 1. Checkpoint diagnostics

`checkpoint_diagnostics/` contains the script used to evaluate saved LoRA checkpoints using:

- EMA-smoothed training-loss plateau detection;
- KID evaluation across checkpoints;
- fixed case-specific generation seeds;
- repeated random subsets of 16 cases from a fixed 20-case diagnostic set.

Checkpoint selection follows a two-stage rule. The loss curve is first used to identify the plateau-qualified interval, after which the checkpoint with the lowest mean KID within that interval is selected. Epoch 16 was selected for the low-rise LoRA and epoch 20 for the high-rise LoRA.

## 2. Image-domain evaluation

`image_domain/` contains scripts and derived outputs for:

- KID comparison between D2 reference images and generated images;
- comparison with Plain Stable Diffusion;
- simplified-prompt ablation;
- DINOv2 feature extraction;
- self-organizing map analysis;
- cluster retention, transition, and occupancy analysis.

The SOM reference map is trained only on the designated reference feature set, while comparison images are projected onto the fixed map.

## 3. Morphological recovery

`morphological_recovery/` contains scripts and derived results for comparing the original morphological parameters (D1) with the reconstructed parameters (D4).

The analyses include:

- normalized mean absolute error;
- Lin’s concordance correlation coefficient;
- normalized Wasserstein distance;
- paired Wilcoxon signed-rank tests;
- matched-pairs rank-biserial correlation;
- pair-preserving bootstrap confidence intervals;
- held-out test evaluation.

## Reproducibility

Input paths may need to be updated before running the scripts. Software versions are listed in the repository `environment/` folder. Case identifiers, prompts, dataset partitions, and generation seeds are provided in the corresponding metadata files.
