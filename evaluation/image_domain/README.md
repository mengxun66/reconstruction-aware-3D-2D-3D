# KID evaluation

This folder contains the script used to calculate Kernel Inception Distance (KID) between one reference image set and one or more generated image sets.

## Requirements

```bash
pip install torch torchvision torchmetrics[image] torch-fidelity pillow numpy matplotlib
```

## Usage

```bash
python kid_evaluation.py \
  --reference-dir "PATH/TO/D2" \
  --reference-label "D2" \
  --generated "LoRA-D3=PATH/TO/D3_MAIN" \
  --generated "Simplified prompt=PATH/TO/D3_SIMPLE" \
  --generated "Plain SD=PATH/TO/D3_PLAIN" \
  --expected-n 100 \
  --subsets 50 \
  --subset-size 16 \
  --seed 42 \
  --output-dir "PATH/TO/OUTPUT" \
  --stem "KID_internal_100" \
  --title "Distributional distance to reference D2" \
  --annotate-ratios
```

For the external 40-case test, replace the reference and generated-image paths and set:

```bash
--expected-n 40
```

## Protocol

* InceptionV3 feature dimension: 2048
* KID subsets: 50
* Subset size: 16
* Random seed: 42
* Reported value: KID mean ± subset standard deviation

KID is a set-level metric and does not use case-level image pairing.

## Outputs

```text
<stem>_results.csv
<stem>_image_manifest.csv
<stem>_run_metadata.json
<stem>_bar.png
<stem>_bar.svg
<stem>_bar.pdf
```

