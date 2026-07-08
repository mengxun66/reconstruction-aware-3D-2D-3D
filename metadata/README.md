# Metadata

This folder provides the case-level metadata, morphological parameters, prompts, and generation seeds used in this study.

The documented dataset contains **140 cases in total**, organized into two non-overlapping parts:

- **Main100**: 100 internal model-development cases  
  - 52 low-range cases
  - 48 high-range cases
  - Used for LoRA training, prompt construction, clustering fitting, and the main D1–D4 analysis.
  - Checkpoint diagnostic subsets were drawn from this Main100 set.

- **Held-out40**: 40 held-out cases  
  - 20 low-range cases
  - 20 high-range cases
  - Not used for LoRA training, checkpoint diagnostics, or clustering refitting.
  - Used only for the post-selection held-out D1-ext–D4-ext test.

Therefore, the 40 held-out cases are independent from the Main100 model-development set. The checkpoint diagnostic cases are **not** independent validation/test cases; they are fixed diagnostic subsets drawn from Main100.

## Files

### `Data S1_140_case_metadata.xlsx`

This file provides case-level metadata for all 140 documented cases.

It contains:

- case identifiers
- dataset role: Main100 or Held-out40
- height domain: low-range or high-range
- building type
- case source and provenance
- source material type
- selection and exclusion criteria
- experimental use indicators

This file is used to clarify dataset provenance, dataset partitioning, and the experimental role of each case.

### `Data S2_140_case_morphological_parameters.xlsx`

This file provides the nine morphological parameters used in the study.

It contains parameter tables for:

- D1 internal Main100 reference cases
- D4 internal Main100 reconstructed cases
- D1-ext Held-out40 reference cases
- D4-ext Held-out40 reconstructed cases

The nine parameters include:

- Maximum Height
- Average Height
- Height Standard Deviation
- Tower–Podium Index
- Floor Area Ratio
- Building Coverage Ratio
- Linear Extension Index
- Central Plaza Ratio
- Volume Concentration Index

These values support the morphological-parameter recovery analysis reported in the paper.

### `Data S3-140_case prompts and generation seeds.xlsx`

This file provides the case-level prompts and fixed generation seeds for all 140 cases.

It contains:

- complete prompt strings used for generation
- fixed case-specific generation seeds
- prompt components and labels
- Main100 and Held-out40 prompt records

For Main100, the same fixed case-specific seeds were retained across the full method, Plain Stable Diffusion baseline, and simplified-prompt ablation to preserve case-level comparability. Held-out40 used a separately generated and subsequently fixed seed set.

## Dataset roles

| Dataset group | Number of cases | Low-range | High-range | Experimental role |
|---|---:|---:|---:|---|
| Main100 | 100 | 52 | 48 | LoRA training, clustering fitting, main D1–D4 analysis |
| Checkpoint diagnostic subsets | 40 | 20 | 20 | Drawn from Main100; used only for checkpoint diagnostics |
| Held-out40 | 40 | 20 | 20 | Not used in training or checkpoint selection; used only for held-out testing |
| Total documented cases | 140 | 72 | 68 | Complete metadata, parameters, prompts, and seeds documented |

## Important clarification

The **Main100** and **Held-out40** datasets should not be mixed.

- Main100 supports the within-distribution training-reference evaluation.
- Held-out40 supports the post-selection held-out test.
- Checkpoint diagnostic subsets are drawn from Main100 and should not be interpreted as independent validation or test data.

All files are linked by case identifiers.
