# Reconstruction-Aware 3D–2D–3D Workflow for Building Massing Design

This repository provides reproducibility materials for the manuscript:

**From Visual Plausibility to Measurable Form: A 3D–2D–3D Reconstruction-Aware Diffusion Workflow for Building Massing Design**

The study develops a reconstruction-aware workflow that connects:

1. simplified 3D building-massing models;
2. stratified RGB height representations;
3. LoRA-adapted Stable Diffusion generation;
4. RGB-to-3D inverse reconstruction; and
5. image-domain and morphological-parameter evaluation.

## Repository contents

```text
reconstruction-aware-3D-2D-3D/
│
├── grasshopper/
│   ├── forward_rgb_encoding.gh
│   ├── inverse_rgb_decoding.gh
│
├── metadata/
│   ├── dataset_split.csv
│   ├── case_metadata.csv
│   ├── case_prompts.csv
│   └── case_generation_seeds.csv
│
├── evaluation/
│   ├── checkpoint_diagnostics/
│   ├── image_domain/
│   └── morphological_recovery/
│
├── models/
│   ├── README.md
│   └── model_card.md
│
├── examples/
│   ├── input_3d_models/
│   ├── encoded_rgb_images/
│   └── reconstructed_3d_models/
│
├── supplementary/
│   ├── supplementary_tables/
│   └── supplementary_figures/
│
├── environment/
│   ├── software_versions.md
│   └── grasshopper_plugins.md
│
├── LICENSE
└── README.md
```
The repository may be updated as the manuscript and supplementary materials are finalized.

## Workflow overview

The computational workflow contains four connected datasets:

- **D1:** morphological parameters extracted from the original 3D massing models;
- **D2:** stratified RGB height representations encoded from the original models;
- **D3:** RGB height representations generated using LoRA-adapted Stable Diffusion;
- **D4:** morphological parameters extracted after RGB-to-3D inverse reconstruction.

The main development set contains 100 cases, including 52 low-rise and 48 high-rise cases. A separate 40-case held-out set is used after model and checkpoint selection.

## RGB height encoding

Two height-domain-specific representations are used:

| Encoding domain | Height range | RGB stratum boundaries |
|---|---:|---|
| Low-range encoding | 0–24 m | 0, 8, 16, and 24 m |
| High-range encoding | 0–60 m | 0, 20, 40, and 60 m |

The red, green, and blue channels represent different vertical height strata. The complete forward-encoding definition is provided in:

`grasshopper/forward_rgb_encoding.gh`
## RGB-to-3D reconstruction

The generated RGB images are decoded channel by channel using the inverse height-mapping equations. The reconstructed channel-specific geometries are then combined to form the complete 3D massing model.

The complete executable definition is provided in:

`grasshopper/inverse_rgb_decoding.gh`

Detailed information on channel thresholds, spatial scaling, geometric tolerances, software versions, and required plug-ins is provided in the corresponding documentation files.
## Morphological parameters
## Dataset metadata
Case-level metadata are provided using anonymized case identifiers. The released metadata include, where applicable:

- dataset assignment;
- building category;
- source-data type;
- height domain;
- layout label;
- diagnostic-subset assignment;
- structured prompt;
- generation seed; and
- reference morphological parameters.

Original architectural drawings and other copyrighted source materials are not redistributed unless their licenses permit public release.
## Software environment
The workflow was developed using Rhino and Grasshopper. Exact software versions, document units, geometric tolerances, and required Grasshopper plug-ins are documented in:
`environment/software_versions.md`
`environment/grasshopper_plugins.md`
## Reproducibility
To reproduce the main workflow:

open the example 3D massing model in Rhino;
run the forward RGB-encoding Grasshopper definition;
generate or load the corresponding RGB height representation;
run the inverse RGB-decoding definition;
extract the nine morphological parameters; and
run the supplied image-domain and morphological-recovery evaluation scripts.

A complete worked example will be provided in the `examples/` directory.
## Data availability
Derived data, prompts, seeds, dataset assignments, evaluation scripts, and Grasshopper definitions are provided in this repository where licensing permits.

Materials that cannot be publicly redistributed because of copyright or third-party restrictions may be made available by the corresponding author upon reasonable request.
## Citation
Liu, M., Zheng, H., Huang, Z., Tan, B., Zhu, Y., and Li, Z.
From Visual Plausibility to Measurable Form:
A 3D–2D–3D Reconstruction-Aware Diffusion Workflow for Building Massing Design.
Manuscript in preparation.
## License
The license for the released code and Grasshopper definitions will be specified in the LICENSE file. Third-party software, pretrained models, source images, and architectural drawings remain subject to their original licenses.
## Contact
For questions regarding the repository or manuscript, please contact:

Mengxun Liu: mengxunliu@outlook.com
