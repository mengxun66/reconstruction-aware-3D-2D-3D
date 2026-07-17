# A Reconstruction-Based 3D–2D–3D Workflow for Assessing the Geometric Recoverability of Diffusion-Generated Building Models

This repository provides reproducibility materials for the manuscript:

**Diffusion-Based Building Generation with Geometric Recoverability Assessment: A 3D–2D–3D Reconstruction Workflow**

The study develops a reconstruction-based workflow that connects:

1. simplified 3D building-form models;
2. stratified RGB height representations;
3. LoRA-adapted Stable Diffusion generation;
4. RGB-to-3D inverse reconstruction; and
5. image-domain and geometric-parameter evaluation.

## Repository contents

```text
reconstruction-based-3d-2d-3d/
│
├── grasshopper/
│   ├── forward_rgb_encoding.gh
│   └── inverse_rgb_decoding.gh
│
├── metadata/
│   ├── Data S1_140_case_metadata.xlsx
│   ├── Data S2_140_case_Geometric_parameters.xlsx
│   └── Data S3_140_case prompts and generation seeds.xlsx
│
├── evaluation/
│   ├── checkpoint_diagnostics/
│   ├── image_domain/
│   └── geometric_recovery/
│
├── models/
│   ├── README.md
│   └── model_card.md
│
├── data/
│   ├── D2_internal_100/
│   ├── D3_internal_100/
│   ├── D2_heldout_40/
│   └── D3_heldout_40/
│
├── examples/
│   ├── input_3d_models/
│   └── encoded_rgb_images/
│
├── environment/
│   └── software_versions.md
│
├── LICENSE
└── README.md
```

The repository may be updated as the manuscript and supplementary materials are finalized.

## Workflow overview

The computational workflow contains four connected datasets:

D1: geometric parameters extracted from the original 3D building-form models;
D2: stratified RGB height representations encoded from the original 3D models;
D3: RGB height representations generated using LoRA-adapted Stable Diffusion;
D4: geometric parameters extracted after RGB-to-3D inverse reconstruction.

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

The generated RGB images are decoded channel by channel using the inverse height-mapping equations. The reconstructed channel-specific geometries are then combined to form the complete 3D building-form model.

The complete executable definition is provided in:

`grasshopper/inverse_rgb_decoding.gh`

Detailed information on channel thresholds, spatial scaling, geometric tolerances, software versions, and required plug-ins is provided in the corresponding documentation files.

## Geometric parameters

The evaluation uses nine geometric parameters: MaxH, AvgH, HeightSD, TPI, FAR, BCR, LEI, CPR, and VCI. Definitions, units, and extraction procedures follow the manuscript and are tabulated in:

`metadata/Data S2_140_case_Geometric_parameters.xlsx`

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
- reference geometric parameters.

Original architectural drawings and other copyrighted source materials are not redistributed unless their licenses permit public release.

## Software environment

The workflow was developed using Rhino and Grasshopper. Exact software versions, document units, geometric tolerances, and required Grasshopper plug-ins are documented in:

`environment/software_versions.md`

## Reproducibility

To reproduce the main workflow:

open an example 3D building-form model in Rhino;
run the forward RGB-encoding Grasshopper definition;
generate or load the corresponding RGB height representation;
run the inverse RGB-decoding definition;
extract the nine geometric parameters; and
run the supplied image-domain and geometric-recovery evaluation scripts.

A complete worked example is provided in the `examples/` directory where licensing permits.

## Data availability

Derived data, prompts, seeds, dataset assignments, evaluation scripts, and Grasshopper definitions are provided in this repository where licensing permits.

Materials that cannot be publicly redistributed because of copyright or third-party restrictions may be made available by the corresponding author upon reasonable request.

## Citation

Liu, M., Zheng, H., Huang, Z., Tan, B., Zhu, Y., and Li, Z.
Diffusion-Based Building Generation with Geometric Recoverability Assessment: A 3D–2D–3D Reconstruction Workflow
Manuscript in preparation.

## License

The license for the released code and Grasshopper definitions will be specified in the LICENSE file. Third-party software, pretrained models, source images, and architectural drawings remain subject to their original licenses.

## Contact

For questions regarding the repository or manuscript, please contact:

Mengxun Liu: 2210266@tongji.edu.cn
