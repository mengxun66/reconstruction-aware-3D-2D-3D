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
