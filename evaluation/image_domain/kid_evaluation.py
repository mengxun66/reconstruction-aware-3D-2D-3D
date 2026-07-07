#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Kernel Inception Distance (KID) evaluation for multiple image sets.

The script compares one reference image set with one or more generated image
sets using TorchMetrics' KernelInceptionDistance. All generated sets in a run
are evaluated with the same reference features, KID settings, and random seed.

Important interpretation
------------------------
KID is a set-level distributional metric. It does not use case-to-case pairing.
TorchMetrics independently samples reference and generated features for each
subset. The reported error is the standard deviation across KID subsets, not
a confidence interval.

Dependencies
------------
pip install torch torchvision torchmetrics[image] pillow numpy matplotlib

Example 1: internal 100-case comparison
---------------------------------------
python kid_evaluation.py ^
  --reference-dir "E:/data/D2_internal_100" ^
  --reference-label "D2" ^
  --generated "LoRA-D3=E:/data/D3_main_100" ^
  --generated "Simplified prompt=E:/data/D3_simple_100" ^
  --generated "Plain SD=E:/data/D3_plain_sd_100" ^
  --expected-n 100 ^
  --subsets 50 ^
  --subset-size 16 ^
  --seed 42 ^
  --output-dir "E:/results/kid_internal" ^
  --stem "KID_internal_100" ^
  --title "Distributional distance to reference D2" ^
  --annotate-ratios

Example 2: external 40-case held-out comparison
-----------------------------------------------
python kid_evaluation.py ^
  --reference-dir "E:/data/D2_ext_40" ^
  --reference-label "D2-ext" ^
  --generated "LoRA-D3-ext=E:/data/D3_ext_40" ^
  --expected-n 40 ^
  --subsets 50 ^
  --subset-size 16 ^
  --seed 42 ^
  --output-dir "E:/results/kid_ext40" ^
  --stem "KID_external_40" ^
  --title "External 40-case held-out test"

Outputs
-------
<stem>_results.csv
<stem>_run_metadata.json
<stem>_bar.png
<stem>_bar.svg
<stem>_bar.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchmetrics
from PIL import Image, ImageOps
from torchmetrics.image.kid import KernelInceptionDistance


VALID_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


@dataclass(frozen=True)
class GeneratedSet:
    label: str
    directory: Path


def parse_generated_spec(value: str) -> GeneratedSet:
    """Parse LABEL=PATH supplied to --generated."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--generated must use the form LABEL=PATH"
        )

    label, path_text = value.split("=", 1)
    label = label.strip()
    path_text = path_text.strip().strip('"')

    if not label:
        raise argparse.ArgumentTypeError(
            "The generated-set label cannot be empty."
        )
    if not path_text:
        raise argparse.ArgumentTypeError(
            "The generated-set path cannot be empty."
        )

    return GeneratedSet(label=label, directory=Path(path_text))


def package_version_or_na(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "not installed"


def set_random_seed(seed: int) -> None:
    """Set the RNG state used by TorchMetrics subset sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def collect_images(folder: Path, expected_n: int | None) -> list[Path]:
    """Recursively collect supported image files in deterministic order."""
    folder = folder.expanduser().resolve()

    if not folder.exists():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )

    if not paths:
        raise ValueError(f"No supported images were found in: {folder}")

    if expected_n is not None and len(paths) != expected_n:
        raise ValueError(
            f"Expected {expected_n} images in {folder}, but found {len(paths)}."
        )

    return paths


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.size
    except Exception as exc:
        raise RuntimeError(f"Unable to read image header: {path}") from exc


def validate_uniform_size(paths: Sequence[Path], label: str) -> tuple[int, int]:
    """Require all source images in a set to have the same pixel dimensions."""
    sizes: dict[tuple[int, int], int] = {}
    for path in paths:
        size = image_size(path)
        sizes[size] = sizes.get(size, 0) + 1

    if len(sizes) != 1:
        summary = ", ".join(
            f"{width}x{height}: {count}"
            for (width, height), count in sorted(sizes.items())
        )
        raise ValueError(
            f"Images in '{label}' do not share one size. Found: {summary}"
        )

    return next(iter(sizes))


def load_rgb_uint8(path: Path) -> torch.Tensor:
    """
    Load an image as a uint8 tensor with shape [3, H, W] and range [0, 255].
    """
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.uint8).copy()
    except Exception as exc:
        raise RuntimeError(f"Unable to read image: {path}") from exc

    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(
            f"Invalid RGB tensor for {path}: shape={tuple(tensor.shape)}"
        )

    return tensor


@torch.inference_mode()
def add_images_to_metric(
    metric: KernelInceptionDistance,
    image_paths: Sequence[Path],
    *,
    real: bool,
    device: torch.device,
    batch_size: int,
    label: str,
) -> None:
    """Extract Inception features and update the KID metric."""
    total = len(image_paths)

    for start in range(0, total, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = [load_rgb_uint8(path) for path in batch_paths]

        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(
                f"A batch from '{label}' contains inconsistent shapes: "
                f"{sorted(shapes)}"
            )

        batch = torch.stack(tensors, dim=0).to(
            device=device,
            dtype=torch.uint8,
            non_blocking=True,
        )
        metric.update(batch, real=real)

        processed = min(start + batch_size, total)
        print(f"Feature extraction [{label}]: {processed}/{total}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(
    output_path: Path,
    reference_label: str,
    reference_paths: Sequence[Path],
    generated_sets: Sequence[tuple[GeneratedSet, Sequence[Path]]],
) -> None:
    """Save exact file lists and SHA-256 hashes used in the analysis."""
    rows: list[dict[str, str]] = []

    for path in reference_paths:
        rows.append({
            "set_role": "reference",
            "set_label": reference_label,
            "filename": path.name,
            "absolute_path": str(path.resolve()),
            "sha256": sha256_file(path),
        })

    for spec, paths in generated_sets:
        for path in paths:
            rows.append({
                "set_role": "generated",
                "set_label": spec.label,
                "filename": path.name,
                "absolute_path": str(path.resolve()),
                "sha256": sha256_file(path),
            })

    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "set_role", "set_label", "filename",
                "absolute_path", "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def safe_stem(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("._") or "kid_evaluation"


def create_bar_figure(
    results: Sequence[dict[str, object]],
    *,
    reference_label: str,
    title: str,
    output_stem: Path,
    annotate_ratios: bool,
    dpi: int,
) -> None:
    """Create one comparison bar chart for all generated sets in the run."""
    labels = [str(row["generated_label"]) for row in results]
    means = np.asarray(
        [float(row["kid_mean_x1000"]) for row in results], dtype=float
    )
    stds = np.asarray(
        [float(row["kid_subset_sd_x1000"]) for row in results], dtype=float
    )

    x = np.arange(len(labels))
    width = 0.62

    fig_width = max(6.8, 2.15 * len(labels) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 7.4))

    bars = ax.bar(
        x,
        means,
        width=width,
        yerr=stds,
        capsize=7,
        linewidth=1.8,
        alpha=0.82,
        error_kw={
            "elinewidth": 1.8,
            "ecolor": "black",
            "capthick": 1.8,
        },
    )

    finite_low = np.nanmin(means - stds)
    finite_high = np.nanmax(means + stds)
    data_span = max(finite_high - min(0.0, finite_low), 1.0)

    lower = min(0.0, finite_low - 0.08 * data_span)
    upper = finite_high + 0.24 * data_span
    ax.set_ylim(lower, upper)

    annotation_gap = 0.025 * (upper - lower)

    first_mean = means[0] if len(means) else np.nan

    for index, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        center = bar.get_x() + bar.get_width() / 2
        top = mean + std

        ax.text(
            center,
            top + 2.5 * annotation_gap,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
        ax.text(
            center,
            top + 1.1 * annotation_gap,
            f"±{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

        if (
            annotate_ratios
            and index > 0
            and np.isfinite(first_mean)
            and first_mean > 0
            and mean > 0
        ):
            ax.text(
                center,
                max(mean * 0.55, lower + 0.12 * (upper - lower)),
                f"×{mean / first_mean:.1f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\nvs {reference_label}" for label in labels],
        fontsize=11,
    )
    ax.set_ylabel("KID (×10⁻³; lower is better)", fontsize=13)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=18)
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    first_row = results[0]
    fig.text(
        0.5,
        0.02,
        (
            f"Reference n={first_row['n_reference']}; "
            f"{first_row['subsets']} subsets × "
            f"{first_row['subset_size']} images; "
            f"seed={first_row['subset_seed']}. "
            "Error bars show subset SD, not a confidence interval."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        style="italic",
    )

    fig.tight_layout(rect=[0.03, 0.07, 0.99, 0.98])

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate KID between one reference set and one or more "
            "generated image sets."
        )
    )

    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory containing the reference images.",
    )
    parser.add_argument(
        "--reference-label",
        default="D2",
        help="Label used for the reference image set.",
    )
    parser.add_argument(
        "--generated",
        type=parse_generated_spec,
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Generated-set label and directory. Repeat this option for "
            "multiple experiments."
        ),
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=None,
        help="Require exactly this many images in every set.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=2048,
        choices=[64, 192, 768, 2048],
    )
    parser.add_argument("--subsets", type=int, default=50)
    parser.add_argument("--subset-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stem",
        default="KID_results",
        help="Base name for output files.",
    )
    parser.add_argument(
        "--title",
        default="Distributional distance to the reference image set",
    )
    parser.add_argument(
        "--annotate-ratios",
        action="store_true",
        help="Annotate each later bar relative to the first generated set.",
    )
    parser.add_argument("--dpi", type=int, default=300)

    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def main() -> None:
    args = build_parser().parse_args()

    if args.subsets <= 0:
        raise ValueError("--subsets must be greater than zero.")
    if args.subset_size <= 1:
        raise ValueError("--subset-size must be greater than one.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.expected_n is not None and args.expected_n <= 1:
        raise ValueError("--expected-n must be greater than one.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(args.stem)

    device = resolve_device(args.device)
    set_random_seed(args.seed)

    print("=" * 72)
    print("Kernel Inception Distance evaluation")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Reference: {args.reference_label} -> {args.reference_dir}")
    print(
        f"KID settings: feature={args.feature_dim}, subsets={args.subsets}, "
        f"subset_size={args.subset_size}, seed={args.seed}"
    )

    reference_paths = collect_images(args.reference_dir, args.expected_n)
    reference_size = validate_uniform_size(
        reference_paths, args.reference_label
    )

    generated_with_paths: list[tuple[GeneratedSet, list[Path]]] = []
    for spec in args.generated:
        paths = collect_images(spec.directory, args.expected_n)
        size = validate_uniform_size(paths, spec.label)

        if size != reference_size:
            raise ValueError(
                f"Image size mismatch: reference={reference_size}, "
                f"{spec.label}={size}"
            )

        generated_with_paths.append((spec, paths))

    if args.subset_size > len(reference_paths):
        raise ValueError(
            f"subset_size={args.subset_size} exceeds reference n="
            f"{len(reference_paths)}."
        )

    for spec, paths in generated_with_paths:
        if args.subset_size > len(paths):
            raise ValueError(
                f"subset_size={args.subset_size} exceeds n={len(paths)} "
                f"for '{spec.label}'."
            )

    metric = KernelInceptionDistance(
        feature=args.feature_dim,
        subsets=args.subsets,
        subset_size=args.subset_size,
        degree=3,
        gamma=None,
        coef=1.0,
        normalize=False,
        reset_real_features=False,
    ).to(device)

    print("\nExtracting reference features...")
    add_images_to_metric(
        metric,
        reference_paths,
        real=True,
        device=device,
        batch_size=args.batch_size,
        label=args.reference_label,
    )

    results: list[dict[str, object]] = []

    for index, (spec, generated_paths) in enumerate(generated_with_paths, 1):
        print(f"\n[{index}/{len(generated_with_paths)}] {spec.label}")
        add_images_to_metric(
            metric,
            generated_paths,
            real=False,
            device=device,
            batch_size=args.batch_size,
            label=spec.label,
        )

        # TorchMetrics samples KID subsets during compute(). Re-seeding here
        # makes every comparison reproducible and, when sample counts match,
        # evaluates comparisons with the same random index sequence.
        set_random_seed(args.seed)
        kid_mean_tensor, kid_std_tensor = metric.compute()

        kid_mean = float(kid_mean_tensor.detach().cpu())
        kid_std = float(kid_std_tensor.detach().cpu())

        row: dict[str, object] = {
            "comparison_order": index,
            "reference_label": args.reference_label,
            "generated_label": spec.label,
            "reference_dir": str(args.reference_dir.expanduser().resolve()),
            "generated_dir": str(spec.directory.expanduser().resolve()),
            "n_reference": len(reference_paths),
            "n_generated": len(generated_paths),
            "image_width": reference_size[0],
            "image_height": reference_size[1],
            "feature_dimension": args.feature_dim,
            "subsets": args.subsets,
            "subset_size": args.subset_size,
            "subset_seed": args.seed,
            "kid_mean": kid_mean,
            "kid_subset_sd": kid_std,
            "kid_mean_x1000": kid_mean * 1000.0,
            "kid_subset_sd_x1000": kid_std * 1000.0,
            "ratio_to_first_mean": None,
        }
        results.append(row)

        print(
            f"KID ×10^-3 = {kid_mean * 1000.0:.2f} "
            f"± {kid_std * 1000.0:.2f}"
        )

        # Clear only generated features. Reference features are retained
        # because reset_real_features=False.
        metric.reset()

    first_mean = float(results[0]["kid_mean"])
    for row in results:
        current = float(row["kid_mean"])
        row["ratio_to_first_mean"] = (
            current / first_mean
            if first_mean > 0 and current > 0
            else None
        )

    results_csv = output_dir / f"{stem}_results.csv"
    with results_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    manifest_csv = output_dir / f"{stem}_image_manifest.csv"
    save_manifest(
        manifest_csv,
        args.reference_label,
        reference_paths,
        generated_with_paths,
    )

    metadata = {
        "protocol": {
            "metric": "Kernel Inception Distance",
            "feature_dimension": args.feature_dim,
            "polynomial_degree": 3,
            "gamma": "1 / feature_dimension",
            "coef": 1.0,
            "subsets": args.subsets,
            "subset_size": args.subset_size,
            "subset_seed": args.seed,
            "reported_error": "standard deviation across KID subsets",
            "pairing": (
                "None. KID is set-level; reference and generated features "
                "are sampled independently within each subset."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchmetrics": torchmetrics.__version__,
            "torchvision": package_version_or_na("torchvision"),
            "torch-fidelity": package_version_or_na("torch-fidelity"),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "matplotlib": matplotlib.__version__,
            "device": str(device),
            "cuda_runtime_reported_by_pytorch": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),
        },
        "reference": {
            "label": args.reference_label,
            "directory": str(args.reference_dir.expanduser().resolve()),
            "n": len(reference_paths),
            "size": list(reference_size),
        },
        "generated_sets": [
            {
                "label": spec.label,
                "directory": str(spec.directory.expanduser().resolve()),
                "n": len(paths),
            }
            for spec, paths in generated_with_paths
        ],
    }

    metadata_json = output_dir / f"{stem}_run_metadata.json"
    metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure_stem = output_dir / f"{stem}_bar"
    create_bar_figure(
        results,
        reference_label=args.reference_label,
        title=args.title,
        output_stem=figure_stem,
        annotate_ratios=args.annotate_ratios,
        dpi=args.dpi,
    )

    print("\nOutputs")
    print(f"- {results_csv}")
    print(f"- {manifest_csv}")
    print(f"- {metadata_json}")
    print(f"- {figure_stem.with_suffix('.png')}")
    print(f"- {figure_stem.with_suffix('.svg')}")
    print(f"- {figure_stem.with_suffix('.pdf')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
