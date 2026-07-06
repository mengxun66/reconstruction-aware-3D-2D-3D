"""
Automated post-hoc KID-vs-epoch evaluation for LoRA training.
PRODUCTION-MATCHED SAMPLING VERSION (Euler a / 20 steps / CFG 7.0 / clip-skip 2; N=20, 50×16 KID resampling):
  Sampling configuration is matched to the main A1111 generation pipeline
  (Sampler: Euler a, Steps: 20, CFG scale: 7, Clip skip: 2).
  NOTE: A1111 "Clip skip: 2" corresponds to diffusers clip_skip=1 (penultimate
  CLIP layer). Pixel-level parity with A1111 is impossible (different seed
  semantics, ENSD) and NOT required: the diagnostic is a distribution-level
  relative comparison across checkpoints under one fixed configuration.
  Default output dirs carry the suffix "_eulerA" so earlier DPMSolver runs
  are never overwritten (keep those as an Appendix sensitivity check).

This full version can evaluate LOW and HIGH in one run and exports complete
seed records for reproducibility:
  - case_generation_seeds.csv: one row per reference case
  - generation_seed_log_by_epoch.csv: one row per generated image per epoch
  - kid_bootstrap_subsets.csv: bootstrap subset records used for KID
  - kid_vs_epoch.csv: KID mean/std for every checkpoint
  - fig_low_loss_kid.png / fig_high_loss_kid.png: loss + KID figures

Default behavior: evaluate both domains sequentially.

Examples:
    python automated_kid_evaluation_eulerA_matched_N20_50x16.py
    python automated_kid_evaluation_eulerA_matched_N20_50x16.py --domain all
    python automated_kid_evaluation_eulerA_matched_N20_50x16.py --domain low
    python automated_kid_evaluation_eulerA_matched_N20_50x16.py --domain high

On AutoDL/Linux, run from the project folder or specify project_root:
    cd /root/autodl-tmp/REVISE2605
    python automated_kid_evaluation_eulerA_matched_N20_50x16.py --base_model ./sd-models/sd-v1-5

On Windows:
    cd /d E:\artificalI\send\REVISE2605
    python automated_kid_evaluation_eulerA_matched_N20_50x16.py --base_model "E:\artificalI\send\REVISE2605\sd-models\sd-v1-5"
"""

import argparse
import hashlib
import inspect
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from cleanfid import fid


# ============================================================
# Default project root and random-control constants
# ============================================================

WINDOWS_PROJECT_ROOT = Path(r"E:\artificalI\send\REVISE2605")
DEFAULT_PROJECT_ROOT = WINDOWS_PROJECT_ROOT if WINDOWS_PROJECT_ROOT.exists() else Path.cwd()

DEFAULT_KID_BOOTSTRAP_SEED = 42
DEFAULT_KID_BOOTSTRAP_REPEATS = 50
DEFAULT_KID_SUBSET_SIZE = 16

# ------------------------------------------------------------
# Production-matched sampling configuration
# (matches the main A1111 generation: Euler a, 20 steps, CFG 7, Clip skip 2)
# A1111 "Clip skip: 2" == diffusers clip_skip=1 (penultimate CLIP layer).
# ------------------------------------------------------------
SAMPLER_NAME = "EulerAncestralDiscrete"
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.0
CLIP_SKIP_DIFFUSERS = 1      # == A1111 "Clip skip: 2"
CLIP_SKIP_A1111_EQUIV = 2


def assert_clip_skip_supported():
    """Fail fast if this diffusers version cannot pass clip_skip at call time.

    The LoRA was trained (kohya clip_skip=2) and deployed (A1111 Clip skip: 2)
    with the penultimate CLIP layer; silently dropping clip_skip would
    reintroduce a configuration mismatch, so we refuse to run instead.
    """
    sig = inspect.signature(StableDiffusionPipeline.__call__)
    if "clip_skip" not in sig.parameters:
        raise RuntimeError(
            "This diffusers version does not support the `clip_skip` argument in "
            "StableDiffusionPipeline.__call__, but the LoRA was trained and "
            "deployed with clip-skip 2. Upgrade first:  pip install -U diffusers"
        )


# ============================================================
# Reproducibility
# ============================================================

def get_case_seed(case_id: str) -> int:
    """
    Deterministic per-case generation seed.

    The seed is derived from case_id, usually the txt/png filename stem.
    Therefore the same case always uses the same seed across all epochs,
    while different cases use different seeds.
    """
    h = hashlib.sha256(str(case_id).encode()).hexdigest()
    return int(h[:8], 16)


def export_case_generation_seeds(val_cases, output_root: Path, domain: str) -> pd.DataFrame:
    """Export one seed row per case."""
    output_root.mkdir(parents=True, exist_ok=True)
    seed_df = pd.DataFrame([
        {
            "domain": domain,
            "case_id": case["case_id"],
            "generation_seed": get_case_seed(case["case_id"]),
            "prompt": case["prompt"],
            "real_image_path": case.get("real_image_path", ""),
        }
        for case in val_cases
    ])
    out_csv = output_root / "case_generation_seeds.csv"
    seed_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Case generation seeds saved: {out_csv}")
    return seed_df


# ============================================================
# Step 1: Load LoRA checkpoint and generate reference images
# ============================================================

def setup_pipeline(base_model_path: str, lora_checkpoint_path: str, device="cuda"):
    """Load SD v1.5 base + apply LoRA weights from checkpoint."""
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    pipe.load_lora_weights(lora_checkpoint_path)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_val_images(pipe, val_cases, output_dir: Path, device="cuda",
                        domain: str = "", epoch: int | None = None,
                        checkpoint_name: str = "", print_seeds: bool = True):
    """
    Generate one image per case with a fixed per-case seed.

    Returns a list of seed-log rows for CSV export.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_log_rows = []

    for case in val_cases:
        seed = get_case_seed(case["case_id"])
        if print_seeds:
            print(f"    Generating {case['case_id']} with seed = {seed}")

        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            result = pipe(
                prompt=case["prompt"],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                clip_skip=CLIP_SKIP_DIFFUSERS,
                generator=generator,
                width=512,
                height=512,
            )

        out_path = output_dir / f"{case['case_id']}.png"
        result.images[0].save(out_path)

        seed_log_rows.append({
            "domain": domain,
            "epoch": epoch,
            "checkpoint": checkpoint_name,
            "case_id": case["case_id"],
            "generation_seed": seed,
            "prompt": case["prompt"],
            "sampler": SAMPLER_NAME,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "clip_skip_diffusers": CLIP_SKIP_DIFFUSERS,
            "clip_skip_a1111_equiv": CLIP_SKIP_A1111_EQUIV,
            "width": 512,
            "height": 512,
            "generated_image_path": str(out_path),
        })

    return seed_log_rows


# ============================================================
# Step 2: Compute KID with bootstrap stabilization
# ============================================================

def compute_kid_bootstrap(real_dir: Path, fake_dir: Path,
                          n_bootstrap: int = DEFAULT_KID_BOOTSTRAP_REPEATS,
                          subset_size: int = DEFAULT_KID_SUBSET_SIZE,
                          bootstrap_seed: int = DEFAULT_KID_BOOTSTRAP_SEED,
                          bootstrap_csv_path: Path | None = None,
                          domain: str = "",
                          epoch: int | None = None,
                          checkpoint_name: str = ""):
    """
    Bootstrap-stabilized KID for relatively small validation/reference sets.

    Exports the actual bootstrap subsets if bootstrap_csv_path is provided.

    Returns
    -------
    kid_mean, kid_std : float, float
    bootstrap_rows : list[dict]
    """
    real_paths = sorted(real_dir.glob("*.png"))
    fake_paths = sorted(fake_dir.glob("*.png"))
    assert len(real_paths) == len(fake_paths), \
        f"Real ({len(real_paths)}) and fake ({len(fake_paths)}) counts mismatch"

    n = len(real_paths)
    if n == 0:
        raise RuntimeError(f"No PNG files found in real_dir={real_dir} or fake_dir={fake_dir}")

    # Optional filename sanity check. It is not required by clean-fid,
    # but it prevents accidental real/fake mismatches in small sets.
    real_names = [p.name for p in real_paths]
    fake_names = [p.name for p in fake_paths]
    if real_names != fake_names:
        print("Warning: real and generated PNG filenames are not identical after sorting.")
        print("         KID can still be computed as distributions, but case-wise seed logs should be checked.")

    bootstrap_rows = []

    if n <= subset_size:
        kid = fid.compute_kid(str(real_dir), str(fake_dir), mode="clean")
        bootstrap_rows.append({
            "domain": domain,
            "epoch": epoch,
            "checkpoint": checkpoint_name,
            "bootstrap_iter": "direct_all",
            "bootstrap_seed": bootstrap_seed,
            "subset_size": n,
            "sampled_case_ids": "|".join([p.stem for p in real_paths]),
            "sampled_indices": "|".join(map(str, range(n))),
            "kid": float(kid),
        })
        if bootstrap_csv_path is not None:
            pd.DataFrame(bootstrap_rows).to_csv(bootstrap_csv_path, index=False, encoding="utf-8-sig")
        return float(kid), 0.0, bootstrap_rows

    kid_samples = []
    rng = np.random.RandomState(bootstrap_seed)

    tmp_real = fake_dir.parent / "_tmp_real"
    tmp_fake = fake_dir.parent / "_tmp_fake"
    tmp_real.mkdir(exist_ok=True)
    tmp_fake.mkdir(exist_ok=True)

    try:
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=subset_size, replace=False)
            idx_sorted = sorted(idx.tolist())

            for p in tmp_real.glob("*.png"):
                p.unlink()
            for p in tmp_fake.glob("*.png"):
                p.unlink()

            for i in idx_sorted:
                shutil.copy2(real_paths[i], tmp_real / real_paths[i].name)
                shutil.copy2(fake_paths[i], tmp_fake / fake_paths[i].name)

            kid = fid.compute_kid(str(tmp_real), str(tmp_fake), mode="clean")
            kid_samples.append(kid)

            bootstrap_rows.append({
                "domain": domain,
                "epoch": epoch,
                "checkpoint": checkpoint_name,
                "bootstrap_iter": b + 1,
                "bootstrap_seed": bootstrap_seed,
                "subset_size": subset_size,
                "sampled_case_ids": "|".join([real_paths[i].stem for i in idx_sorted]),
                "sampled_indices": "|".join(map(str, idx_sorted)),
                "kid": float(kid),
            })
    finally:
        for p in tmp_real.glob("*.png"):
            p.unlink()
        for p in tmp_fake.glob("*.png"):
            p.unlink()
        if tmp_real.exists():
            tmp_real.rmdir()
        if tmp_fake.exists():
            tmp_fake.rmdir()

    if bootstrap_csv_path is not None:
        pd.DataFrame(bootstrap_rows).to_csv(bootstrap_csv_path, index=False, encoding="utf-8-sig")

    return float(np.mean(kid_samples)), float(np.std(kid_samples)), bootstrap_rows


# ============================================================
# Step 3: Load reference cases (prompts + real D2 images)
# ============================================================

def load_val_cases(val_dir: Path):
    """
    Expected directory structure:
      val_dir/
        ├─ low_01.png
        ├─ low_01.txt
        ├─ low_02.png
        ├─ low_02.txt
        └─ real_d2/
             low_01.png
             low_02.png
    """
    cases = []
    for txt_path in sorted(val_dir.glob("*.txt")):
        case_id = txt_path.stem
        png_path = val_dir / f"{case_id}.png"
        if not png_path.exists():
            print(f"Warning: no PNG for {case_id}")
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        cases.append({
            "case_id": case_id,
            "prompt": prompt,
            "real_image_path": str(png_path),
        })
    return cases


# ============================================================
# Step 4: Parse loss data
# ============================================================

def load_loss_csv(csv_path: Path):
    """Load step-by-step loss from CSV. Supports Step,Value or Step,Value,LossAverage."""
    df = pd.read_csv(csv_path)
    if "LossAverage" in df.columns:
        value_col = "LossAverage"
    elif "Value" in df.columns:
        value_col = "Value"
    else:
        raise ValueError(f"CSV {csv_path} must contain 'Value' or 'LossAverage' column")
    return df["Step"].values, df[value_col].values


def ema_smooth(values, alpha=0.9):
    """TensorBoard-style EMA smoothing."""
    values = np.asarray(values, dtype=float)
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed


def detect_plateau(losses, steps, window_size=2000,
                   threshold=0.20, patience=2, ema_alpha=0.9):
    """Detect a plateau region from smoothed loss."""
    losses = np.asarray(losses, dtype=float)
    steps = np.asarray(steps)
    smoothed = ema_smooth(losses, alpha=ema_alpha)

    if len(smoothed) < max(2 * window_size + 1, 1000):
        return None, None, smoothed

    warmup = min(500, max(1, len(smoothed) // 20))
    if warmup + 2 * window_size >= len(smoothed):
        return None, None, smoothed

    L_start = smoothed[warmup:warmup + window_size].mean()
    L_after = smoothed[warmup + window_size:warmup + 2 * window_size].mean()
    initial_slope = max((L_start - L_after) / window_size, 1e-6)

    plateau_idx = None
    consecutive = 0
    for i in range(2 * window_size, len(smoothed), window_size):
        L_window = smoothed[i - window_size:i].mean()
        L_prev = smoothed[i - 2 * window_size:i - window_size].mean()
        if L_prev <= 0:
            continue
        current_slope = (L_prev - L_window) / window_size
        r = current_slope / initial_slope
        if r < threshold:
            consecutive += 1
            if consecutive >= patience and plateau_idx is None:
                plateau_idx = i - (patience - 1) * window_size
        else:
            consecutive = 0

    if plateau_idx is None:
        return None, None, smoothed
    return int(steps[plateau_idx]), float(smoothed[plateau_idx]), smoothed


# ============================================================
# Step 5: Full evaluation for one domain
# ============================================================

def evaluate_all_checkpoints(
    base_model_path: str,
    checkpoint_dir: Path,
    val_dir: Path,
    real_d2_dir: Path,
    output_root: Path,
    domain: str,
    checkpoint_pattern: str = "*_ep*.safetensors",
    epoch_regex: str = r"_ep(\d+)",
    device: str = "cuda",
    kid_bootstrap_seed: int = DEFAULT_KID_BOOTSTRAP_SEED,
    kid_bootstrap_repeats: int = DEFAULT_KID_BOOTSTRAP_REPEATS,
    kid_subset_size: int = DEFAULT_KID_SUBSET_SIZE,
    print_seeds: bool = True,
):
    output_root.mkdir(parents=True, exist_ok=True)

    val_cases = load_val_cases(val_dir)
    print(f"Loaded {len(val_cases)} cases from: {val_dir}")
    if len(val_cases) == 0:
        raise RuntimeError(f"No valid cases found in {val_dir}")

    # Export one stable seed table for all reference cases in this domain.
    export_case_generation_seeds(val_cases, output_root, domain=domain)

    checkpoint_files = sorted(checkpoint_dir.glob(checkpoint_pattern))
    epoch_pattern = re.compile(epoch_regex)
    checkpoints = []
    for ckpt in checkpoint_files:
        match = epoch_pattern.search(ckpt.stem)
        if match:
            checkpoints.append((int(match.group(1)), ckpt))
    checkpoints.sort()

    print(f"Found {len(checkpoints)} checkpoints in: {checkpoint_dir}")
    if len(checkpoints) == 0:
        raise RuntimeError(f"No checkpoints found matching {checkpoint_pattern} in {checkpoint_dir}")

    results = []
    all_generation_seed_rows = []
    all_bootstrap_rows = []

    for epoch, ckpt_path in checkpoints:
        print(f"\n=== Epoch {epoch}: {ckpt_path.name} ===")
        pipe = setup_pipeline(base_model_path, str(ckpt_path), device=device)
        gen_dir = output_root / f"generated_ep{epoch}"

        generation_seed_rows = generate_val_images(
            pipe,
            val_cases,
            gen_dir,
            device=device,
            domain=domain,
            epoch=epoch,
            checkpoint_name=ckpt_path.name,
            print_seeds=print_seeds,
        )
        all_generation_seed_rows.extend(generation_seed_rows)

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        kid_mean, kid_std, bootstrap_rows = compute_kid_bootstrap(
            real_d2_dir,
            gen_dir,
            n_bootstrap=kid_bootstrap_repeats,
            subset_size=kid_subset_size,
            bootstrap_seed=kid_bootstrap_seed,
            bootstrap_csv_path=None,
            domain=domain,
            epoch=epoch,
            checkpoint_name=ckpt_path.name,
        )
        all_bootstrap_rows.extend(bootstrap_rows)
        print(f"  KID = {kid_mean:.5f} ± {kid_std:.5f}")

        results.append({
            "domain": domain,
            "epoch": epoch,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "checkpoint": ckpt_path.name,
            "kid_bootstrap_seed": kid_bootstrap_seed,
            "kid_bootstrap_repeats": kid_bootstrap_repeats,
            "kid_subset_size": kid_subset_size if len(val_cases) > kid_subset_size else len(val_cases),
        })

    # Save output CSV files.
    df = pd.DataFrame(results)
    csv_path = output_root / "kid_vs_epoch.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"KID results saved: {csv_path}")

    generation_seed_csv = output_root / "generation_seed_log_by_epoch.csv"
    pd.DataFrame(all_generation_seed_rows).to_csv(generation_seed_csv, index=False, encoding="utf-8-sig")
    print(f"Generation seed log saved: {generation_seed_csv}")

    bootstrap_csv = output_root / "kid_bootstrap_subsets.csv"
    pd.DataFrame(all_bootstrap_rows).to_csv(bootstrap_csv, index=False, encoding="utf-8-sig")
    print(f"KID bootstrap subset log saved: {bootstrap_csv}")

    # One compact reproducibility summary.
    summary_csv = output_root / "random_seed_summary.csv"
    summary_df = pd.DataFrame([{
        "domain": domain,
        "generation_seed_rule": "sha256(case_id) first 8 hex digits converted to int",
        "sampler": SAMPLER_NAME,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "clip_skip_diffusers": CLIP_SKIP_DIFFUSERS,
        "clip_skip_a1111_equiv": CLIP_SKIP_A1111_EQUIV,
        "kid_bootstrap_seed": kid_bootstrap_seed,
        "kid_bootstrap_repeats": kid_bootstrap_repeats,
        "kid_subset_size": kid_subset_size,
        "device": device,
        "num_cases": len(val_cases),
        "num_checkpoints": len(checkpoints),
    }])
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Random seed summary saved: {summary_csv}")

    return df


# ============================================================
# Step 6: Plotting
# ============================================================

def plot_combined(loss_csv: Path, kid_results: pd.DataFrame,
                  output_path: Path, title: str):
    """Generate loss + KID figure."""
    steps, losses = load_loss_csv(loss_csv)
    plateau_step, plateau_loss, smoothed = detect_plateau(losses, steps)
    best_epoch_idx = kid_results["kid_mean"].idxmin()
    best_epoch = int(kid_results.loc[best_epoch_idx, "epoch"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=False,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    ax1.plot(steps, losses, color="#bcd2e8", linewidth=0.8, alpha=0.5)
    ax1.plot(steps, smoothed, color="#1f3a5f", linewidth=1.5)
    if plateau_step is not None:
        ax1.axvline(plateau_step, color="#c0392b", linestyle="--", linewidth=1.3)
        ax1.annotate(
            f"Plateau detected\nstep={plateau_step}",
            xy=(plateau_step, plateau_loss),
            xytext=(plateau_step * 0.5, plateau_loss * 1.25),
            fontsize=9, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8),
        )
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Training Loss")
    ax1.set_title(f"{title}: Training Loss")
    ax1.grid(alpha=0.3)

    epochs = kid_results["epoch"].values
    kids = kid_results["kid_mean"].values
    kid_stds = kid_results["kid_std"].values
    ax2.errorbar(
        epochs, kids * 1000, yerr=kid_stds * 1000,
        fmt="o-", color="#c0392b", linewidth=1.5,
        markersize=6, capsize=3
    )
    ax2.axvline(best_epoch, color="#27ae60", linestyle=":", linewidth=1.3)
    ax2.annotate(
        f"Min KID\nepoch={best_epoch}",
        xy=(best_epoch, kids[best_epoch_idx] * 1000),
        xytext=(best_epoch + 1, kids[best_epoch_idx] * 1000 * 1.08),
        fontsize=9, color="#27ae60",
    )
    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel("Training-reference KID (×10⁻³)")
    ax2.set_title(f"{title}: Training-reference KID across epochs")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")
    return best_epoch, plateau_step


def plot_kid_only(kid_results: pd.DataFrame, output_path: Path, title: str):
    best_epoch_idx = kid_results["kid_mean"].idxmin()
    best_epoch = int(kid_results.loc[best_epoch_idx, "epoch"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        kid_results["epoch"].values,
        kid_results["kid_mean"].values * 1000,
        yerr=kid_results["kid_std"].values * 1000,
        fmt="o-", linewidth=1.5, markersize=6, capsize=3,
    )
    ax.axvline(best_epoch, linestyle=":", linewidth=1.3)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Training-reference KID (×10⁻³)")
    ax.set_title(f"{title}: Training-reference KID across epochs")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")
    return best_epoch


# ============================================================
# Utilities
# ============================================================

def resolve_project_paths(project_root: Path, args, domain: str):
    val_base_1 = project_root / "val_set" / domain
    val_base_2 = project_root / "valset" / domain
    val_dir = Path(args.val_dir) if args.val_dir else (val_base_1 if val_base_1.exists() else val_base_2)

    real_default = val_dir / "real_d2"
    real_d2_dir = Path(args.real_d2_dir) if args.real_d2_dir else real_default

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else project_root / "output"
    loss_csv = Path(args.loss_csv) if args.loss_csv else project_root / "logs" / f"{domain}_loss.csv"
    output_root = Path(args.output_root) if args.output_root else project_root / "eval_results" / f"{domain}_eulerA"

    return {
        "val_dir": val_dir,
        "real_d2_dir": real_d2_dir,
        "checkpoint_dir": checkpoint_dir,
        "loss_csv": loss_csv,
        "output_root": output_root,
    }


def run_one_domain(domain: str, args):
    print("\n" + "=" * 72)
    print(f"Running domain: {domain}")
    print("=" * 72)
    print("Sampling config (production-matched): "
          f"sampler={SAMPLER_NAME}, steps={NUM_INFERENCE_STEPS}, "
          f"CFG={GUIDANCE_SCALE}, clip_skip(diffusers)={CLIP_SKIP_DIFFUSERS} "
          f"(= A1111 Clip skip {CLIP_SKIP_A1111_EQUIV})")
    assert_clip_skip_supported()

    project_root = Path(args.project_root)
    paths = resolve_project_paths(project_root, args, domain)

    print(f"Project root : {project_root}")
    print(f"Checkpoint dir: {paths['checkpoint_dir']}")
    print(f"Val dir       : {paths['val_dir']}")
    print(f"Real D2 dir   : {paths['real_d2_dir']}")
    print(f"Loss CSV      : {paths['loss_csv']}")
    print(f"Output root   : {paths['output_root']}")
    print(f"KID bootstrap seed   : {args.kid_bootstrap_seed}")
    print(f"KID bootstrap repeats: {args.kid_bootstrap_repeats}")
    print(f"KID subset size      : {args.kid_subset_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")

    df = evaluate_all_checkpoints(
        base_model_path=args.base_model,
        checkpoint_dir=paths["checkpoint_dir"],
        val_dir=paths["val_dir"],
        real_d2_dir=paths["real_d2_dir"],
        output_root=paths["output_root"],
        domain=domain,
        checkpoint_pattern=f"{domain}*_ep*.safetensors",
        device=device,
        kid_bootstrap_seed=args.kid_bootstrap_seed,
        kid_bootstrap_repeats=args.kid_bootstrap_repeats,
        kid_subset_size=args.kid_subset_size,
        print_seeds=not args.no_print_seeds,
    )

    if paths["loss_csv"].exists():
        best_epoch, plateau_step = plot_combined(
            loss_csv=paths["loss_csv"],
            kid_results=df,
            output_path=paths["output_root"] / f"fig_{domain}_loss_kid.png",
            title=f"{domain.capitalize()}-rise LoRA",
        )
    else:
        print(f"Warning: loss CSV not found: {paths['loss_csv']}")
        plateau_step = None
        best_epoch = plot_kid_only(
            kid_results=df,
            output_path=paths["output_root"] / f"fig_{domain}_kid_only.png",
            title=f"{domain.capitalize()}-rise LoRA",
        )

    print(f"\n=== Summary ({domain}) ===")
    print(f"Plateau step    : {plateau_step}")
    print(f"Best epoch (KID): {best_epoch}")
    print(f"All results in  : {paths['output_root']}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["all", "low", "high"], default="all",
                        help="Default is 'all', which runs low and high sequentially.")
    parser.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--base_model", default="./sd-models/sd-v1-5")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--real_d2_dir", default=None)
    parser.add_argument("--loss_csv", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--kid_bootstrap_seed", type=int, default=DEFAULT_KID_BOOTSTRAP_SEED)
    parser.add_argument("--kid_bootstrap_repeats", type=int, default=DEFAULT_KID_BOOTSTRAP_REPEATS)
    parser.add_argument("--kid_subset_size", type=int, default=DEFAULT_KID_SUBSET_SIZE)
    parser.add_argument("--no_print_seeds", action="store_true",
                        help="Do not print each case generation seed to the terminal.")
    args = parser.parse_args()

    domains = ["low", "high"] if args.domain == "all" else [args.domain]
    for domain in domains:
        run_one_domain(domain, args)
