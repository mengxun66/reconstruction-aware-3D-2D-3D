# som_d3_base_vs_simplified.py
# -*- coding: utf-8 -*-
"""
D3-base SOM comparison:
    D3 full-prompt outputs = fixed reference manifold / base map
    Simplified-prompt outputs = projected comparison set

Main outputs:
1. d3base_joint_som_large.png
2. d3base_difference.png
3. d3base_migration_filtered.png
4. d3base_cluster_bars.png
5. d3base_shift_analysis.png
6. d3base_cluster_flow.png
7. case_matching_report.csv
8. d3_vs_simplified_assignments.csv
9. cluster_transition_matrix.csv
10. som_comparison_summary.csv

Important:
- The SOM is trained ONLY on the matched D3 feature vectors.
- Simplified-prompt images do not participate in SOM training.
- Pairwise migration is matched by normalized case ID, not by folder sorting order.
- Type A-E labels are local to this D3-base SOM and should not be treated as
  identical to labels from a separately trained D2-D3 SOM.
"""

import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

from minisom import MiniSom
from sklearn.cluster import KMeans


# ============================================================
# 1. CONFIGURATION: edit these three paths
# ============================================================

# Full-prompt D3 images: final low ep16 + high ep20 outputs
D3_FOLDER = r"E:\artificalI\send\REVISE2605\simple\NEWALL"

# Simplified-prompt outputs generated with the same case seeds
SIMPLIFIED_FOLDER = r"E:\artificalI\send\REVISE2605\simple\LORAsim\image"

# Output folder
OUTPUT_FOLDER = r"E:\artificalI\send\REVISE2605\som_d3base_vs_simplified"

SOM_SIZE = 10
SEED = 42
N_CLUSTERS = 5
SOM_ITERATIONS = 20000

# Percentage of largest BMU shifts shown in the migration-arrow figure
MIGRATION_TOP_PERCENT = 50

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. CASE-ID NORMALIZATION AND MATCHING
# ============================================================

def normalize_case_id(path_or_name):
    """
    Convert filename variants to a shared case ID.

    Built-in mappings are based on the naming examples shown in the project:
      Simplified AA01.png  <-> D3 1.png
      Simplified BB01.png  <-> D3 B01.png
      CC01.png             <-> CC01.png
      D01.png              <-> D01.png
      X01.png              <-> X01.png

    Examples:
      1.png      -> A01
      15.png     -> A15
      AA01.png   -> A01
      BB03.png   -> B03
      B03.png    -> B03
      CC04.png   -> CC04
      D07.png    -> D07
      X48.png    -> X48

    Edit this function if your actual filename convention differs.
    """
    stem = Path(path_or_name).stem.upper()
    stem = re.sub(r"[^A-Z0-9]", "", stem)

    # AA01 -> A01
    m = re.fullmatch(r"AA(\d+)", stem)
    if m:
        return f"A{int(m.group(1)):02d}"

    # Numeric-only D3 names: 1 -> A01
    m = re.fullmatch(r"(\d+)", stem)
    if m:
        return f"A{int(m.group(1)):02d}"

    # BB01 -> B01
    m = re.fullmatch(r"BB(\d+)", stem)
    if m:
        return f"B{int(m.group(1)):02d}"

    # B01 / CC01 / D01 / X01 and similar letter+number IDs
    m = re.fullmatch(r"([A-Z]+)(\d+)", stem)
    if m:
        prefix = m.group(1)
        number = int(m.group(2))
        return f"{prefix}{number:02d}"

    return stem


def list_image_paths(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    if not paths:
        raise ValueError(f"No images found in: {folder}")
    return paths


def build_case_map(paths, set_name):
    """
    Returns:
        case_map: normalized_case_id -> path
        duplicate_rows: duplicate-ID records
    """
    case_map = {}
    duplicate_rows = []

    for p in paths:
        cid = normalize_case_id(p.name)
        if cid in case_map:
            duplicate_rows.append({
                "set": set_name,
                "normalized_case_id": cid,
                "kept_file": case_map[cid].name,
                "duplicate_file": p.name,
            })
        else:
            case_map[cid] = p

    return case_map, duplicate_rows


def match_case_paths(d3_folder, simplified_folder, output_folder):
    d3_paths = list_image_paths(d3_folder)
    simple_paths = list_image_paths(simplified_folder)

    d3_map, d3_duplicates = build_case_map(d3_paths, "D3")
    simple_map, simple_duplicates = build_case_map(simple_paths, "Simplified")

    d3_ids = set(d3_map)
    simple_ids = set(simple_map)
    common_ids = sorted(d3_ids & simple_ids)

    if len(common_ids) < 2:
        raise ValueError(
            "Fewer than 2 matched cases. Check normalize_case_id() and filenames."
        )

    report_rows = []
    for cid in sorted(d3_ids | simple_ids):
        report_rows.append({
            "normalized_case_id": cid,
            "d3_file": d3_map[cid].name if cid in d3_map else "",
            "simplified_file": simple_map[cid].name if cid in simple_map else "",
            "matched": cid in d3_map and cid in simple_map,
        })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(
        Path(output_folder) / "case_matching_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    duplicate_df = pd.DataFrame(d3_duplicates + simple_duplicates)
    if not duplicate_df.empty:
        duplicate_df.to_csv(
            Path(output_folder) / "duplicate_case_ids.csv",
            index=False,
            encoding="utf-8-sig"
        )
        print("Warning: duplicate normalized case IDs were found.")
        print(duplicate_df.to_string(index=False))

    print("\nCase matching:")
    print(f"  D3 images:            {len(d3_paths)}")
    print(f"  Simplified images:    {len(simple_paths)}")
    print(f"  Matched cases:        {len(common_ids)}")
    print(f"  D3 only:              {len(d3_ids - simple_ids)}")
    print(f"  Simplified only:      {len(simple_ids - d3_ids)}")

    if d3_ids - simple_ids:
        print("  Unmatched D3 IDs:", sorted(d3_ids - simple_ids))
    if simple_ids - d3_ids:
        print("  Unmatched simplified IDs:", sorted(simple_ids - d3_ids))

    matched_d3_paths = [d3_map[cid] for cid in common_ids]
    matched_simple_paths = [simple_map[cid] for cid in common_ids]

    return common_ids, matched_d3_paths, matched_simple_paths


# ============================================================
# 4. DINOV2 FEATURE EXTRACTION
# ============================================================

def extract_features(image_paths, model, transform, device, description):
    features = []
    valid_paths = []

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(img_t).cpu().numpy().flatten()

            norm = np.linalg.norm(feat)
            if norm < 1e-12:
                raise ValueError(f"Zero-norm feature: {p}")

            feat = feat / norm
            features.append(feat)
            valid_paths.append(str(p))

        except Exception as exc:
            print(f"Warning: failed to process {p}: {exc}")

    if len(features) < 2:
        raise ValueError(f"Too few valid images in {description}")

    return np.asarray(features, dtype=np.float32), valid_paths


# ============================================================
# 5. D3-BASE SOM COLOR MAP AND CLUSTERS
# ============================================================

def get_colormap(som, n_clusters=5, contrast_strength=0.80):
    """
    K-means clusters are fitted to D3-base SOM node weights.
    U-matrix values modulate within-cluster lightness.
    """
    weights = som.get_weights()
    h, w, d = weights.shape
    weights_flat = weights.reshape(-1, d)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=SEED,
        n_init=10
    )
    labels = kmeans.fit_predict(weights_flat).reshape(h, w)
    u_matrix = som.distance_map()

    cluster_color_ranges = [
        {"dark": (0.10, 0.35, 0.10), "light": (0.45, 0.78, 0.45)},  # green
        {"dark": (0.08, 0.25, 0.55), "light": (0.45, 0.68, 0.90)},  # blue
        {"dark": (0.65, 0.45, 0.05), "light": (0.95, 0.78, 0.30)},  # yellow
        {"dark": (0.55, 0.12, 0.10), "light": (0.92, 0.45, 0.40)},  # red
        {"dark": (0.35, 0.50, 0.12), "light": (0.72, 0.88, 0.45)},  # light green
    ]

    color_map = np.zeros((h, w, 3), dtype=float)

    for c in range(n_clusters):
        mask = labels == c
        if not mask.any():
            continue

        u_vals = u_matrix[mask]
        u_min, u_max = u_vals.min(), u_vals.max()
        if u_max - u_min < 1e-8:
            u_cluster_norm = np.zeros_like(u_vals)
        else:
            u_cluster_norm = (u_vals - u_min) / (u_max - u_min)

        dark = np.asarray(cluster_color_ranges[c]["dark"])
        light = np.asarray(cluster_color_ranges[c]["light"])

        idx = 0
        for i in range(h):
            for j in range(w):
                if labels[i, j] == c:
                    blend = u_cluster_norm[idx] * contrast_strength
                    color_map[i, j] = light * (1.0 - blend) + dark * blend
                    idx += 1

    return color_map, labels


def type_name(cluster_id):
    return f"Type {chr(65 + int(cluster_id))}"


# ============================================================
# 6. ASSIGNMENTS AND TABLE OUTPUT
# ============================================================

def build_assignment_table(
    case_ids, som, d3_feats, simple_feats,
    d3_paths, simple_paths, labels
):
    rows = []

    for cid, d3_f, simple_f, d3_p, simple_p in zip(
        case_ids, d3_feats, simple_feats, d3_paths, simple_paths
    ):
        d3_bmu = som.winner(d3_f)
        simple_bmu = som.winner(simple_f)

        d3_cluster = int(labels[d3_bmu[0], d3_bmu[1]])
        simple_cluster = int(labels[simple_bmu[0], simple_bmu[1]])

        shift = float(np.hypot(
            d3_bmu[0] - simple_bmu[0],
            d3_bmu[1] - simple_bmu[1]
        ))

        rows.append({
            "case_id": cid,
            "d3_file": Path(d3_p).name,
            "simplified_file": Path(simple_p).name,
            "d3_bmu_x": d3_bmu[0],
            "d3_bmu_y": d3_bmu[1],
            "simplified_bmu_x": simple_bmu[0],
            "simplified_bmu_y": simple_bmu[1],
            "d3_cluster_id": d3_cluster,
            "d3_cluster": type_name(d3_cluster),
            "simplified_cluster_id": simple_cluster,
            "simplified_cluster": type_name(simple_cluster),
            "same_bmu": d3_bmu == simple_bmu,
            "same_cluster": d3_cluster == simple_cluster,
            "bmu_shift_distance": shift,
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. FIGURE 1: D3-BASE JOINT SOM
# ============================================================

def plot_joint_som_large_images(
    som, d3_feats, simple_feats, d3_paths, simple_paths,
    color_map, labels, save_path
):
    fig, ax = plt.subplots(figsize=(18, 18))

    for i in range(SOM_SIZE):
        for j in range(SOM_SIZE):
            ax.add_patch(Rectangle(
                (i, j), 1, 1,
                facecolor=color_map[i, j],
                edgecolor="white",
                linewidth=1
            ))

    cell_d3 = {}
    cell_simple = {}

    for idx, feat in enumerate(d3_feats):
        cell_d3.setdefault(som.winner(feat), []).append(idx)

    for idx, feat in enumerate(simple_feats):
        cell_simple.setdefault(som.winner(feat), []).append(idx)

    all_cells = set(cell_d3) | set(cell_simple)

    for x, y in all_cells:
        # D3 reference image, top-left
        if (x, y) in cell_d3:
            idx = cell_d3[(x, y)][0]
            try:
                img = Image.open(d3_paths[idx]).convert("RGB").resize(
                    (80, 80), Image.Resampling.LANCZOS
                )
                ab = AnnotationBbox(
                    OffsetImage(img, zoom=0.85),
                    (x + 0.32, y + 0.68),
                    frameon=True,
                    pad=0.01,
                    bboxprops=dict(edgecolor="#00E5FF", linewidth=4.5)
                )
                ax.add_artist(ab)
            except Exception:
                pass

        # Simplified image, bottom-right
        if (x, y) in cell_simple:
            idx = cell_simple[(x, y)][0]
            try:
                img = Image.open(simple_paths[idx]).convert("RGB").resize(
                    (80, 80), Image.Resampling.LANCZOS
                )
                ab = AnnotationBbox(
                    OffsetImage(img, zoom=0.85),
                    (x + 0.68, y + 0.32),
                    frameon=True,
                    pad=0.01,
                    bboxprops=dict(edgecolor="#FF00FF", linewidth=4.5)
                )
                ax.add_artist(ab)
            except Exception:
                pass

    for i in range(SOM_SIZE):
        for j in range(SOM_SIZE):
            txt = ax.text(
                i + 0.08, j + 0.92,
                type_name(labels[i, j]),
                ha="left", va="top",
                fontsize=9.5,
                fontweight="bold",
                color="white",
                zorder=20,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="black",
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.45
                )
            )
            txt.set_path_effects([
                path_effects.withStroke(linewidth=1.2, foreground="black")
            ])

    ax.set_xlim(0, SOM_SIZE)
    ax.set_ylim(0, SOM_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [
        Patch(
            facecolor="white", edgecolor="#00E5FF", linewidth=3,
            label="Full-prompt D3 (top-left)"
        ),
        Patch(
            facecolor="white", edgecolor="#FF00FF", linewidth=3,
            label="Simplified prompt (bottom-right)"
        ),
        Patch(facecolor="#388E3C", edgecolor="black", label="Type A"),
        Patch(facecolor="#1976D2", edgecolor="black", label="Type B"),
        Patch(facecolor="#FFA000", edgecolor="black", label="Type C"),
        Patch(facecolor="#D32F2F", edgecolor="black", label="Type D"),
        Patch(facecolor="#7CB342", edgecolor="black", label="Type E"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        fontsize=12,
        framealpha=0.95,
        edgecolor="black"
    )

    ax.set_title(
        "D3-base SOM: Full-prompt D3 vs Simplified-prompt Outputs",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 8. FIGURE 2: OCCUPANCY DIFFERENCE
# ============================================================

def plot_difference_heatmap(
    som, d3_feats, simple_feats, color_map, save_path
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    d3_heatmap = np.zeros((SOM_SIZE, SOM_SIZE))
    simple_heatmap = np.zeros((SOM_SIZE, SOM_SIZE))

    for feat in d3_feats:
        w = som.winner(feat)
        d3_heatmap[w[0], w[1]] += 1

    for feat in simple_feats:
        w = som.winner(feat)
        simple_heatmap[w[0], w[1]] += 1

    diff = simple_heatmap - d3_heatmap

    # D3 base occupancy
    ax = axes[0]
    for i in range(SOM_SIZE):
        for j in range(SOM_SIZE):
            ax.add_patch(Rectangle(
                (i, j), 1, 1,
                facecolor=color_map[i, j],
                edgecolor="white",
                linewidth=0.5
            ))
            if d3_heatmap[i, j] > 0:
                ax.text(
                    i + 0.5, j + 0.5,
                    f"{int(d3_heatmap[i, j])}",
                    ha="center", va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=3, foreground="black"
                        )
                    ]
                )
    ax.set_xlim(0, SOM_SIZE)
    ax.set_ylim(0, SOM_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Full-prompt D3 Base Distribution", fontweight="bold")

    # Simplified occupancy
    ax = axes[1]
    for i in range(SOM_SIZE):
        for j in range(SOM_SIZE):
            ax.add_patch(Rectangle(
                (i, j), 1, 1,
                facecolor=color_map[i, j],
                edgecolor="white",
                linewidth=0.5
            ))
            if simple_heatmap[i, j] > 0:
                ax.text(
                    i + 0.5, j + 0.5,
                    f"{int(simple_heatmap[i, j])}",
                    ha="center", va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=3, foreground="black"
                        )
                    ]
                )
    ax.set_xlim(0, SOM_SIZE)
    ax.set_ylim(0, SOM_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Simplified-prompt Distribution", fontweight="bold")

    # Difference
    ax = axes[2]
    vmax = max(abs(diff.min()), abs(diff.max()), 1)
    im = ax.imshow(
        diff.T,
        cmap="RdBu_r",
        origin="lower",
        vmin=-vmax,
        vmax=vmax
    )

    for i in range(SOM_SIZE):
        for j in range(SOM_SIZE):
            if diff[i, j] != 0:
                text_color = "white" if abs(diff[i, j]) > vmax * 0.4 else "black"
                stroke_color = "black" if text_color == "white" else "white"
                ax.text(
                    i, j,
                    f"{int(diff[i, j]):+d}",
                    ha="center", va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=text_color,
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=2.5,
                            foreground=stroke_color
                        )
                    ]
                )

    ax.set_title(
        "Difference (Simplified - D3)\n"
        "Red = simplified increase, Blue = simplified decrease",
        fontweight="bold"
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Change in node occupancy")

    plt.suptitle(
        "D3-base SOM: Distribution Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    return d3_heatmap, simple_heatmap, diff


# ============================================================
# 9. FIGURE 3: PAIRED MIGRATION
# ============================================================

def plot_filtered_migration(
    som, d3_feats, simple_feats, color_map, save_path,
    top_percent=50
):
    fig, ax = plt.subplots(figsize=(14, 14))

    for i in range(SOM_SIZE):
        for j in range(SOM_SIZE):
            ax.add_patch(Rectangle(
                (i, j), 1, 1,
                facecolor=color_map[i, j],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.85
            ))

    migrations = []
    for idx, (d3_feat, simple_feat) in enumerate(zip(d3_feats, simple_feats)):
        d3_bmu = som.winner(d3_feat)
        simple_bmu = som.winner(simple_feat)
        shift = float(np.hypot(
            d3_bmu[0] - simple_bmu[0],
            d3_bmu[1] - simple_bmu[1]
        ))
        migrations.append((d3_bmu, simple_bmu, shift, idx))

    migrations_sorted = sorted(
        migrations, key=lambda item: item[2], reverse=True
    )
    n_show = max(1, int(len(migrations_sorted) * top_percent / 100))
    shown = migrations_sorted[:n_show]

    threshold = shown[-1][2] if shown else 0
    max_shift = migrations_sorted[0][2] if migrations_sorted else 1
    if max_shift == 0:
        max_shift = 1

    rng = np.random.default_rng(SEED)

    for d3_bmu, simple_bmu, shift, _ in shown:
        if shift == 0:
            continue

        norm_shift = shift / max_shift
        linewidth = 0.5 + 2.0 * norm_shift
        jitter = (rng.random(2) - 0.5) * 0.15

        arrow = FancyArrowPatch(
            (
                d3_bmu[0] + 0.5 + jitter[0],
                d3_bmu[1] + 0.5 + jitter[1]
            ),
            (
                simple_bmu[0] + 0.5 + jitter[0],
                simple_bmu[1] + 0.5 + jitter[1]
            ),
            arrowstyle="-|>",
            mutation_scale=15,
            color=plt.cm.Reds(0.4 + 0.6 * norm_shift),
            linewidth=linewidth,
            alpha=0.85
        )
        ax.add_patch(arrow)

    for d3_bmu, _, shift, _ in migrations:
        if shift == 0:
            ax.add_patch(Circle(
                (d3_bmu[0] + 0.5, d3_bmu[1] + 0.5),
                0.18,
                color="#4CAF50",
                alpha=0.9,
                zorder=10,
                edgecolor="white",
                linewidth=1.5
            ))

    shifts = np.asarray([m[2] for m in migrations])
    no_shift_count = int(np.sum(shifts == 0))
    mean_shift = float(shifts.mean())

    ax.set_xlim(0, SOM_SIZE)
    ax.set_ylim(0, SOM_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="#4CAF50",
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=f"No BMU shift: {no_shift_count}/{len(migrations)}"
        ),
        Line2D([0], [0], color="#EF9A9A", linewidth=2, label="Medium shift"),
        Line2D([0], [0], color="#B71C1C", linewidth=4, label="Large shift"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=12,
        framealpha=0.95,
        edgecolor="black"
    )

    ax.set_title(
        "Pairwise SOM Migration: Full-prompt D3 → Simplified Prompt\n"
        f"Top {top_percent}% shifts shown; threshold ≥ {threshold:.1f}; "
        f"mean BMU shift = {mean_shift:.2f}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    return migrations


# ============================================================
# 10. FIGURE 4: CLUSTER DISTRIBUTION
# ============================================================

def plot_cluster_bar_chart(
    som, d3_feats, simple_feats, labels, save_path
):
    d3_cluster = np.zeros(N_CLUSTERS)
    simple_cluster = np.zeros(N_CLUSTERS)

    for feat in d3_feats:
        w = som.winner(feat)
        d3_cluster[labels[w[0], w[1]]] += 1

    for feat in simple_feats:
        w = som.winner(feat)
        simple_cluster[labels[w[0], w[1]]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    x = np.arange(N_CLUSTERS)
    width = 0.35

    ax.bar(
        x - width / 2,
        d3_cluster,
        width,
        label="Full-prompt D3",
        alpha=0.85,
        edgecolor="black"
    )
    ax.bar(
        x + width / 2,
        simple_cluster,
        width,
        label="Simplified prompt",
        alpha=0.85,
        edgecolor="black"
    )

    for i in range(N_CLUSTERS):
        ax.text(
            x[i] - width / 2,
            d3_cluster[i] + 0.8,
            f"{int(d3_cluster[i])}",
            ha="center",
            fontweight="bold"
        )
        ax.text(
            x[i] + width / 2,
            simple_cluster[i] + 0.8,
            f"{int(simple_cluster[i])}",
            ha="center",
            fontweight="bold"
        )

    ax.set_xlabel("D3-base SOM cluster")
    ax.set_ylabel("Sample count")
    ax.set_title("Cluster Distribution Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([type_name(i) for i in range(N_CLUSTERS)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    changes = simple_cluster - d3_cluster
    bars = ax.bar(
        np.arange(N_CLUSTERS),
        changes,
        alpha=0.85,
        edgecolor="black"
    )
    ax.axhline(0, color="black", linewidth=1.5)

    for bar, change in zip(bars, changes):
        y_offset = 0.8 if change >= 0 else -1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            change + y_offset,
            f"{change:+.0f}",
            ha="center",
            fontweight="bold"
        )

    ax.set_xlabel("D3-base SOM cluster")
    ax.set_ylabel("Change: simplified - D3")
    ax.set_title("Cluster Occupancy Shift", fontweight="bold")
    ax.set_xticks(np.arange(N_CLUSTERS))
    ax.set_xticklabels([type_name(i) for i in range(N_CLUSTERS)])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    return d3_cluster, simple_cluster


# ============================================================
# 11. FIGURE 5: SHIFT DISTRIBUTION
# ============================================================

def plot_shift_distribution(migrations, save_path):
    shifts = np.asarray([m[2] for m in migrations], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    max_shift = max(float(shifts.max()), 0.5)
    bins = np.arange(0, max_shift + 1.0, 0.5)
    ax.hist(shifts, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(
        shifts.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {shifts.mean():.2f}"
    )
    ax.axvline(
        np.median(shifts),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Median = {np.median(shifts):.2f}"
    )
    ax.set_xlabel("BMU shift distance")
    ax.set_ylabel("Count")
    ax.set_title("Shift Distance Histogram", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    no_shift = int(np.sum(shifts == 0))
    small = int(np.sum((shifts > 0) & (shifts <= 2)))
    medium = int(np.sum((shifts > 2) & (shifts <= 5)))
    large = int(np.sum(shifts > 5))

    sizes = [no_shift, small, medium, large]
    pie_labels = [
        f"No shift\n{no_shift}",
        f"Small (0,2]\n{small}",
        f"Medium (2,5]\n{medium}",
        f"Large >5\n{large}",
    ]
    ax.pie(
        sizes,
        labels=pie_labels,
        autopct="%1.0f%%",
        startangle=90
    )
    ax.set_title("Shift Categories", fontweight="bold")

    ax = axes[2]
    sorted_shifts = np.sort(shifts)
    cumulative = (
        np.arange(1, len(sorted_shifts) + 1) /
        len(sorted_shifts) * 100
    )
    ax.plot(sorted_shifts, cumulative, linewidth=2.5)
    ax.fill_between(sorted_shifts, cumulative, alpha=0.3)

    for pct in [25, 50, 75, 90]:
        value = float(np.percentile(shifts, pct))
        ax.axhline(pct, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(value, color="gray", linestyle=":", alpha=0.5)
        ax.plot(value, pct, "o")
        ax.annotate(f"P{pct}: {value:.1f}", (value + 0.2, pct + 2))

    ax.set_xlabel("BMU shift distance")
    ax.set_ylabel("Cumulative percentage (%)")
    ax.set_title("Cumulative Distribution", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    plt.suptitle(
        "D3 → Simplified-prompt Pairwise Shift Analysis",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 12. FIGURE 6: CLUSTER TRANSITION AND RETENTION
# ============================================================

def plot_cluster_flow_matrix(
    som, d3_feats, simple_feats, labels, save_path
):
    flow_matrix = np.zeros((N_CLUSTERS, N_CLUSTERS))

    for d3_feat, simple_feat in zip(d3_feats, simple_feats):
        d3_bmu = som.winner(d3_feat)
        simple_bmu = som.winner(simple_feat)

        d3_cluster = labels[d3_bmu[0], d3_bmu[1]]
        simple_cluster = labels[simple_bmu[0], simple_bmu[1]]
        flow_matrix[d3_cluster, simple_cluster] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(flow_matrix, cmap="YlOrRd")

    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            value = int(flow_matrix[i, j])
            if value > 0:
                text_color = (
                    "white"
                    if value > flow_matrix.max() * 0.5
                    else "black"
                )
                ax.text(
                    j, i, str(value),
                    ha="center", va="center",
                    fontsize=20,
                    fontweight="bold" if i == j else "normal",
                    color=text_color
                )

    ax.set_xticks(range(N_CLUSTERS))
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_xticklabels(
        [f"Simplified-{chr(65+i)}" for i in range(N_CLUSTERS)]
    )
    ax.set_yticklabels(
        [f"D3-{chr(65+i)}" for i in range(N_CLUSTERS)]
    )
    ax.set_xlabel("Simplified-prompt destination cluster")
    ax.set_ylabel("Full-prompt D3 source cluster")
    ax.set_title(
        "Cluster Transition Matrix\nDiagonal = retained",
        fontweight="bold"
    )
    plt.colorbar(im, ax=ax, shrink=0.8)

    retention = np.diag(flow_matrix)
    total_per_cluster = flow_matrix.sum(axis=1)
    retention_rate = np.divide(
        retention,
        total_per_cluster,
        out=np.zeros_like(retention),
        where=total_per_cluster > 0
    ) * 100

    overall_retention = (
        retention.sum() / max(flow_matrix.sum(), 1) * 100
    )
    valid_rates = retention_rate[total_per_cluster > 0]
    macro_retention = (
        valid_rates.mean() if len(valid_rates) else 0.0
    )

    ax = axes[1]
    bars = ax.bar(
        range(N_CLUSTERS),
        retention_rate,
        alpha=0.85,
        edgecolor="black"
    )

    for bar, rate, retained, total in zip(
        bars, retention_rate, retention, total_per_cluster
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{rate:.0f}%\n({int(retained)}/{int(total)})",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_xticks(range(N_CLUSTERS))
    ax.set_xticklabels([type_name(i) for i in range(N_CLUSTERS)])
    ax.set_xlabel("D3-base cluster")
    ax.set_ylabel("Retention rate (%)")
    ax.set_ylim(0, 100)
    ax.axhline(
        overall_retention,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall: {overall_retention:.1f}%"
    )
    ax.set_title(
        "Cluster Retention under Prompt Simplification\n"
        f"Macro-average = {macro_retention:.1f}%",
        fontweight="bold"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    return flow_matrix, retention_rate, overall_retention, macro_retention


# ============================================================
# 13. MAIN
# ============================================================

def main():
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Matching D3 and simplified-prompt cases...")
    case_ids, d3_paths, simple_paths = match_case_paths(
        D3_FOLDER,
        SIMPLIFIED_FOLDER,
        output_dir
    )

    print("=" * 70)
    print("Loading DINOv2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitl14"
    )
    model.to(device).eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    print("=" * 70)
    print("Extracting matched D3 features...")
    d3_feats, valid_d3_paths = extract_features(
        d3_paths, model, transform, device, "D3"
    )

    print("Extracting matched simplified-prompt features...")
    simple_feats, valid_simple_paths = extract_features(
        simple_paths, model, transform, device, "Simplified"
    )

    # Feature-extraction failures would break pair alignment.
    if len(valid_d3_paths) != len(case_ids) or len(valid_simple_paths) != len(case_ids):
        raise RuntimeError(
            "One or more matched images failed during feature extraction. "
            "Fix the failed files and rerun so pair alignment remains valid."
        )

    print(f"Matched feature pairs: {len(case_ids)}")

    # --------------------------------------------------------
    # Train the SOM ONLY on full-prompt D3 features
    # --------------------------------------------------------
    print("=" * 70)
    print(
        f"Training D3-base SOM on {len(d3_feats)} D3 images only..."
    )

    som = MiniSom(
        SOM_SIZE,
        SOM_SIZE,
        d3_feats.shape[1],
        sigma=1.5,
        learning_rate=0.5,
        random_seed=SEED
    )
    som.pca_weights_init(d3_feats)
    som.train_random(
        d3_feats,
        num_iteration=SOM_ITERATIONS,
        verbose=True
    )

    color_map, labels = get_colormap(som, N_CLUSTERS)

    # Assignments table
    assignment_df = build_assignment_table(
        case_ids,
        som,
        d3_feats,
        simple_feats,
        valid_d3_paths,
        valid_simple_paths,
        labels
    )
    assignment_df.to_csv(
        output_dir / "d3_vs_simplified_assignments.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 70)
    print("Generating figures...")

    plot_joint_som_large_images(
        som, d3_feats, simple_feats,
        valid_d3_paths, valid_simple_paths,
        color_map, labels,
        output_dir / "d3base_joint_som_large.png"
    )

    d3_heatmap, simple_heatmap, diff = plot_difference_heatmap(
        som, d3_feats, simple_feats,
        color_map,
        output_dir / "d3base_difference.png"
    )

    migrations = plot_filtered_migration(
        som, d3_feats, simple_feats,
        color_map,
        output_dir / "d3base_migration_filtered.png",
        top_percent=MIGRATION_TOP_PERCENT
    )

    d3_cluster, simple_cluster = plot_cluster_bar_chart(
        som, d3_feats, simple_feats,
        labels,
        output_dir / "d3base_cluster_bars.png"
    )

    plot_shift_distribution(
        migrations,
        output_dir / "d3base_shift_analysis.png"
    )

    (
        flow_matrix,
        retention_rate,
        overall_retention,
        macro_retention
    ) = plot_cluster_flow_matrix(
        som, d3_feats, simple_feats,
        labels,
        output_dir / "d3base_cluster_flow.png"
    )

    pd.DataFrame(
        flow_matrix,
        index=[f"D3_{type_name(i)}" for i in range(N_CLUSTERS)],
        columns=[
            f"Simplified_{type_name(i)}"
            for i in range(N_CLUSTERS)
        ]
    ).to_csv(
        output_dir / "cluster_transition_matrix.csv",
        encoding="utf-8-sig"
    )

    shifts = assignment_df["bmu_shift_distance"].to_numpy()

    summary_df = pd.DataFrame([{
        "n_matched_cases": len(case_ids),
        "som_training_set": "Full-prompt D3 only",
        "som_size": f"{SOM_SIZE}x{SOM_SIZE}",
        "n_clusters": N_CLUSTERS,
        "mean_bmu_shift": float(shifts.mean()),
        "median_bmu_shift": float(np.median(shifts)),
        "no_bmu_shift_n": int(np.sum(shifts == 0)),
        "no_bmu_shift_pct": float(np.mean(shifts == 0) * 100),
        "same_cluster_n": int(assignment_df["same_cluster"].sum()),
        "overall_cluster_retention_pct": float(overall_retention),
        "macro_cluster_retention_pct": float(macro_retention),
        "d3_cluster_counts": "|".join(
            str(int(v)) for v in d3_cluster
        ),
        "simplified_cluster_counts": "|".join(
            str(int(v)) for v in simple_cluster
        ),
    }])

    summary_df.to_csv(
        output_dir / "som_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("D3-base SOM comparison summary")
    print("=" * 70)
    print(f"Matched cases:               {len(case_ids)}")
    print(f"Mean BMU shift:              {shifts.mean():.2f}")
    print(f"Median BMU shift:            {np.median(shifts):.2f}")
    print(
        f"No BMU shift:                "
        f"{np.sum(shifts == 0)}/{len(shifts)} "
        f"({np.mean(shifts == 0) * 100:.1f}%)"
    )
    print(
        f"Overall cluster retention:   "
        f"{overall_retention:.1f}%"
    )
    print(
        f"Macro cluster retention:     "
        f"{macro_retention:.1f}%"
    )

    for i, rate in enumerate(retention_rate):
        print(f"  {type_name(i)} retention: {rate:.1f}%")

    print("\nAll outputs saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
