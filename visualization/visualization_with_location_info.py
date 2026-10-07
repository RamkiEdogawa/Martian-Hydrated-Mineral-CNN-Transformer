# -*- coding: utf-8 -*-
"""
visualization_with_location_info.py

Automated CRISM Hyperspectral Mineral Mapping & Geospatial Metadata Extraction Pipeline:
1. Recursively traverses input directories to batch process infrared CRISM cubes (*if*.hdr).
2. Parses PDS metadata labels (*.lbl) to extract geographic bounding coordinates (Min/Max Lat/Lon).
3. Renders a standalone, publication-ready global mineral color key grid.
4. Executes deep mineral classification using the Hybrid CNN-Transformer framework.
5. Employs broadband median luminance modulation with 1:1 physical aspect ratio locking.
6. Exports borderless, transparent true-pixel thematic PNGs (pad_inches=0.0) for GIS mosaicking.
7. Aggregates scene footprint coordinates into a unified geospatial summary CSV.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spectral
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standardize academic font styling across all Matplotlib outputs
plt.rcParams["font.family"] = "Arial"


# ==============================================================================
# Path Configurations (Relative to project root for seamless reproducibility)
# ==============================================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "weights" / "best_model.pt"
)
DEFAULT_INPUT_DIR = Path(os.environ.get("CRISM_INPUT_DIR", PROJECT_ROOT / "data" / "demo"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("CRISM_OUTPUT_DIR", PROJECT_ROOT / "data" / "mapping_results_geo"))

# Mineral Abundance Ranking Filter Configuration
TARGET_START_RANK = 2
TARGET_END_RANK = 5

# Preprocessing Thresholds
REFLECTANCE_MIN_THRESH = 0.005
SPECTRUM_DIV_THRESH = 0.003
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Model Architecture
# ==============================================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class LocalCNNBranch(nn.Module):
    """Hierarchical 1D-CNN branch extracting local spectral absorption features."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x.unsqueeze(1)).squeeze(-1))


class GlobalTransformerBranch(nn.Module):
    """Transformer encoder branch capturing long-range inter-band dependencies."""

    def __init__(
        self,
        input_dim: int = 256,
        patch_size: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        ff_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = input_dim // patch_size
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_tokens + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.patch_embed(x.view(b, self.num_tokens, self.patch_size))
        x = self.pos_enc(torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1))
        return self.fc(self.norm(self.encoder(x))[:, 0, :])


class FeatureFusion(nn.Module):
    """Non-linear projection fusing local and global representation spaces."""

    def __init__(self, loc_dim: int = 128, glo_dim: int = 128, fused_dim: int = 128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(loc_dim + glo_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, z_loc: torch.Tensor, z_glo: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([z_loc, z_glo], dim=1))


class HybridMineralNet(nn.Module):
    """Unified CNN-Transformer network for Martian hyperspectral mineral identification."""

    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.local_branch = LocalCNNBranch(128)
        self.global_branch = GlobalTransformerBranch(
            input_dim=256,
            patch_size=8,
            d_model=128,
            nhead=8,
            num_layers=2,
            ff_dim=256,
            out_dim=128,
        )
        self.fusion = FeatureFusion(128, 128, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fusion(self.local_branch(x), self.global_branch(x)))


# Calibrated RGB Color Palette for 15 Target Mineral Classes
MINERAL_COLORS: Dict[int, List[int]] = {
    0: [140, 67, 46],
    1: [0, 0, 255],
    2: [255, 100, 0],
    3: [0, 255, 200],
    4: [164, 75, 155],
    5: [101, 174, 255],
    6: [118, 254, 254],
    7: [60, 91, 112],
    8: [255, 255, 0],
    9: [255, 255, 255],
    10: [255, 0, 255],
    11: [100, 0, 255],
    12: [0, 200, 254],
    13: [0, 255, 0],
    14: [171, 175, 80],
}


# ==============================================================================
# Cartographic & Geospatial Metadata Extraction Helpers
# ==============================================================================
def generate_and_save_global_legend(class_names: List[str], output_path: Path) -> None:
    """Renders an independent, grid-aligned academic color bar suitable for manuscript figures."""
    num_classes = len(class_names)
    cols = 4
    rows = math.ceil(num_classes / cols)

    fig, ax = plt.subplots(figsize=(10, rows * 0.5), dpi=300)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    for i, name in enumerate(class_names):
        if i not in MINERAL_COLORS:
            continue
        r = rows - 1 - (i // cols)
        c = i % cols
        color = np.array(MINERAL_COLORS[i]) / 255.0

        # Draw categorical colored patch
        rect = plt.Rectangle((c + 0.1, r + 0.1), 0.2, 0.6, facecolor=color, edgecolor="none")
        ax.add_patch(rect)
        # Class taxonomy text annotation
        ax.text(c + 0.35, r + 0.4, name, va="center", ha="left", fontsize=10, fontname="Arial")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"🌍 [Export] Standalone academic global legend saved to: {output_path.name}")


def extract_geo_range_from_lbl(lbl_path: Path) -> Optional[Dict[str, Any]]:
    """Parse Martian geographical coordinate boundaries from PDS label (.lbl) headers."""
    if not lbl_path.exists():
        return None

    try:
        content = lbl_path.read_text(encoding="utf-8", errors="ignore")

        def find_val(pattern: str) -> Optional[float]:
            match = re.search(pattern, content)
            return float(match.group(1)) if match else None

        min_lat = find_val(r"MINIMUM_LATITUDE\s*=\s*([-\d\.]+)")
        max_lat = find_val(r"MAXIMUM_LATITUDE\s*=\s*([-\d\.]+)")
        west_lon = find_val(r"WESTERNMOST_LONGITUDE\s*=\s*([-\d\.]+)")
        east_lon = find_val(r"EASTERNMOST_LONGITUDE\s*=\s*([-\d\.]+)")

        if None in (min_lat, max_lat, west_lon, east_lon):
            return None

        return {
            "MIN_LAT": min_lat,
            "MAX_LAT": max_lat,
            "WEST_LON": west_lon,
            "EAST_LON": east_lon,
            "CENTER_LAT": round((min_lat + max_lat) / 2.0, 6),
            "CENTER_LON": round((west_lon + east_lon) / 2.0, 6),
        }
    except Exception:
        return None


# ==============================================================================
# Main Mapping Pipeline Execution
# ==============================================================================
def run_pipeline(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    target_start_rank: int = TARGET_START_RANK,
    target_end_rank: int = TARGET_END_RANK,
) -> None:
    """Execute end-to-end CRISM mapping and geospatial summary logging."""
    print("=" * 80)
    print("🚀 Initializing CRISM Mineral Mapping & Geospatial Summary Pipeline...")
    print(f"📦 Computing Device   : {DEVICE}")
    print(f"🔬 Checkpoint Path    : {model_path}")
    print(f"📁 Recursive Scan Root: {input_dir}")
    print(f"💾 Output Directory   : {output_dir}")
    print(f"🎯 Target Ranks Filter: Rank {target_start_rank} to Rank {target_end_rank}")
    print("=" * 80 + "\n")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for infrared CRISM image headers (*if*.hdr) recursively
    hdr_files = [f for f in input_dir.rglob("*.hdr") if "if" in f.name.lower()]
    total_files = len(hdr_files)

    if total_files == 0:
        print(f"❌ [Error] No valid infrared CRISM (*if*.hdr) files located under: {input_dir}")
        return

    # Load neural network checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names", [f"Mineral_{i}" for i in range(15)])
    model = HybridMineralNet(num_classes=15).to(DEVICE).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Pre-render standalone publication legend
    generate_and_save_global_legend(class_names, output_dir / "00_GLOBAL_MINERAL_LEGEND.png")

    geo_summary_records: List[Dict[str, Any]] = []

    print(f"🔎 Discovered {total_files} CRISM hyperspectral cubes. Starting batch execution...\n")

    for current_idx, crism_hdr_path in enumerate(hdr_files, 1):
        img_stem_name = crism_hdr_path.stem
        print(f" -> [{current_idx:03d}/{total_files:03d}] 📄 Processing: {img_stem_name:<30s} ... ", end="", flush=True)

        # Parse and log geospatial bounding coordinates
        lbl_file_path = crism_hdr_path.with_suffix(".lbl")
        geo_info = extract_geo_range_from_lbl(lbl_file_path)
        if geo_info:
            geo_info["CRISM_ID"] = img_stem_name
            geo_summary_records.append(geo_info)

        try:
            img_open = spectral.open_image(str(crism_hdr_path))
            img_wavelengths = np.array(
                img_open.metadata.get("wavelength", np.linspace(1021.0, 2635.0, img_open.shape[2])),
                dtype=np.float32,
            )
            if img_wavelengths.max() < 10.0:
                img_wavelengths *= 1000.0  # Convert μm to nm if necessary

            cube_data = img_open.load()
            h, w, b = cube_data.shape

            # Extract continuum basemap using broadband median filtering
            clean_band_mask = (img_wavelengths >= 1200.0) & (img_wavelengths <= 1400.0)
            if clean_band_mask.sum() > 0:
                base_map_raw = np.nanmedian(np.asarray(cube_data[:, :, clean_band_mask]), axis=2)
            else:
                clean_band_idx = np.argmin(np.abs(img_wavelengths - 1300.0))
                base_map_raw = np.asarray(cube_data[:, :, clean_band_idx])
            base_map_raw = np.squeeze(base_map_raw)

            # Establish observational swath valid mask, removing dead pixels (>=65000) and NaNs
            true_swath_mask = (
                np.isfinite(base_map_raw) & (base_map_raw < 65000.0) & (base_map_raw > -999.0)
            )

            # Percentile contrast normalization for topographic background
            base_map_normalized = np.zeros((h, w), dtype=np.float32)
            if true_swath_mask.any():
                b_lo = np.percentile(base_map_raw[true_swath_mask], 2.0)
                b_hi = np.percentile(base_map_raw[true_swath_mask], 98.0)
                base_map_clipped = np.clip(base_map_raw, b_lo, b_hi)
                base_map_normalized[true_swath_mask] = (
                    base_map_clipped[true_swath_mask] - b_lo
                ) / (b_hi - b_lo + 1e-8)

            flattened_cube = cube_data.reshape(-1, b)
            total_pixels = flattened_cube.shape[0]

            valid_band_mask = (img_wavelengths >= 1000.0) & (img_wavelengths <= 2650.0)
            cropped_wl = img_wavelengths[valid_band_mask]
            model_target_wl = np.linspace(1021.0, 2635.0, 256, dtype=np.float32)

            resampled_dataset, pixel_is_mineral_mask = [], []
            true_swath_flat = true_swath_mask.flatten()

            for i in range(total_pixels):
                if not true_swath_flat[i]:
                    resampled_dataset.append(np.zeros(256, dtype=np.float32))
                    pixel_is_mineral_mask.append(False)
                    continue

                spec = flattened_cube[i].copy()
                spec_crop = spec[valid_band_mask]
                mask_finite = np.isfinite(spec_crop)

                if not mask_finite.any():
                    resampled_dataset.append(np.zeros(256, dtype=np.float32))
                    pixel_is_mineral_mask.append(False)
                    continue
                elif not mask_finite.all():
                    idx_arr = np.arange(len(spec_crop))
                    spec_crop[~mask_finite] = np.interp(
                        idx_arr[~mask_finite], idx_arr[mask_finite], spec_crop[mask_finite]
                    )

                lo = np.percentile(spec_crop, 1.0)
                hi = np.percentile(spec_crop, 99.0)
                mean_val = np.mean(spec_crop)

                if (hi - lo < SPECTRUM_DIV_THRESH) or (mean_val < REFLECTANCE_MIN_THRESH):
                    resampled_dataset.append(np.zeros(256, dtype=np.float32))
                    pixel_is_mineral_mask.append(False)
                else:
                    norm_spec = (spec_crop - lo) / (hi - lo + 1e-8)
                    resampled_dataset.append(
                        np.interp(model_target_wl, cropped_wl, norm_spec).astype(np.float32)
                    )
                    pixel_is_mineral_mask.append(True)

            resampled_dataset = np.array(resampled_dataset, dtype=np.float32)

            # Forward inference
            predicted_labels = np.zeros(total_pixels, dtype=np.int64)
            mineral_pixel_indices = np.where(pixel_is_mineral_mask)[0]

            if len(mineral_pixel_indices) > 0:
                valid_inputs = resampled_dataset[mineral_pixel_indices]
                valid_preds = []
                for idx in range(0, len(valid_inputs), BATCH_SIZE):
                    batch_data = valid_inputs[idx : idx + BATCH_SIZE]
                    inp_tensor = torch.from_numpy(batch_data).float().to(DEVICE)
                    with torch.no_grad():
                        logits = model(inp_tensor)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        valid_preds.extend(preds)

                valid_preds = np.array(valid_preds)

                unique_classes = np.unique(valid_preds)
                class_area_dict = {c: (valid_preds == c).sum() for c in unique_classes}
                sorted_classes = sorted(
                    class_area_dict.keys(), key=lambda k: class_area_dict[k], reverse=True
                )

                selected_classes = sorted_classes[target_start_rank - 1 : target_end_rank]
                valid_preds[~np.isin(valid_preds, selected_classes)] = 15
                predicted_labels[mineral_pixel_indices] = valid_preds

            predicted_labels[~true_swath_flat] = 15
            classification_map = predicted_labels.reshape(h, w)

            # Reconstruct 4-channel RGBA matrix
            fused_rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
            fused_rgba_image[true_swath_mask, 0] = (base_map_normalized[true_swath_mask] * 255).astype(np.uint8)
            fused_rgba_image[true_swath_mask, 1] = (base_map_normalized[true_swath_mask] * 255).astype(np.uint8)
            fused_rgba_image[true_swath_mask, 2] = (base_map_normalized[true_swath_mask] * 255).astype(np.uint8)
            fused_rgba_image[true_swath_mask, 3] = 255
            fused_rgba_image[~true_swath_mask, 3] = 0  # Dead observational swath is transparent

            for idx, color in MINERAL_COLORS.items():
                mask = (classification_map == idx) & true_swath_mask
                if mask.sum() == 0:
                    continue
                brightness_factor = 0.4 + 0.6 * base_map_normalized[mask]
                for c in range(3):
                    fused_rgba_image[mask, c] = np.clip(color[c] * brightness_factor, 0, 255).astype(np.uint8)
                fused_rgba_image[mask, 3] = 255

            # Export classification array to CSV
            matrix_csv_path = output_dir / f"{img_stem_name}_mapping_output.csv"
            pd.DataFrame(classification_map).to_csv(matrix_csv_path, index=False, header=False)

            # Render borderless, pure-pixel image (ideal for seamless manuscript layout integration)
            fig = plt.figure(figsize=(9, 11))
            ax = plt.axes([0, 0, 1, 1])
            ax.imshow(fused_rgba_image, aspect="equal")
            ax.axis("off")

            png_out_path = output_dir / f"{img_stem_name}.png"
            plt.savefig(png_out_path, dpi=300, bbox_inches="tight", pad_inches=0.0, transparent=True)
            plt.close(fig)

            print("✅ Done")

        except Exception as e:
            print(f"❌ Failed ({e})")
            continue

    # Export consolidated geographic range table
    if geo_summary_records:
        df_geo = pd.DataFrame(geo_summary_records)
        df_geo = df_geo[["CRISM_ID", "MIN_LAT", "MAX_LAT", "WEST_LON", "EAST_LON", "CENTER_LAT", "CENTER_LON"]]
        summary_csv_path = output_dir / "00_CRISM_GEO_RANGE_SUMMARY.csv"
        df_geo.to_csv(summary_csv_path, index=False)
        print(f"\n🌍 [Export] Geospatial coordinate boundaries written to: {summary_csv_path.name}")

    print("\n" + "=" * 80)
    print("🎯 Batch mapping pipeline completed successfully!")
    print(f"📁 Destination Directory : {output_dir}")
    print("=" * 80 + "\n")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CRISM Hyperspectral Mineral Mapping and Geospatial Summary Pipeline."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained PyTorch model checkpoint (.pt).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root directory for recursively scanning CRISM hyperspectral cubes (*if*.hdr).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output mapping CSVs, figures, and geospatial summary.",
    )
    parser.add_argument(
        "--start-rank",
        type=int,
        default=TARGET_START_RANK,
        help="Starting abundance rank to include in visualization (default: 6).",
    )
    parser.add_argument(
        "--end-rank",
        type=int,
        default=TARGET_END_RANK,
        help="Ending abundance rank to include in visualization (default: 6).",
    )

    args = parser.parse_args()

    run_pipeline(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_start_rank=args.start_rank,
        target_end_rank=args.end_rank,
    )


if __name__ == "__main__":
    main()