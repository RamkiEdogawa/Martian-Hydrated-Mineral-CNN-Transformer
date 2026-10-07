# -*- coding: utf-8 -*-
"""
demo.py

Interactive Demonstration for CRISM Hyperspectral Mineral Mapping:
1. Loads an input CRISM hyperspectral cube and pretrained Hybrid CNN-Transformer weights.
2. Extracts a 3D topographic basemap via broadband median filtering (1200 - 1400 nm).
3. Executes batch forward inference on all valid observational pixels.
4. Computes spatial abundance rankings (pixel area and coverage percentage).
5. Displays detailed mineralogical statistics directly in the console.
6. Renders the interactive thematic mineral map window (locking 1:1 physical aspect ratio).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import spectral
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Path Configurations & Defaults (Relative to project root)
# ==============================================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))

DEFAULT_MODEL_PATH = PROJECT_ROOT / "weights" / "best_model.pt"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "demo"

# Abundance Ranking Filter (Filters ubiquitous surface dust Rank 1; shows Ranks 2-5)
TARGET_START_RANK = 2
TARGET_END_RANK = 5

# Preprocessing Physical Thresholds
REFLECTANCE_MIN_THRESH = 0.005
SPECTRUM_DIV_THRESH = 0.003
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Model Architecture
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LocalCNNBranch(nn.Module):
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
        assert input_dim % patch_size == 0
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


# Standardized 15-class RGB palette
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
# Interactive Demo Pipeline
# ==============================================================================
def run_demo(
    model_path: Path,
    input_dir: Path,
    target_start_rank: int = TARGET_START_RANK,
    target_end_rank: int = TARGET_END_RANK,
) -> None:
    print("=" * 80)
    print("🚀 Running Interactive CRISM Mineral Mapping Demonstration")
    print(f"📦 Computing Device   : {DEVICE}")
    print(f"🔬 Checkpoint Path    : {model_path}")
    print(f"📁 Target Scan Root   : {input_dir}")
    print(f"🎯 Display Rank Range : Rank {target_start_rank} to Rank {target_end_rank}")
    print("=" * 80 + "\n")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Recursively locate CRISM infrared cubes (*if*.hdr)
    hdr_files = [f for f in input_dir.rglob("*.hdr") if "if" in f.name.lower()]
    if not hdr_files:
        hdr_files = list(input_dir.rglob("*.hdr"))  # Fallback to any available HDR
        if not hdr_files:
            raise FileNotFoundError(f"No valid CRISM (.hdr) files discovered under: {input_dir}")

    print(f"🔎 Located {len(hdr_files)} hyperspectral cube(s). Processing demo scene...\n")

    # Load neural network checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names", [f"Mineral_{i}" for i in range(15)])
    model = HybridMineralNet(num_classes=15).to(DEVICE).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Process each scene and display interactively
    for current_idx, crism_hdr_path in enumerate(hdr_files, 1):
        img_stem_name = crism_hdr_path.stem
        print(f"\n[{current_idx:02d}/{len(hdr_files):02d}] 📄 Analyzing Scene: {img_stem_name}")
        print("-" * 80)

        img_open = spectral.open_image(str(crism_hdr_path))
        img_wavelengths = np.array(
            img_open.metadata.get("wavelength", np.linspace(1021.0, 2635.0, img_open.shape[2])),
            dtype=np.float32,
        )
        if img_wavelengths.max() < 10.0:
            img_wavelengths *= 1000.0  # Convert μm to nm

        cube_data = img_open.load()
        h, w, b = cube_data.shape
        print(f"  • Dimensions : Height={h}, Width={w}, Bands={b}")

        # Extract broadband median terrain basemap (1200 - 1400 nm)
        clean_band_mask = (img_wavelengths >= 1200.0) & (img_wavelengths <= 1400.0)
        if clean_band_mask.sum() > 0:
            base_map_raw = np.nanmedian(np.asarray(cube_data[:, :, clean_band_mask]), axis=2)
        else:
            clean_band_idx = np.argmin(np.abs(img_wavelengths - 1300.0))
            base_map_raw = np.asarray(cube_data[:, :, clean_band_idx])
        base_map_raw = np.squeeze(base_map_raw)

        # Observation swath valid mask
        true_swath_mask = (
            np.isfinite(base_map_raw) & (base_map_raw < 65000.0) & (base_map_raw > -999.0)
        )

        # Topographic contrast normalization
        base_map_normalized = np.zeros((h, w), dtype=np.float32)
        if true_swath_mask.any():
            b_lo = np.percentile(base_map_raw[true_swath_mask], 2.0)
            b_hi = np.percentile(base_map_raw[true_swath_mask], 98.0)
            base_map_clipped = np.clip(base_map_raw, b_lo, b_hi)
            base_map_normalized[true_swath_mask] = (
                base_map_clipped[true_swath_mask] - b_lo
            ) / (b_hi - b_lo + 1e-8)

        # Spectral resampling & preprocessing
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

        # Batch forward inference
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
            total_mineral_pixels = len(valid_preds)

            # Area ranking calculation
            unique_classes = np.unique(valid_preds)
            class_area_dict = {c: int((valid_preds == c).sum()) for c in unique_classes}
            sorted_classes = sorted(
                class_area_dict.keys(), key=lambda k: class_area_dict[k], reverse=True
            )

            # Select targeted abundance ranks
            selected_classes = sorted_classes[target_start_rank - 1 : target_end_rank]

            # Output comprehensive mineral area statistics directly to console
            print("\n📊 Mineral Spatial Area & Abundance Ranking:")
            print(f"  Total Valid Mineral Pixels: {total_mineral_pixels:,}")
            print(f"  Display Filter : Rank {target_start_rank} to Rank {target_end_rank}\n")
            print(f"  {'Rank':<6}{'Status':<16}{'Mineral Name':<24}{'Pixel Area':<14}{'Coverage (%)'}")
            print("  " + "-" * 68)

            for rank, c in enumerate(sorted_classes, 1):
                pixel_cnt = class_area_dict[c]
                percentage = (pixel_cnt / total_mineral_pixels) * 100.0
                status = "✅ [Displayed]" if c in selected_classes else "❌ [Filtered]"
                print(f"  {rank:<6}{status:<16}{class_names[c]:<24}{pixel_cnt:<14,}{percentage:>6.2f}%")

            print("  " + "-" * 68 + "\n")

            valid_preds[~np.isin(valid_preds, selected_classes)] = 15
            predicted_labels[mineral_pixel_indices] = valid_preds

        predicted_labels[~true_swath_flat] = 15
        classification_map = predicted_labels.reshape(h, w)

        # 4-channel RGBA matrix construction with 3D luminance modulation
        fused_rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
        fused_rgba_image[true_swath_mask, 0] = (base_map_normalized[true_swath_mask] * 255).astype(np.uint8)
        fused_rgba_image[true_swath_mask, 1] = (base_map_normalized[true_swath_mask] * 255).astype(np.uint8)
        fused_rgba_image[true_swath_mask, 2] = (base_map_normalized[true_swath_mask] * 255).astype(np.uint8)
        fused_rgba_image[true_swath_mask, 3] = 255
        fused_rgba_image[~true_swath_mask, 3] = 0  # Fully transparent dead background

        for idx, color in MINERAL_COLORS.items():
            mask = (classification_map == idx) & true_swath_mask
            if mask.sum() == 0:
                continue
            brightness_factor = 0.4 + 0.6 * base_map_normalized[mask]
            for c in range(3):
                fused_rgba_image[mask, c] = np.clip(color[c] * brightness_factor, 0, 255).astype(np.uint8)
            fused_rgba_image[mask, 3] = 255

        # Render interactive figure window
        fig = plt.figure(figsize=(9, 11))
        plt.imshow(fused_rgba_image, aspect="equal")
        plt.title(f"Interactive Mapping Demo: {img_stem_name}", fontsize=12, pad=12)
        plt.axis("off")

        present_classes = [c for c in np.unique(classification_map) if c != 15]
        patches = [
            plt.plot(
                [],
                [],
                marker="s",
                ms=10,
                ls="",
                color=np.array(MINERAL_COLORS[c]) / 255.0,
                label=f"{class_names[c]} ({class_area_dict[c]:,} px)",
            )[0]
            for c in present_classes
        ]
        if patches:
            plt.legend(
                handles=patches,
                loc="upper right",
                bbox_to_anchor=(1.40, 1.0),
                fontsize=9,
                title="Classified Minerals (Area)",
            )

        print("🖼️ Displaying interactive visualization window (close window to proceed)...")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    print("\n" + "=" * 80)
    print("🎯 Demonstration completed successfully!")
    print("=" * 80)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Interactive quickstart demo for CRISM hyperspectral mineral mapping."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the target CRISM hyperspectral cube (*if*.hdr).",
    )
    parser.add_argument(
        "--start-rank",
        type=int,
        default=TARGET_START_RANK,
        help="Starting abundance rank to display (default: 2, filtering rank-1 dust).",
    )
    parser.add_argument(
        "--end-rank",
        type=int,
        default=TARGET_END_RANK,
        help="Ending abundance rank to display (default: 5).",
    )

    args = parser.parse_args()

    run_demo(
        model_path=args.model_path,
        input_dir=args.input_dir,
        target_start_rank=args.start_rank,
        target_end_rank=args.end_rank,
    )


if __name__ == "__main__":
    main()