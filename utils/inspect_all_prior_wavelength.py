# -*- coding: utf-8 -*-
"""
predict_final_all.py

Batch Hyperspectral Mineral Inference Pipeline:
1. Scans and parses all CRISM spectra (*.csv) within the target directory.
2. Crops and resamples spectra to standardized target spectral bands (1021.0 - 2635.0 nm, 256 dimensions).
3. Executes forward inference using the unified Hybrid CNN-Transformer checkpoint.
4. Generates an individual Top-5 probability log (*_prediction.txt) for each file.
5. Aggregates complete multi-sample mineralogical predictions into a consolidated summary CSV.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Path Configurations (Relative to project root for seamless reproducibility)
# ==============================================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

MODEL_PATH = PROJECT_ROOT / "data" / "final_unified_hybrid_transformer_allpriors" / "best_model.pt"
CSV_DIR = PROJECT_ROOT / "data" / "final_unified_hybrid_transformer_allpriors"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target Spectral Calibration Parameters
TARGET_MIN_NM = 1021.0
TARGET_MAX_NM = 2635.0
TARGET_DIM = 256

SUMMARY_CSV_NAME = "all_frt_prediction.csv"


# ==============================================================================
# Class Taxonomy
# ==============================================================================
GLOBAL_CLASS_ORDER = [
    "Analcime",
    "Bassanite",
    "Chlorite",
    "Epidote",
    "Fe-Olivine",
    "High-Ca Pyroxene",
    "Illite/Muscovite",
    "Low-Ca Pyroxene",
    "Margarite",
    "Mg-Carbonate",
    "Mg-Smectite",
    "Monohydrated sulfate",
    "Plagioclase",
    "Prehnite",
    "Serpentine",
]


# ==============================================================================
# Numerical & Spectral Utility Functions
# ==============================================================================
def smart_torch_load(path: Path) -> Any:
    """Safely load serialized PyTorch objects handling cross-version differences."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fix_nonfinite_1d(x: np.ndarray) -> np.ndarray:
    """Impute non-finite values (NaN/Inf) via linear 1D interpolation."""
    x = np.asarray(x, dtype=np.float32).copy()
    mask = np.isfinite(x)
    if mask.all():
        return x
    if not mask.any():
        return np.zeros_like(x, dtype=np.float32)

    idx = np.arange(len(x))
    x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return x


def robust_minmax_1d(x: np.ndarray, low_p: float = 1.0, high_p: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    """Robust percentile-based min-max scaling to suppress spectral outlier spikes."""
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, low_p)
    hi = np.percentile(x, high_p)

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo < eps):
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    return x.astype(np.float32)


def resample_to_target_len(x: np.ndarray, target_len: int = 256) -> np.ndarray:
    """Resample 1D spectral curve to the target dimension using linear interpolation."""
    x = np.asarray(x, dtype=np.float32)
    src_len = len(x)
    if src_len == target_len:
        return x.copy()

    src_grid = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    tgt_grid = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(tgt_grid, src_grid, x).astype(np.float32)


def preprocess_spectrum_for_model(x: np.ndarray) -> np.ndarray:
    """Standardized preprocessing pipeline: NaN imputation -> Min-Max scaling -> Resampling."""
    x = fix_nonfinite_1d(x)
    x = robust_minmax_1d(x, low_p=1.0, high_p=99.0, eps=1e-8)
    x = resample_to_target_len(x, TARGET_DIM)
    return x.astype(np.float32)


def crop_by_wavelength(
    wavelength_nm: np.ndarray,
    reflectance: np.ndarray,
    wl_min: float = 1021.0,
    wl_max: float = 2635.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate spectral curves to the target diagnostic infrared window."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float32)
    reflectance = np.asarray(reflectance, dtype=np.float32)

    mask = (wavelength_nm >= wl_min) & (wavelength_nm <= wl_max)
    if mask.sum() < 5:
        raise ValueError(f"Insufficient valid data points within {wl_min}-{wl_max} nm for inference.")

    return wavelength_nm[mask], reflectance[mask]


def resample_by_wavelength(
    wavelength_nm: np.ndarray,
    reflectance: np.ndarray,
    wl_min: float = 1021.0,
    wl_max: float = 2635.0,
    target_len: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly resample reflectance spectrum across a calibrated wavelength grid."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float32)
    reflectance = np.asarray(reflectance, dtype=np.float32)

    tgt_wl = np.linspace(wl_min, wl_max, target_len, dtype=np.float32)
    tgt_ref = np.interp(tgt_wl, wavelength_nm, reflectance).astype(np.float32)
    return tgt_wl, tgt_ref


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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

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
        x = x.unsqueeze(1)
        feat = self.net(x).squeeze(-1)
        return self.fc(feat)


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
        assert input_dim % patch_size == 0, "Input dimension must be divisible by patch size."
        self.input_dim = input_dim
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
        x = x.view(b, self.num_tokens, self.patch_size)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.fc(x[:, 0, :])


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
        self.local_branch = LocalCNNBranch(out_dim=128)
        self.global_branch = GlobalTransformerBranch(
            input_dim=256,
            patch_size=8,
            d_model=128,
            nhead=8,
            num_layers=2,
            ff_dim=256,
            out_dim=128,
        )
        self.fusion = FeatureFusion(loc_dim=128, glo_dim=128, fused_dim=128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(
        self, x: torch.Tensor, return_feat: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_loc = self.local_branch(x)
        z_glo = self.global_branch(x)
        z = self.fusion(z_loc, z_glo)
        logits = self.classifier(z)
        if return_feat:
            return logits, z_loc, z_glo, z
        return logits


# ==============================================================================
# Spectral CSV Parsing
# ==============================================================================
def load_csv_spectrum(csv_path: Path) -> Tuple[Optional[np.ndarray], np.ndarray, str]:
    """Parse CSV files containing spectral curves under varying schema conventions."""
    df = pd.read_csv(csv_path)
    cols_lower = [c.lower() for c in df.columns]

    if "wavelength" in cols_lower and "reflectance" in cols_lower:
        wl_col = df.columns[cols_lower.index("wavelength")]
        rf_col = df.columns[cols_lower.index("reflectance")]
        wavelength = df[wl_col].to_numpy(dtype=np.float32)
        reflectance = df[rf_col].to_numpy(dtype=np.float32)
        return wavelength, reflectance, "wavelength_reflectance"

    if df.shape[1] >= 2:
        wavelength = df.iloc[:, 0].to_numpy(dtype=np.float32)
        reflectance = df.iloc[:, 1].to_numpy(dtype=np.float32)
        return wavelength, reflectance, "first_two_columns"

    if df.shape[1] == 1:
        reflectance = df.iloc[:, 0].to_numpy(dtype=np.float32)
        return None, reflectance, "single_column"

    raise ValueError("Unrecognized CSV layout: unable to extract spectral vectors.")


# ==============================================================================
# Single-File Inference
# ==============================================================================
def predict_one_csv(model: nn.Module, class_names: list[str], csv_path: Path) -> Dict[str, Any]:
    """Execute end-to-end preprocessing, inference, and report generation for a single spectrum."""
    wavelength, reflectance, mode = load_csv_spectrum(csv_path)

    info: Dict[str, Any] = {
        "file_name": csv_path.name,
        "file_stem": csv_path.stem,
        "csv_mode": mode,
        "orig_points": None,
        "orig_wl_min_nm": None,
        "orig_wl_max_nm": None,
        "crop_points": None,
        "crop_wl_min_nm": None,
        "crop_wl_max_nm": None,
    }

    if wavelength is not None:
        info["orig_points"] = int(len(wavelength))
        info["orig_wl_min_nm"] = float(wavelength.min())
        info["orig_wl_max_nm"] = float(wavelength.max())

        wavelength, reflectance = crop_by_wavelength(
            wavelength,
            reflectance,
            wl_min=TARGET_MIN_NM,
            wl_max=TARGET_MAX_NM,
        )

        info["crop_points"] = int(len(wavelength))
        info["crop_wl_min_nm"] = float(wavelength.min())
        info["crop_wl_max_nm"] = float(wavelength.max())

        _, reflectance_256 = resample_by_wavelength(
            wavelength,
            reflectance,
            wl_min=TARGET_MIN_NM,
            wl_max=TARGET_MAX_NM,
            target_len=TARGET_DIM,
        )
        x = preprocess_spectrum_for_model(reflectance_256)
    else:
        reflectance = fix_nonfinite_1d(reflectance)
        info["orig_points"] = int(len(reflectance))
        x = preprocess_spectrum_for_model(reflectance)

    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = np.argsort(probs)[::-1]
    top5 = top_idx[:5]

    # Write individual prediction report
    out_txt = csv_path.parent / f"{csv_path.stem}_prediction.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Hyperspectral Mineral Classification Prediction\n")
        f.write("=" * 80 + "\n")
        f.write(f"Checkpoint : {MODEL_PATH}\n")
        f.write(f"Input File : {csv_path}\n")
        f.write(f"Inference Device : {DEVICE}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Schema Detection : {mode}\n")

        if wavelength is not None:
            f.write(f"Wavelength Window: {TARGET_MIN_NM:.1f} - {TARGET_MAX_NM:.1f} nm\n")

        f.write(f"Input Band Dimension: {TARGET_DIM}\n")
        f.write("=" * 80 + "\n")
        f.write("Top-5 Predicted Minerals & Confidence Scores:\n")
        f.write("=" * 80 + "\n")

        for rank, idx in enumerate(top5, start=1):
            f.write(f"{rank:02d}. {class_names[idx]:<25s} | Probability: {probs[idx]:.6f}\n")

        f.write("=" * 80 + "\n")
        f.write(f"Top-1 Candidate : {class_names[top_idx[0]]} (Confidence: {probs[top_idx[0]]:.4f})\n")
        f.write("=" * 80 + "\n")

    row = dict(info)
    for r, idx in enumerate(top5, start=1):
        row[f"top{r}_class"] = class_names[idx]
        row[f"top{r}_prob"] = float(probs[idx])
    row["txt_path"] = str(out_txt)

    return row


# ==============================================================================
# Main Batch Prediction Execution
# ==============================================================================
def main():
    print("🚀 Initializing Batch Spectral Inference Pipeline...")
    print(f"📦 Computing Device   : {DEVICE}")
    print(f"🔬 Checkpoint Path    : {MODEL_PATH}")
    print(f"📁 Target CSV Folder  : {CSV_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"CSV directory not found: {CSV_DIR}")

    csv_files = sorted(CSV_DIR.glob("frt*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 'frt*.csv' files located under {CSV_DIR}")

    print(f"🔎 Located {len(csv_files)} hyperspectral CSV files for inference.")

    ckpt = smart_torch_load(MODEL_PATH)
    class_names = ckpt.get("class_names", GLOBAL_CLASS_ORDER)

    model = HybridMineralNet(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    rows = []
    failed = []

    print("\n" + "=" * 80)
    print("⚡ Starting Inference Batch...")
    print("=" * 80)

    for i, csv_path in enumerate(csv_files, start=1):
        try:
            row = predict_one_csv(model, class_names, csv_path)
            rows.append(row)
            print(f"[{i:03d}/{len(csv_files):03d}] 📄 {csv_path.name:<28s} -> Top-1: {row['top1_class']:<20s} (Prob: {row['top1_prob']:.4f})")
        except Exception as e:
            print(f"[{i:03d}/{len(csv_files):03d}] ❌ {csv_path.name} Inference failed: {e}")
            failed.append({"file_name": csv_path.name, "error": str(e)})

    # Export consolidated summary CSV
    summary_csv = CSV_DIR / SUMMARY_CSV_NAME
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("🎯 Batch prediction completed successfully!")
    print(f"💾 Aggregated Summary Saved to : {summary_csv}")
    print("=" * 80)

    if failed:
        failed_csv = CSV_DIR / "all_frt_prediction_failed.csv"
        pd.DataFrame(failed).to_csv(failed_csv, index=False, encoding="utf-8-sig")
        print(f"⚠️ Warning: {len(failed)} files encountered errors. Log saved to: {failed_csv}")


if __name__ == "__main__":
    main()