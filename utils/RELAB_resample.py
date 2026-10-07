# -*- coding: utf-8 -*-
"""
inspect_all_prior_wavelength.py

Spectral Prior Diagnostic Tool:
Inspects serialized PyTorch prior tensors (*.pt) to verify the existence,
structure, array shapes, label distribution, and wavelength band intervals.

Target Spectral Libraries:
1. usgs_priors_425.pt
2. relab_meteorite_priors_425.pt
3. relab_earthminerals_priors_425.pt
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ==============================================================================
# Path Configurations (Relative to project root for seamless reproducibility)
# ==============================================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_ROOT / "data"

PT_FILES = [
    DATA_DIR / "usgs_priors_425.pt",
    DATA_DIR / "relab_meteorite_priors_425.pt",
    DATA_DIR / "relab_earthminerals_priors_425.pt",
]


# ==============================================================================
# Numerical & Diagnostic Utility Functions
# ==============================================================================
def smart_torch_load(path: Path) -> Any:
    """Safely load serialized PyTorch objects handling cross-version differences."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(x: Any) -> np.ndarray:
    """Convert a PyTorch tensor or array-like object to a NumPy ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def print_array_preview(arr: np.ndarray, name: str, max_show: int = 10) -> None:
    """Display comprehensive dimensionality and head/tail previews for 1D arrays."""
    arr = np.asarray(arr).reshape(-1)
    n = len(arr)

    print(f"  • Shape      : {arr.shape}")
    print(f"  • Data Type  : {arr.dtype}")

    if n == 0:
        print(f"  ⚠️ {name}: Empty array.")
        return

    head = arr[:max_show]
    tail = arr[-max_show:] if n > max_show else arr

    print(f"  • Head [{min(max_show, n)} values]: {head}")
    print(f"  • Tail [{min(max_show, n)} values]: {tail}")
    print(f"  • Dynamic Range    : [{np.min(arr):.4f}, {np.max(arr):.4f}]")


def inspect_wavelengths(w: np.ndarray) -> None:
    """Analyze wavelength monotonicity and spectral sampling step distributions."""
    w = np.asarray(w).reshape(-1).astype(np.float64)

    if len(w) < 2:
        print("  ⚠️ Wavelength array length is insufficient to compute delta intervals.")
        return

    diffs = np.diff(w)
    is_monotonic_inc = np.all(diffs > 0)
    is_monotonic_nondec = np.all(diffs >= 0)

    print("\n  📈 Wavelength Interval & Monotonicity Analysis:")
    print(f"    - Strictly Increasing : {bool(is_monotonic_inc)}")
    print(f"    - Non-Decreasing      : {bool(is_monotonic_nondec)}")
    print(f"    - Delta (Min / Max)   : {diffs.min():.6f} / {diffs.max():.6f}")
    print(f"    - Delta (Mean ± Std)  : {diffs.mean():.6f} ± {diffs.std():.6f}")

    unique_diffs = np.unique(np.round(diffs, 8))
    print(f"    - Unique Step Count (rounded to 8 decimals): {len(unique_diffs)}")
    print(f"    - Sample Sampling Steps (first 10): {unique_diffs[:10]}")


def inspect_one_pt(pt_path: Path) -> None:
    """Inspect the internal dictionary structure and spectral arrays of a checkpoint file."""
    print("\n" + "=" * 90)
    print(f"🔬 Inspecting Target File: {pt_path.name}")
    print(f"📁 Absolute Path         : {pt_path.resolve()}")
    print(f"📦 File Exists           : {pt_path.exists()}")

    if not pt_path.exists():
        print("❌ File does not exist. Skipping inspection.")
        return

    obj = smart_torch_load(pt_path)
    print(f"🏗️ Top-level Type        : {type(obj)}")

    if not isinstance(obj, dict):
        print("⚠️ Top-level container is not a dictionary. Cannot inspect keys.")
        return

    print(f"🔑 Dictionary Keys       : {list(obj.keys())}")

    # Inspect 'samples'
    if "samples" in obj:
        samples = to_numpy(obj["samples"])
        print("\n  [📊 samples]")
        print(f"  • Shape      : {samples.shape}")
        print(f"  • Data Type  : {samples.dtype}")
        if samples.size > 0:
            print(f"  • Value Range: [{np.nanmin(samples):.4f}, {np.nanmax(samples):.4f}]")
    else:
        print("\n  ⚠️ [samples] Field not found.")

    # Inspect 'labels'
    if "labels" in obj:
        labels = to_numpy(obj["labels"])
        print("\n  [🏷️ labels]")
        print(f"  • Shape      : {labels.shape}")
        print(f"  • Data Type  : {labels.dtype}")
        if labels.size > 0:
            labels_flat = labels.reshape(-1)
            uniq, cnt = np.unique(labels_flat, return_counts=True)
            print("  • Class Frequency Distribution:")
            for u, c in zip(uniq, cnt):
                print(f"    - Class [{int(u):02d}]: {c:>5d} samples")
    else:
        print("\n  ⚠️ [labels] Field not found.")

    # Inspect 'wavelengths'
    if "wavelengths" in obj:
        wavelengths = to_numpy(obj["wavelengths"])
        print("\n  [🌈 wavelengths] Present:")
        print_array_preview(wavelengths, "wavelengths", max_show=10)
        inspect_wavelengths(wavelengths)
    else:
        print("\n  ⚠️ [wavelengths] Field not found.")

    # Scan for other candidate spectral keys
    candidate_keys = []
    for k in obj.keys():
        k_low = str(k).lower()
        if any(token in k_low for token in ["wave", "band", "channel", "lambda", "wl"]):
            candidate_keys.append(k)

    print("\n  🔍 Additional Wavelength-Related Keys:")
    if not candidate_keys:
        print("    None detected.")
    else:
        print(f"    Detected: {candidate_keys}")
        for k in candidate_keys:
            if k == "wavelengths":
                continue
            v = obj[k]
            try:
                arr = to_numpy(v)
                print(f"\n    [{k}]")
                print(f"    • Shape    : {arr.shape}")
                print(f"    • Data Type: {arr.dtype}")
                if arr.size > 0 and arr.ndim == 1:
                    print_array_preview(arr, str(k), max_show=10)
            except Exception as e:
                print(f"    ⚠️ Unable to convert key '{k}' to NumPy array: {e}")

    print("=" * 90)


# ==============================================================================
# Main Diagnostic Execution
# ==============================================================================
def main():
    print("🚀 Initializing Spectral Prior Diagnostic Tool...")
    print(f"⚙️ PyTorch Runtime Version: {torch.__version__}")

    for pt_path in PT_FILES:
        inspect_one_pt(pt_path)

    print("\n✅ All spectral prior inspections completed successfully!")


if __name__ == "__main__":
    main()