# -*- coding: utf-8 -*-
"""
USGS_resample.py

USGS Spectral Library Processing & Standardization Pipeline:
1. Parses USGS ASCII mineral reflectance signatures (handling Fortran D-notation).
2. Performs keyword-based fuzzy mapping onto a unified 15-class Martian mineral taxonomy.
3. Automatically retrieves and matches corresponding wavelength reference files.
4. Resamples spectra onto the benchmark 425-channel grid (1.021 - 2.635 μm).
5. Computes Convex-Hull based Continuum Removal (CR) for absorption feature enhancement.
6. Serializes standardized samples, labels, and wavelength grids into a PyTorch archive (.pt).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import torch


# ==============================================================================
# Path Configurations (Relative to project root for seamless reproducibility)
# ==============================================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DEFAULT_USGS_ROOT = PROJECT_ROOT / "data" / "usgs_splib07" / "ASCIIdata" / "ASCIIdata_splib07a"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "usgs_priors_425.pt"


# ==============================================================================
# Spectral Reference & Taxonomy Configurations
# ==============================================================================
# Calibrated 425-band benchmark wavelength grid (1.021 to 2.635 μm)
BENCHMARK_W = np.linspace(1.021, 2.635, 425, dtype=np.float32)

# Unified 15-Class Mineral Taxonomy (0-indexed)
MINERAL_MAP = {
    "Analcime": 0,
    "Bassanite": 1,
    "Chlorite": 2,
    "Epidote": 3,
    "Fe-Olivine": 4,
    "High-Ca Pyroxene": 5,
    "Illite/Muscovite": 6,
    "Low-Ca Pyroxene": 7,
    "Margarite": 8,
    "Mg-Carbonate": 9,
    "Mg-Smectite": 10,
    "Monohydrated sulfate": 11,
    "Plagioclase": 12,
    "Prehnite": 13,
    "Serpentine": 14,
}

# Fuzzy keyword matching patterns for filename-level mineral identification
KEYWORDS = {
    "Analcime": ["analcime"],
    "Bassanite": ["bassanite", "gypsum"],
    "Chlorite": ["chlorite", "clinochlore"],
    "Epidote": ["epidote"],
    "Fe-Olivine": ["olivine", "fayalite", "forsterite"],
    "High-Ca Pyroxene": ["augite", "diopside", "hcp", "cpx"],
    "Illite/Muscovite": ["illite", "muscovite", "mica"],
    "Low-Ca Pyroxene": ["enstatite", "pigeonite", "lcp", "opx"],
    "Margarite": ["margarite"],
    "Mg-Carbonate": ["magnesite", "siderite", "carbonate"],
    "Mg-Smectite": ["saponite", "montmorillonite", "smectite"],
    "Monohydrated sulfate": ["kieserite", "szomolnokite"],
    "Plagioclase": ["plagioclase", "anorthite", "albite"],
    "Prehnite": ["prehnite"],
    "Serpentine": ["serpentine"],
}


# ==============================================================================
# Core Spectral Processing Functions
# ==============================================================================
def load_clean_data(file_path: Union[str, Path]) -> np.ndarray:
    """
    Parse ASCII spectral values, converting Fortran exponent formatting ('D+'/'D-')
    to standard scientific notation ('E+'/'E-') and pruning metadata count headers.
    """
    data = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.replace("D+", "E+").replace("D-", "E-").strip()
            if not line:
                continue
            try:
                data.append(float(line.split()[0]))
            except ValueError:
                continue

    data_arr = np.array(data, dtype=np.float32)
    # Strip initial header count integer if present at index 0
    if len(data_arr) > 1 and (int(data_arr[0]) == len(data_arr) - 1 or data_arr[0] > 5):
        data_arr = data_arr[1:]
    return data_arr


def calculate_cr(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute Convex-Hull Continuum Removal (CR) to isolate diagnostic absorption features.
    Computes ratio between raw reflectance and the upper convex hull envelope.
    """
    mask = ~np.isnan(r) & (r > 0)
    if np.sum(mask) < 10:
        return r

    w_s, r_s = w[mask], r[mask]
    try:
        points = np.column_stack((w_s, r_s))
        augmented = np.concatenate([points, [[w_s[0], 0.0], [w_s[-1], 0.0]]])
        hull = ConvexHull(augmented)
        verts = sorted([i for i in hull.vertices if i < len(w_s)])
        continuum = np.interp(w_s, w_s[verts], r_s[verts])
        cr = np.ones_like(r, dtype=np.float32)
        cr[mask] = r_s / continuum
        return cr
    except Exception:
        return r


# ==============================================================================
# Prior Library Generation Workflow
# ==============================================================================
def build_priors_library(usgs_root: Path, output_path: Path) -> None:
    """Scan, resample, continuum-remove, and export USGS mineral signatures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 Initializing USGS Spectral Prior Standardization Pipeline")
    print(f"📁 Source Library Directory : {usgs_root}")
    print(f"💾 Target Checkpoint Path   : {output_path}")
    print("=" * 80 + "\n")

    # Locate mineral reflectance spectra files
    search_dir = usgs_root / "ChapterM_Minerals"
    files = sorted(glob.glob(str(search_dir / "*.txt")))

    if not files:
        print(f"❌ No spectral text files discovered under {search_dir}")
        return

    # Pre-cache all corresponding wavelength calibration files for fast lookups
    print("🔍 Indexing and pre-caching wavelength calibration reference tables...")
    wave_files = glob.glob(str(usgs_root / "**" / "*Wavelengths*.txt"), recursive=True)
    wave_cache: Dict[int, np.ndarray] = {
        len(load_clean_data(wf)): load_clean_data(wf) for wf in wave_files
    }
    print(f"✅ Cached {len(wave_cache)} distinct wavelength band configurations.\n")

    all_samples: List[np.ndarray] = []
    all_labels: List[int] = []

    print(f"🔬 Processing {len(files)} candidate USGS spectral files...")

    for f_path in files:
        fname = Path(f_path).name.lower()

        # Fuzzy matching against target mineral taxonomy
        matched_label: Optional[int] = None
        for std_name, keys in KEYWORDS.items():
            if any(k in fname for k in keys):
                matched_label = MINERAL_MAP[std_name]
                break

        if matched_label is not None:
            r_raw = load_clean_data(f_path)
            w_raw = wave_cache.get(len(r_raw))

            if w_raw is not None:
                w_arr = w_raw.copy()
                # Standardize units from nanometers (nm) to micrometers (μm) if needed
                if np.nanmean(w_arr) > 10.0:
                    w_arr /= 1000.0

                # 1. Resample onto standardized 425-channel grid
                interp_f = interp1d(w_arr, r_raw, bounds_error=False, fill_value="extrapolate")
                r_resampled = interp_f(BENCHMARK_W).astype(np.float32)

                # 2. Continuum Removal (CR)
                r_cr = calculate_cr(BENCHMARK_W, r_resampled)

                all_samples.append(r_cr)
                all_labels.append(matched_label)

    # Serialize into PyTorch tensor dataset
    if all_samples:
        dataset = {
            "samples": torch.tensor(np.array(all_samples), dtype=torch.float32),
            "labels": torch.tensor(np.array(all_labels), dtype=torch.long),
            "wavelengths": torch.tensor(BENCHMARK_W, dtype=torch.float32),
        }
        torch.save(dataset, output_path)

        print("\n" + "=" * 80)
        print("✨ USGS Prior Standardization Pipeline Completed!")
        print(f"📊 Valid Extracted Samples : {len(all_samples)}")
        print(f"📂 Saved Archive Checkpoint : {output_path}")
        print("=" * 80 + "\n")
    else:
        print("\n❌ Failed to generate any valid samples. Please verify paths and keyword definitions.\n")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Standardize USGS ASCII mineral spectra into a 425-band PyTorch prior dataset."
    )
    parser.add_argument(
        "--usgs-root",
        type=Path,
        default=DEFAULT_USGS_ROOT,
        help="Root path containing the USGS splib07 ASCII dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Target .pt file path to save the serialized prior dataset.",
    )

    args = parser.parse_args()
    build_priors_library(usgs_root=args.usgs_root, output_path=args.output_path)


if __name__ == "__main__":
    main()