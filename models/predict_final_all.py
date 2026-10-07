# -*- coding: utf-8 -*-
"""
final_unified_model_allpriors_ccalign.py

Unified 1D-CNN Baseline: Multi-Source Spectral Priors & Class-Conditional Alignment.
- Study Sites: Co-training across all Martian regional datasets (HC, NF, UP).
- Target Hierarchy: 15-class unified Martian mineral taxonomy.
- Knowledge Fusion: Multi-source laboratory spectral libraries (USGS, RELAB Meteorite, RELAB Earth Minerals).
- Objectives: Class-weighted Cross-Entropy + Spectral Prior Alignment + Class-Conditional Regional Consistency.
"""

from __future__ import annotations

import json
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# Path Configurations (Relative to project root for seamless reproducibility)
# ==============================================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_ROOT / "data" / "dataset"

USGS_PRIOR_PT = PROJECT_ROOT / "data" / "usgs_priors_425.pt"
RELAB_METEORITE_PRIOR_PT = PROJECT_ROOT / "data" / "relab_meteorite_priors_425.pt"
RELAB_EARTH_PRIOR_PT = PROJECT_ROOT / "data" / "relab_earthminerals_priors_425.pt"

OUT_DIR = PROJECT_ROOT / "data" / "final_unified_model_allpriors_ccalign"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Hyperparameters & Experiment Configurations
# ==============================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
EARLY_STOP_PATIENCE = 15

# Regularization hyperparameters
LAMBDA_PROTO = 0.20
LAMBDA_ALIGN = 0.20
NEG_WEIGHT = 0.50
NEG_MARGIN = 0.25

# Robust normalization percentiles
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0
EPS = 1e-8

INPUT_DIM = 256
NUM_CLASSES = 15
FEAT_DIM = 128


# ==============================================================================
# Class & Regional Domain Taxonomy
# ==============================================================================
REGION_ID_TO_NAME = {0: "HC", 1: "NF", 2: "UP"}

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
GLOBAL_ID_TO_NAME = {i: name for i, name in enumerate(GLOBAL_CLASS_ORDER)}


# ==============================================================================
# Numerical & Spectral Utility Functions
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    """Set random seeds across runtime libraries for strict determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smart_torch_load(path: Path):
    """Safely load serialized PyTorch objects handling cross-version compatibility."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(x) -> np.ndarray:
    """Convert tensor or array-like object to a NumPy ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def resample_1d(x: np.ndarray, target_len: int = 256) -> np.ndarray:
    """Resample 1D spectral curve to the target dimension using linear interpolation."""
    x = np.asarray(x, dtype=np.float32)
    src_len = len(x)

    if src_len == target_len:
        return x.copy()

    src_grid = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    tgt_grid = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(tgt_grid, src_grid, x).astype(np.float32)


def preprocess_prior_spectrum(x: np.ndarray) -> np.ndarray:
    """Preprocess raw prior laboratory spectral signatures into standardized 256-band profiles."""
    x = fix_nonfinite_1d(x)
    x = robust_minmax_1d(x, LOW_PERCENTILE, HIGH_PERCENTILE, EPS)
    x = resample_1d(x, INPUT_DIM)
    return x


# ==============================================================================
# Dataset Pipeline
# ==============================================================================
def load_unified_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load unified Martian spectra, ground truth labels, and regional domain IDs."""
    x = np.load(DATA_DIR / "all_spectra_unified.npy").astype(np.float32)
    y_global = np.load(DATA_DIR / "all_labels_unified.npy").astype(np.int64)
    region_ids = np.load(DATA_DIR / "all_region_ids_unified.npy").astype(np.int64)

    y = y_global - 1  # Convert to standard 0-based label indexing
    return x, y, region_ids


class SpectraDataset(Dataset):
    """Dataset providing Martian hyperspectral samples, targets, and regional identifiers."""

    def __init__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray):
        self.X = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.r = torch.from_numpy(r).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.r[idx]


class PriorDataset(Dataset):
    """Dataset providing standardized laboratory prior spectral signatures and class annotations."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ==============================================================================
# Multi-Source Spectral Prior Aggregation
# ==============================================================================
def parse_single_prior_pt(prior_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse and validate serialized spectral prior tensors."""
    obj = smart_torch_load(prior_path)

    if not isinstance(obj, dict):
        raise ValueError(f"Invalid format: {prior_path} must be a dictionary, got {type(obj)}")
    if "samples" not in obj or "labels" not in obj:
        raise ValueError(f"Missing mandatory keys ('samples', 'labels') in {prior_path}")

    spectra = to_numpy(obj["samples"]).astype(np.float32)
    labels = to_numpy(obj["labels"]).astype(np.int64).reshape(-1)

    if spectra.ndim != 2:
        raise ValueError(f"Expected 2D sample array in {prior_path}, got shape {spectra.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D label array in {prior_path}, got shape {labels.shape}")
    if spectra.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Sample-label mismatch in {prior_path}: {spectra.shape[0]} vs {labels.shape[0]}"
        )

    return spectra, labels


def build_merged_prior_loader(prior_paths: List[Path]) -> Tuple[DataLoader, int]:
    """Merge and filter multiple external spectral libraries into a unified prior DataLoader."""
    all_x_prior = []
    all_y_prior = []

    print("\n" + "=" * 80)
    print("🔬 Aggregating Multi-Source Prior Spectral Libraries...")

    for prior_path in prior_paths:
        if not prior_path.exists():
            raise FileNotFoundError(f"Prior library not found: {prior_path}")

        spectra, labels = parse_single_prior_pt(prior_path)
        x_part = []
        y_part = []

        for spec, lab in zip(spectra, labels):
            lab = int(lab)
            if not (0 <= lab < NUM_CLASSES):
                continue
            spec_256 = preprocess_prior_spectrum(spec)
            x_part.append(spec_256)
            y_part.append(lab)

        if len(x_part) == 0:
            print(f"⚠️ [Warning] No valid samples parsed from {prior_path.name}")
            continue

        x_part = np.stack(x_part, axis=0).astype(np.float32)
        y_part = np.array(y_part, dtype=np.int64)

        print(f"  • {prior_path.name:<32s} : {x_part.shape[0]:>5d} samples")
        all_x_prior.append(x_part)
        all_y_prior.append(y_part)

    if len(all_x_prior) == 0:
        raise RuntimeError("No valid spectral prior samples could be extracted.")

    x_prior = np.concatenate(all_x_prior, axis=0).astype(np.float32)
    y_prior = np.concatenate(all_y_prior, axis=0).astype(np.int64)

    print("-" * 80)
    print("📊 Consolidated Prior Knowledge Base Distribution:")
    print(f"  Total Samples : {x_prior.shape[0]}")
    for cid in range(NUM_CLASSES):
        cnt = int((y_prior == cid).sum())
        cname = GLOBAL_ID_TO_NAME[cid]
        print(f"  Class [{cid:02d}] {cname:<24s} : {cnt:>4d} samples")
    print("=" * 80 + "\n")

    ds = PriorDataset(x_prior, y_prior)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    return loader, x_prior.shape[0]


# ==============================================================================
# Model Architecture
# ==============================================================================
class SpectralEncoder1D(nn.Module):
    """Deep 1D-CNN backbone for extracting hierarchical local spectral absorption features."""

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (B, 1, 256)
        x = self.net(x).squeeze(-1)  # (B, 128)
        return self.fc(x)


class UnifiedMineralNet(nn.Module):
    """End-to-end 1D spectral classification model for mineral identification."""

    def __init__(self, num_classes: int = 15, feat_dim: int = 128):
        super().__init__()
        self.encoder = SpectralEncoder1D(feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_feat: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        return logits


# ==============================================================================
# Objective Functions
# ==============================================================================
@torch.no_grad()
def compute_prior_prototypes(
    model: nn.Module, prior_loader: DataLoader, device: torch.device
) -> Dict[int, torch.Tensor]:
    """Compute normalized prototype feature centroids from external laboratory spectra."""
    model.eval()
    feat_bank: Dict[int, List[torch.Tensor]] = {}

    for x_prior, y_prior in prior_loader:
        x_prior = x_prior.to(device)
        y_prior = y_prior.to(device)

        _, feat = model(x_prior, return_feat=True)
        feat = F.normalize(feat, dim=-1)

        for cls in torch.unique(y_prior):
            cid = int(cls.item())
            mask = (y_prior == cls)
            feat_bank.setdefault(cid, [])
            feat_bank[cid].append(feat[mask].detach().cpu())

    proto_bank = {}
    for cid, feat_list in feat_bank.items():
        feat_all = torch.cat(feat_list, dim=0)
        proto = feat_all.mean(dim=0, keepdim=True)
        proto_bank[cid] = F.normalize(proto, dim=-1).squeeze(0)

    model.train()
    return proto_bank


def compute_proto_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    prior_proto_bank: Dict[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Cosine distance loss aligning observed spectral embeddings with laboratory prior prototypes."""
    features = F.normalize(features, dim=-1)
    losses = []

    for cls in torch.unique(targets):
        cid = int(cls.item())
        if cid not in prior_proto_bank:
            continue

        mask = (targets == cls)
        batch_proto = features[mask].mean(dim=0, keepdim=True)
        batch_proto = F.normalize(batch_proto, dim=-1)

        prior_proto = prior_proto_bank[cid].to(device).unsqueeze(0)
        prior_proto = F.normalize(prior_proto, dim=-1)

        loss_cls = 1.0 - F.cosine_similarity(batch_proto, prior_proto, dim=-1).mean()
        losses.append(loss_cls)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def compute_class_conditional_region_align_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    regions: torch.Tensor,
    device: torch.device,
    neg_margin: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes class-conditional intra-class attraction (pull) and inter-class
    margin-based separation (push) across distinct regional domains.
    """
    features = F.normalize(features, dim=-1)
    unique_regions = torch.unique(regions)

    if unique_regions.numel() < 2:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    proto_dict = {}
    for r in unique_regions:
        cls_in_r = torch.unique(targets[regions == r])
        for c in cls_in_r:
            mask = (regions == r) & (targets == c)
            center = features[mask].mean(dim=0, keepdim=True)
            proto_dict[(int(r.item()), int(c.item()))] = F.normalize(center, dim=-1)

    pull_losses = []
    push_losses = []

    # Pull loss: enforce inter-regional feature invariance for identical mineral classes
    all_classes = sorted({c for (_, c) in proto_dict.keys()})
    for cls_id in all_classes:
        region_list = sorted([r for (r, c) in proto_dict.keys() if c == cls_id])
        if len(region_list) < 2:
            continue
        for ra, rb in combinations(region_list, 2):
            pa = proto_dict[(ra, cls_id)]
            pb = proto_dict[(rb, cls_id)]
            pull_losses.append(1.0 - F.cosine_similarity(pa, pb, dim=-1).mean())

    # Push loss: separate distinct classes across regions using a margin threshold
    region_pairs = list(combinations(sorted([int(r.item()) for r in unique_regions]), 2))
    for ra, rb in region_pairs:
        classes_a = sorted([c for (r, c) in proto_dict.keys() if r == ra])
        classes_b = sorted([c for (r, c) in proto_dict.keys() if r == rb])

        for ca in classes_a:
            for cb in classes_b:
                if ca == cb:
                    continue
                pa = proto_dict[(ra, ca)]
                pb = proto_dict[(rb, cb)]
                sim = F.cosine_similarity(pa, pb, dim=-1).mean()
                push_losses.append(F.relu(sim - neg_margin))

    pull_loss = torch.stack(pull_losses).mean() if pull_losses else torch.tensor(0.0, device=device)
    push_loss = torch.stack(push_losses).mean() if push_losses else torch.tensor(0.0, device=device)
    return pull_loss, push_loss


# ==============================================================================
# Validation & Evaluation
# ==============================================================================
def compute_class_weights(y_train: np.ndarray, num_classes: int = 15) -> torch.Tensor:
    """Compute balanced inverse frequency weights for multi-class classification."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, float, float]:
    """Evaluate Overall Accuracy (OA), Average Accuracy (AA), and Macro-F1 score."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(all_labels)
    oa = accuracy_score(all_labels, all_preds)
    aa = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, float(oa), float(aa), float(mf1)


# ==============================================================================
# Training Pipeline
# ==============================================================================
def main():
    set_seed(SEED)

    print("🚀 Initializing Unified 1D-CNN Baseline Training Pipeline...")
    print(f"📦 Computing Device   : {DEVICE}")
    print(f"⚙️ Hyperparameters     : batch_size={BATCH_SIZE}, lr={LR}, lambda_proto={LAMBDA_PROTO}, lambda_align={LAMBDA_ALIGN}")

    x, y, region_ids = load_unified_data()
    print("✅ Unified dataset successfully loaded.")
    print(f"  • Spectra array shape : {x.shape}")
    print(f"  • Label array shape   : {y.shape}")
    print(f"  • Regional domains    : {[REGION_ID_TO_NAME.get(r, str(r)) for r in sorted(np.unique(region_ids))]}")

    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=y
    )

    x_train, y_train, r_train = x[train_idx], y[train_idx], region_ids[train_idx]
    x_val, y_val, r_val = x[val_idx], y[val_idx], region_ids[val_idx]

    print(f"📊 Dataset Split       : Train={x_train.shape[0]} samples, Validation={x_val.shape[0]} samples")

    train_ds = SpectraDataset(x_train, y_train, r_train)
    val_ds = SpectraDataset(x_val, y_val, r_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    prior_loader, total_prior_num = build_merged_prior_loader([
        USGS_PRIOR_PT,
        RELAB_METEORITE_PRIOR_PT,
        RELAB_EARTH_PRIOR_PT,
    ])

    model = UnifiedMineralNet(num_classes=NUM_CLASSES, feat_dim=FEAT_DIM).to(DEVICE)
    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_val_f1 = -1.0
    best_epoch = -1
    early_counter = 0
    history = []
    best_model_path = OUT_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        prior_proto_bank = compute_prior_prototypes(model, prior_loader, DEVICE)

        total_loss, total_cls, total_proto, total_pull, total_push = 0.0, 0.0, 0.0, 0.0, 0.0

        for xb, yb, rb in train_loader:
            xb, yb, rb = xb.to(DEVICE), yb.to(DEVICE), rb.to(DEVICE)
            optimizer.zero_grad()

            logits, feat = model(xb, return_feat=True)

            cls_loss = criterion(logits, yb)
            proto_loss = compute_proto_loss(feat, yb, prior_proto_bank, DEVICE)
            pull_loss, push_loss = compute_class_conditional_region_align_loss(
                feat, yb, rb, DEVICE, NEG_MARGIN
            )

            align_loss = pull_loss + NEG_WEIGHT * push_loss
            loss = cls_loss + LAMBDA_PROTO * proto_loss + LAMBDA_ALIGN * align_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_cls += cls_loss.item() * xb.size(0)
            total_proto += proto_loss.item() * xb.size(0)
            total_pull += pull_loss.item() * xb.size(0)
            total_push += push_loss.item() * xb.size(0)

        n = len(train_loader.dataset)
        train_loss = total_loss / n
        train_cls = total_cls / n
        train_proto = total_proto / n
        train_pull = total_pull / n
        train_push = total_push / n

        val_loss, val_oa, val_aa, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_loss": train_cls,
            "train_proto_loss": train_proto,
            "train_pull_loss": train_pull,
            "train_push_loss": train_push,
            "val_loss": val_loss,
            "val_oa": val_oa,
            "val_aa": val_aa,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} (Cls: {train_cls:.4f}, Proto: {train_proto:.4f}, Pull: {train_pull:.4f}, Push: {train_push:.4f}) | "
            f"Val Loss: {val_loss:.4f} | OA: {val_oa:.4f} | AA: {val_aa:.4f} | Macro-F1: {val_f1:.4f}"
        )

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            early_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "input_dim": INPUT_DIM,
                "class_names": GLOBAL_CLASS_ORDER,
                "lambda_proto": LAMBDA_PROTO,
                "lambda_align": LAMBDA_ALIGN,
                "neg_weight": NEG_WEIGHT,
                "neg_margin": NEG_MARGIN,
                "total_prior_samples": int(total_prior_num),
                "model_type": "Unified1DCNN_AllPriors_CCAlign",
            }, best_model_path)
            print(f"  ✨ Best model checkpoint updated at Epoch {epoch:03d} (Val Macro-F1: {val_f1:.4f})")
        else:
            early_counter += 1

        if early_counter >= EARLY_STOP_PATIENCE:
            print(f"🛑 Early stopping triggered at epoch {epoch}.")
            break

    # Save detailed training log and configuration metadata
    pd.DataFrame(history).to_csv(OUT_DIR / "train_history.csv", index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "device": str(DEVICE),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "val_ratio": VAL_RATIO,
            "lambda_proto": LAMBDA_PROTO,
            "lambda_align": LAMBDA_ALIGN,
            "neg_weight": NEG_WEIGHT,
            "neg_margin": NEG_MARGIN,
            "class_names": GLOBAL_CLASS_ORDER,
            "prior_files": [
                USGS_PRIOR_PT.name,
                RELAB_METEORITE_PRIOR_PT.name,
                RELAB_EARTH_PRIOR_PT.name,
            ],
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_f1),
            "total_prior_samples": int(total_prior_num),
            "model_type": "Unified1DCNN_AllPriors_CCAlign",
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("🎯 Unified 1D-CNN baseline model training completed successfully!")
    print(f"💾 Checkpoint : {best_model_path}")
    print(f"🏆 Best Epoch : {best_epoch:03d} | Best Val Macro-F1: {best_val_f1:.4f}")
    print(f"📁 Output Dir : {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()