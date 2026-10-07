# -*- coding: utf-8 -*-
"""
final_train_unified_hybrid_no_prior.py

Ablation Baseline: Unified Hybrid CNN-Transformer without Spectral Prior Loss.
- Architecture: Local 1D-CNN + Global Spectral Transformer.
- Regularization: Class-conditional region alignment loss (pull/push).
- Ablated component: Prior prototype alignment loss is excluded.
"""

from __future__ import annotations

import json
import math
import os
import random
from itertools import combinations
from pathlib import Path

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
OUT_DIR = PROJECT_ROOT / "data" / "final_unified_hybrid_no_prior"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Hyperparameters & Experiment Setup
# ==============================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
EARLY_STOP_PATIENCE = 15

# Region alignment loss hyper-parameters
LAMBDA_REGION = 0.20
NEG_WEIGHT = 0.50
NEG_MARGIN = 0.25

# Input & Network Dimensions
INPUT_DIM = 256
NUM_CLASSES = 15

CNN_FEAT_DIM = 128
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_FF_DIM = 256
FUSED_DIM = 128
PATCH_SIZE = 8

GLOBAL_CLASS_ORDER = [
    "Analcime", "Bassanite", "Chlorite", "Epidote", "Fe-Olivine",
    "High-Ca Pyroxene", "Illite/Muscovite", "Low-Ca Pyroxene",
    "Margarite", "Mg-Carbonate", "Mg-Smectite", "Monohydrated sulfate",
    "Plagioclase", "Prehnite", "Serpentine",
]


# ==============================================================================
# Utility Functions
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    """Set random seeds across libraries for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(y_train: np.ndarray, num_classes: int = 15) -> torch.Tensor:
    """Compute balanced inverse frequency weights for multi-class classification."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ==============================================================================
# Dataset Pipeline
# ==============================================================================
def load_unified_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load unified hyperspectral arrays, class labels, and geographical region indices."""
    x = np.load(DATA_DIR / "all_spectra_unified.npy").astype(np.float32)
    y_global = np.load(DATA_DIR / "all_labels_unified.npy").astype(np.int64)
    region_ids = np.load(DATA_DIR / "all_region_ids_unified.npy").astype(np.int64)
    y = y_global - 1  # Re-index 1-based labels to standard 0-based indexing
    return x, y, region_ids


class SpectraDataset(Dataset):
    """Spectral dataset yielding spectral curves, class targets, and spatial region tags."""

    def __init__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray):
        self.X = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.r = torch.from_numpy(r).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.r[idx]


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

            nn.AdaptiveAvgPool1d(1)
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
        self.local_branch = LocalCNNBranch(CNN_FEAT_DIM)
        self.global_branch = GlobalTransformerBranch(
            input_dim=INPUT_DIM,
            patch_size=PATCH_SIZE,
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_NHEAD,
            num_layers=TRANSFORMER_NUM_LAYERS,
            ff_dim=TRANSFORMER_FF_DIM,
            out_dim=TRANSFORMER_D_MODEL,
        )
        self.fusion = FeatureFusion(CNN_FEAT_DIM, TRANSFORMER_D_MODEL, FUSED_DIM)
        self.classifier = nn.Linear(FUSED_DIM, num_classes)

    def forward(
        self, x: torch.Tensor, return_feat: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_loc = self.local_branch(x)
        z_glo = self.global_branch(x)
        z = self.fusion(z_loc, z_glo)
        logits = self.classifier(z)
        if return_feat:
            return logits, z_loc, z_glo, z
        return logits


# ==============================================================================
# Objective Functions
# ==============================================================================
def compute_class_conditional_region_align_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    regions: torch.Tensor,
    device: torch.device,
    neg_margin: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes class-conditional intra-class attraction (pull) and inter-class
    margin-based separation (push) across distinct regional domains.
    """
    features = F.normalize(features, dim=-1)
    unique_regions = torch.unique(regions)

    if unique_regions.numel() < 2:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    # Compute region-wise class prototype centers
    proto_dict = {}
    for r in unique_regions:
        cls_in_r = torch.unique(targets[regions == r])
        for c in cls_in_r:
            mask = (regions == r) & (targets == c)
            center = features[mask].mean(dim=0, keepdim=True)
            proto_dict[(int(r.item()), int(c.item()))] = F.normalize(center, dim=-1)

    pull_losses = []
    push_losses = []

    # Pull loss: align identical classes across different regions
    all_classes = sorted({c for (_, c) in proto_dict.keys()})
    for cls_id in all_classes:
        region_list = sorted([r for (r, c) in proto_dict.keys() if c == cls_id])
        if len(region_list) < 2:
            continue
        for i in range(len(region_list)):
            for j in range(i + 1, len(region_list)):
                pa = proto_dict[(region_list[i], cls_id)]
                pb = proto_dict[(region_list[j], cls_id)]
                pull_losses.append(1.0 - F.cosine_similarity(pa, pb, dim=-1).mean())

    # Push loss: separate distinct classes across regions with a cosine margin
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
@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, float, float]:
    """Evaluate Overall Accuracy (OA), Average Accuracy (AA), and Macro-F1 score."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

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

    print("🚀 Initializing training pipeline (Ablation: No Spectral Prior)...")
    print(f"📦 Computing Device   : {DEVICE}")
    print(f"⚙️ Hyperparameters     : lambda_region={LAMBDA_REGION}, neg_weight={NEG_WEIGHT}, neg_margin={NEG_MARGIN}")

    x, y, region_ids = load_unified_data()
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_RATIO, random_state=SEED, stratify=y
    )

    x_train, y_train, r_train = x[train_idx], y[train_idx], region_ids[train_idx]
    x_val, y_val, r_val = x[val_idx], y[val_idx], region_ids[val_idx]

    print(f"📊 Dataset Split       : Train={x_train.shape[0]} samples, Validation={x_val.shape[0]} samples")

    train_ds = SpectraDataset(x_train, y_train, r_train)
    val_ds = SpectraDataset(x_val, y_val, r_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HybridMineralNet(num_classes=NUM_CLASSES).to(DEVICE)
    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
        total_loss, total_cls, total_pull, total_push = 0.0, 0.0, 0.0, 0.0

        for xb, yb, rb in train_loader:
            xb, yb, rb = xb.to(DEVICE), yb.to(DEVICE), rb.to(DEVICE)
            optimizer.zero_grad()

            logits, _, _, z = model(xb, return_feat=True)
            cls_loss = criterion(logits, yb)

            pull_loss, push_loss = compute_class_conditional_region_align_loss(
                z, yb, rb, DEVICE, NEG_MARGIN
            )
            region_loss = pull_loss + NEG_WEIGHT * push_loss

            loss = cls_loss + LAMBDA_REGION * region_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_cls += cls_loss.item() * xb.size(0)
            total_pull += pull_loss.item() * xb.size(0)
            total_push += push_loss.item() * xb.size(0)

        n = len(train_loader.dataset)
        train_loss = total_loss / n
        train_cls = total_cls / n
        train_pull = total_pull / n
        train_push = total_push / n

        val_loss, val_oa, val_aa, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_loss": train_cls,
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
            f"Train Loss: {train_loss:.4f} (Cls: {train_cls:.4f}, Pull: {train_pull:.4f}, Push: {train_push:.4f}) | "
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
                "lambda_region": LAMBDA_REGION,
                "neg_weight": NEG_WEIGHT,
                "neg_margin": NEG_MARGIN,
                "model_type": "HybridCNNTransformer_NoPrior",
            }, best_model_path)
            print(f"  ✨ Best model checkpoint updated at Epoch {epoch:03d} (Val Macro-F1: {val_f1:.4f})")
        else:
            early_counter += 1

        if early_counter >= EARLY_STOP_PATIENCE:
            print(f"🛑 Early stopping triggered at epoch {epoch}.")
            break

    # Save training logs and configuration
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
            "lambda_region": LAMBDA_REGION,
            "neg_weight": NEG_WEIGHT,
            "neg_margin": NEG_MARGIN,
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_f1),
            "model_type": "HybridCNNTransformer_NoPrior",
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("🎯 Training completed successfully!")
    print(f"💾 Checkpoint : {best_model_path}")
    print(f"🏆 Best Epoch : {best_epoch:03d} | Best Val Macro-F1: {best_val_f1:.4f}")
    print(f"📁 Output Dir : {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()