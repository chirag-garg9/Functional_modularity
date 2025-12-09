# plot_latent_pca_tsne.py
#
# Usage:
#   python plot_latent_pca_tsne.py \
#       --config path/to/config.yaml \
#       --checkpoint path/to/ckpt.pth \
#       --data ./data/dsprites.npz \
#       --out_dir ./eval_results/exp_name/latents_viz \
#       --max_samples 10000
#
# This will:
#   - extract latents z from the trained model on dSprites
#   - build factor labels (shape, scale, rot, pos_x, pos_y)
#   - compute PCA(2D) and t-SNE(2D) on z
#   - for each factor, save:
#       <factor>_pca.png
#       <factor>_tsne.png

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import yaml

# Make CUDA errors synchronous for easier debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------------- Models import ----------------
from models import (
    Autoencoder,
    Classifier,
    MultiTaskAE,
    RegularizedAE,
    BranchedAE,
    TaskFactorizedAE,
    ScaledAE,
    TransferAE,
    MoEBranchedAE,
    ConvEncoder,
)

# ---------------- Constants ----------------
# Dataset-specific factor definitions
DSPRITES_FACTORS = ["shape", "scale", "rot", "pos_x", "pos_y"]
DSPRITES_FACTOR_TYPES = {
    "shape": "categorical",
    "scale": "continuous",
    "rot": "continuous",
    "pos_x": "continuous",
    "pos_y": "continuous",
}

SHAPES3D_FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
SHAPES3D_FACTOR_TYPES = {
    "floor_hue": "categorical",
    "wall_hue": "categorical",
    "object_hue": "categorical",
    "scale": "continuous",
    "shape": "categorical",
    "orientation": "continuous",
}

def get_factors_and_types(dataset_type):
    """Get factors and factor types based on dataset type"""
    if dataset_type == 'shapes3d':
        return SHAPES3D_FACTORS, SHAPES3D_FACTOR_TYPES
    else:
        return DSPRITES_FACTORS, DSPRITES_FACTOR_TYPES


# ---------------- Dataset ----------------
class DSpritesDataset(torch.utils.data.Dataset):
    """
    Minimal dSprites loader, consistent with training/eval code.
    """

    def __init__(self, path="data/dsprites.npz"):
        data = np.load(path, allow_pickle=True, encoding="latin1")
        self.imgs = data["imgs"]
        self.latents_values = data["latents_values"]
        self.latents_classes = data["latents_classes"]
        self.transform = T.Compose(
            [T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0))]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx])
        vals = torch.tensor(self.latents_values[idx], dtype=torch.float32)
        cls = torch.tensor(self.latents_classes[idx], dtype=torch.long)
        return img, {"values": vals, "classes": cls}


def get_dsprites_loader(batch_size, path, num_workers=0, shuffle=False):
    ds = DSpritesDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ------------------------------
# 3D Shapes dataset
# ------------------------------
class Shapes3DDataset(torch.utils.data.Dataset):
    """
    3D Shapes dataset from DeepMind.
    Factors: floor_hue, wall_hue, object_hue, scale, shape, orientation
    Images are 64x64x3 RGB
    """

    def __init__(self, path="data/3dshapes.h5"):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for 3D Shapes dataset. Install with: pip install h5py")
        
        with h5py.File(path, 'r') as f:
            self.imgs = f['images'][:]  # (N, 64, 64, 3)
            self.latents_values = f['labels'][:]  # (N, 6) continuous values
            # Convert continuous values to discrete classes for compatibility
            self.latents_classes = self._values_to_classes(self.latents_values)

        self.transform = T.Compose([
            T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) / 255.0)  # HWC -> CHW, normalize
        ])

    def _values_to_classes(self, values):
        """Convert continuous latent values to discrete class indices"""
        classes = np.zeros_like(values, dtype=np.int64)
        for i in range(values.shape[1]):
            unique_vals = np.unique(values[:, i])
            # Map each value to its index in unique_vals
            classes[:, i] = np.searchsorted(unique_vals, values[:, i])
        return classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # (64, 64, 3)
        img = self.transform(img)
        vals = torch.tensor(self.latents_values[idx], dtype=torch.float32)
        cls = torch.tensor(self.latents_classes[idx], dtype=torch.long)
        return img, {"values": vals, "classes": cls}


def get_shapes3d_loader(batch_size, path, num_workers=0, shuffle=False):
    ds = Shapes3DDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ------------------------------
# Dataset type detection
# ------------------------------
def get_dataset_type(data_path: str):
    """Detect dataset type from file path"""
    if data_path.endswith('.h5') or '3dshapes' in data_path.lower() or 'shapes3d' in data_path.lower():
        return 'shapes3d'
    elif data_path.endswith('.npz') or 'dsprites' in data_path.lower():
        return 'dsprites'
    else:
        # Default to dsprites for backward compatibility
        return 'dsprites'


# ---------------- Model builder ----------------
def build_model(cfg, pretrained_encoder=None):
    mtype = cfg["model"]["type"].lower()
    latent_dim = cfg["model"].get("latent_dim", 10)
    num_classes = cfg["model"].get("num_classes", 3)
    depth = cfg["model"].get("depth", 4)
    freeze_encoder = cfg["model"].get("freeze_encoder", False)

    if mtype == "autoencoder":
        return Autoencoder(latent_dim=latent_dim)
    elif mtype in ["classifier", "classification"]:
        return Classifier(latent_dim=latent_dim, num_classes=num_classes)
    elif mtype in ["multi_task", "multitask"]:
        num_reg = cfg["model"].get("num_reg", 2)
        return MultiTaskAE(latent_dim=latent_dim, num_classes=num_classes, num_reg=num_reg)
    elif mtype in ["regularized", "regularizedae"]:
        reg_type = cfg["model"].get("reg_type", "l1")
        return RegularizedAE(
            latent_dim=latent_dim, num_classes=num_classes, reg_type=reg_type
        )
    elif mtype in ["branched", "branchedae"]:
        num_subspaces = cfg["model"].get("num_subspaces", 2)
        return BranchedAE(
            latent_dim=latent_dim,
            num_classes=num_classes,
            num_subspaces=num_subspaces,
        )
    elif mtype in ["taskfactorized", "task_factorized"]:
        num_reg = cfg["model"].get("num_reg", 2)
        return TaskFactorizedAE(
            latent_dim=latent_dim, num_classes=num_classes, num_reg=num_reg
        )
    elif mtype in ["scaled", "scaledae"]:
        width_multiplier = cfg["model"].get("width_multiplier", 1)
        depth = cfg["model"].get("depth", 4)
        input_size = cfg.get("data", {}).get("input_size", 64)
        return ScaledAE(
            latent_dim=latent_dim,
            num_classes=num_classes,
            width_multiplier=width_multiplier,
            depth=depth,
            input_size=input_size,
        )
    elif mtype in ["transfer", "transferae"]:
        return TransferAE(
            latent_dim=latent_dim,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            pretrained_encoder=pretrained_encoder,
        )
    elif mtype in ["moe", "mixture_of_experts"]:
        num_experts = cfg["model"].get("num_experts", 3)
        return MoEBranchedAE(
            latent_dim=latent_dim, num_classes=num_classes, num_experts=num_experts
        )
    else:
        raise ValueError(f"Unknown model type {mtype}")


# ---------------- Latent extraction ----------------
def collect_latents(model, loader, device, max_samples=None):
    """
    Extract latent codes z from the encoder.
    Respects max_samples if provided.
    """
    zs, vals, cls = [], [], []
    model.eval()
    total = 0
    with torch.no_grad():
        for x, y in loader:
            if max_samples is not None and total >= max_samples:
                break
            x = x.to(device)

            if hasattr(model, "get_latent"):
                z = model.get_latent(x)
            elif hasattr(model, "encoder"):
                z = model.encoder(x)
            else:
                out = model(x)
                z = out[1] if isinstance(out, tuple) else out

            zs.append(z.cpu())
            vals.append(y["values"])
            cls.append(y["classes"])
            total += x.size(0)

    z_all = torch.cat(zs).numpy()
    labels_all = {
        "values": torch.cat(vals).numpy(),
        "classes": torch.cat(cls).numpy(),
    }
    return z_all, labels_all


# ---------------- Factor dictionary ----------------
def build_factor_dict(labels, dataset_type="dsprites"):
    """
    Build factor dictionary based on dataset type.
    
    dsprites:
      - latents_values:  [color, shape, scale, orientation, posX, posY]
      - latents_classes: [color, shape, scale, orientation, posX, posY]
    
    shapes3d:
      - latents_values:  [floor_hue, wall_hue, object_hue, scale, shape, orientation]
      - latents_classes: [floor_hue, wall_hue, object_hue, scale, shape, orientation]
    """
    vals = labels["values"]
    cls = labels["classes"]

    if dataset_type == 'shapes3d':
        return {
            "floor_hue": cls[:, 0].astype(int),
            "wall_hue": cls[:, 1].astype(int),
            "object_hue": cls[:, 2].astype(int),
            "scale": vals[:, 3],
            "shape": cls[:, 4].astype(int),
            "orientation": vals[:, 5],
        }
    else:  # dsprites
        return {
            "shape": cls[:, 1].astype(int),
            "scale": vals[:, 2],
            "rot": vals[:, 3],
            "pos_x": vals[:, 4],
            "pos_y": vals[:, 5],
        }


# ---------------- Plotting helpers ----------------
def _prepare_subsample(z, factors_dict, max_points=5000, seed=0):
    """
    Subsample points for visualization to keep plots light.
    """
    N = z.shape[0]
    if N <= max_points:
        idx = np.arange(N)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, size=max_points, replace=False)
    z_sub = z[idx]
    factors_sub = {k: v[idx] for k, v in factors_dict.items()}
    return z_sub, factors_sub


def _compute_pca(z, n_components=2):
    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(z)
    return z_pca, pca


def _compute_tsne(z, n_components=2, perplexity=30, seed=0):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
    )
    z_tsne = tsne.fit_transform(z)
    return z_tsne, tsne


def _plot_embedding(
    Z2d, y, factor_name, method_name, out_path, factor_type="continuous"
):
    """
    Z2d: (N, 2) embedding
    y: (N,) factor values
    factor_type: "categorical" or "continuous"
    """
    plt.figure(figsize=(5, 4))

    if factor_type == "categorical":
        y = np.asarray(y).astype(int)
        classes = np.unique(y)
        for c in classes:
            mask = y == c
            plt.scatter(
                Z2d[mask, 0],
                Z2d[mask, 1],
                s=5,
                alpha=0.7,
                label=str(c),
            )
        plt.legend(title=f"{factor_name}")
    else:
        y = np.asarray(y)
        sc = plt.scatter(
            Z2d[:, 0],
            Z2d[:, 1],
            s=5,
            c=y,
            alpha=0.7,
        )
        cbar = plt.colorbar(sc)
        cbar.set_label(factor_name)

    plt.xlabel(f"{method_name} dim 1")
    plt.ylabel(f"{method_name} dim 2")
    plt.title(f"{method_name} of latent space colored by {factor_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------- Main routine ----------------
def run_pca_tsne_viz(config_path, ckpt_path, data_path, out_dir,
                     device=None, batch_size=256, max_samples=None,
                     max_points_vis=5000):
    # device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load config + model
    cfg = yaml.safe_load(open(config_path))
    model = build_model(cfg).to(device)

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # data loader - detect dataset type
    dataset_type = get_dataset_type(data_path)
    if dataset_type == 'shapes3d':
        print(f"→ Loading 3D Shapes dataset from {data_path}")
        loader = get_shapes3d_loader(batch_size, data_path, shuffle=False)
    else:
        print(f"→ Loading dSprites dataset from {data_path}")
        loader = get_dsprites_loader(batch_size, data_path, shuffle=False)

    # extract latents
    print("[1] Extracting latents...")
    z_all, labels_all = collect_latents(model, loader, device, max_samples=max_samples)
    print(f"Collected {z_all.shape[0]} samples with latent dim = {z_all.shape[1]}")

    factors_dict = build_factor_dict(labels_all, dataset_type)

    # Get dataset-specific factors
    FACTORS, FACTOR_TYPES = get_factors_and_types(dataset_type)

    # subsample for visualization
    print("[2] Subsampling latents for visualization...")
    z_sub, factors_sub = _prepare_subsample(z_all, factors_dict, max_points=max_points_vis)

    # global PCA / t-SNE
    print("[3] Computing PCA...")
    z_pca, pca = _compute_pca(z_sub, n_components=2)
    print("[4] Computing t-SNE (this may take some time)...")
    z_tsne, tsne = _compute_tsne(z_sub, n_components=2, perplexity=30, seed=0)

    # per-factor plots
    print("[5] Generating per-factor PCA / t-SNE plots...")
    for f in FACTORS:
        y = factors_sub[f]
        ftype = FACTOR_TYPES[f]

        # PCA
        pca_path = out_dir / f"{f}_pca.png"
        _plot_embedding(
            z_pca, y, factor_name=f,
            method_name="PCA",
            out_path=pca_path,
            factor_type=ftype,
        )

        # t-SNE
        tsne_path = out_dir / f"{f}_tsne.png"
        _plot_embedding(
            z_tsne, y, factor_name=f,
            method_name="t-SNE",
            out_path=tsne_path,
            factor_type=ftype,
        )

        print(f"  Saved {f}: {pca_path.name}, {tsne_path.name}")

    print(f"\n✅ All PCA/t-SNE plots saved in: {out_dir}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA / t-SNE visualization of latent space wrt dSprites factors"
    )
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint with model_state_dict")
    parser.add_argument("--data", default="./data/dsprites.npz", help="Path to dsprites.npz")
    parser.add_argument("--out_dir", default="plots", help="Output directory for plots")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on number of samples to extract latents from.")
    parser.add_argument("--max_points_vis", type=int, default=10000,
                        help="Max points used in PCA / t-SNE scatter plots.")

    args = parser.parse_args()

    run_pca_tsne_viz(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        data_path=args.data,
        out_dir=args.out_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_points_vis=args.max_points_vis,
    )
