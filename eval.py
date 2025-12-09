# eval.py
import os
import json
import yaml
import argparse
import torch
from pathlib import Path
import datetime
import hashlib
import traceback

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T

from models import (Autoencoder, Classifier, MultiTaskAE, RegularizedAE,
                    BranchedAE, TaskFactorizedAE, ScaledAE, TransferAE,
                    MoEBranchedAE, ConvEncoder)

from analysis.evaluation import evaluate_all_metrics

# ------------------------------
# DSprites dataset (same as train)
# ------------------------------
class DSpritesDataset(torch.utils.data.Dataset):
    def __init__(self, path="data/dsprites.npz"):
        data = np.load(path, allow_pickle=True, encoding="latin1")
        self.imgs = data["imgs"]
        self.latents_values = data["latents_values"]
        self.latents_classes = data["latents_classes"]
        self.transform = T.Compose([
            T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx])
        vals = torch.tensor(self.latents_values[idx], dtype=torch.float32)
        cls = torch.tensor(self.latents_classes[idx], dtype=torch.long)
        return img, {"values": vals, "classes": cls}


def get_dsprites_loader(batch_size, path, num_workers=0):
    ds = DSpritesDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ------------------------------
# 3D Shapes dataset
# ------------------------------
class Shapes3DDataset(torch.utils.data.Dataset):
    def __init__(self, path="data/3dshapes.h5"):
        """
        3D Shapes dataset from DeepMind.
        Factors: floor_hue, wall_hue, object_hue, scale, shape, orientation
        Images are 64x64x3 RGB
        """
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


def get_shapes3d_loader(batch_size, path, num_workers=0):
    ds = Shapes3DDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


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


# ------------------------------
# Model builder (same as train)
# ------------------------------
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
        return RegularizedAE(latent_dim=latent_dim, num_classes=num_classes, reg_type=reg_type)
    elif mtype in ["branched", "branchedae"]:
        num_subspaces = cfg["model"].get("num_subspaces", 2)
        return BranchedAE(latent_dim=latent_dim, num_classes=num_classes, num_subspaces=num_subspaces)
    elif mtype in ["taskfactorized", "task_factorized"]:
        num_reg = cfg["model"].get("num_reg", 2)
        return TaskFactorizedAE(latent_dim=latent_dim, num_classes=num_classes, num_reg=num_reg)
    elif mtype in ["scaled", "scaledae"]:
        width_multiplier = cfg["model"].get("width_multiplier", 1)
        depth = cfg["model"].get("depth", 4)
        input_size = cfg["data"].get("input_size", 64) if "data" in cfg else 64
        return ScaledAE(latent_dim=latent_dim, num_classes=num_classes,
                        width_multiplier=width_multiplier, depth=depth, input_size=input_size)
    elif mtype in ["transfer", "transferae"]:
        return TransferAE(latent_dim=latent_dim, num_classes=num_classes,
                         freeze_encoder=freeze_encoder, pretrained_encoder=pretrained_encoder)
    elif mtype in ["moe", "mixture_of_experts"]:
        num_experts = cfg["model"].get("num_experts", 3)
        return MoEBranchedAE(latent_dim=latent_dim, num_classes=num_classes, num_experts=num_experts)
    else:
        raise ValueError(f"Unknown model type {mtype}")


# ------------------------------
# Latent extractor
# ------------------------------
def collect_latents(model, loader, device):
    zs, vals, cls = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if hasattr(model,"get_latent"):
                z = model.get_latent(x)
            elif hasattr(model,"encoder"):
                z = model.encoder(x)
            else:
                out = model(x)
                z = out[1] if isinstance(out,tuple) else out
            zs.append(z.cpu())
            vals.append(y["values"])
            cls.append(y["classes"])
    return torch.cat(zs).numpy(), {"values": torch.cat(vals).numpy(), "classes": torch.cat(cls).numpy()}

# ------------------------------
# Main eval
# ------------------------------
def run_eval(config_path, ckpt_path, data_path):
    cfg = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset - detect dataset type
    bs = cfg.get("training",{}).get("batch_size",128)
    dataset_type = get_dataset_type(data_path)
    
    if dataset_type == 'shapes3d':
        print(f"→ Loading 3D Shapes dataset from {data_path}")
        test_loader = get_shapes3d_loader(bs, data_path)
    else:
        print(f"→ Loading dSprites dataset from {data_path}")
        test_loader = get_dsprites_loader(bs, data_path)

    # Build model
    model = build_model(cfg).to(device)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Output directory
    exp_name = Path(config_path).stem
    out_dir = Path(f"eval_results/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract latents + evaluate
    zs, labels = collect_latents(model, test_loader, device)

    metrics = evaluate_all_metrics(
        model, torch.from_numpy(zs), labels, device=device
    )
    print("\n===== Final Evaluation =====")
    for k,v in metrics.items():
        print(f"{k}: {v}")

    with open(out_dir/"final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved results to: {out_dir}")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="./data/dsprites.npz")
    args = parser.parse_args()

    try:
        run_eval(args.config, args.checkpoint, args.data)
    except Exception as e:
        print("❌ Eval failed:", e)
        traceback.print_exc()
