# train.py
import os
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import datetime
import hashlib
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as T
import multiprocessing

# Import your models (these should point to your updated basemodels.py)
from models import (Autoencoder, Classifier, MultiTaskAE, RegularizedAE,
                    BranchedAE, TaskFactorizedAE, ScaledAE, TransferAE, MoEBranchedAE,ConvEncoder)

# Import evaluation helper (master evaluator + visuals)
# Make sure this path matches your project layout (analysis/evaluation.py)
from analysis.evaluation import evaluate_all_metrics, visualize_functional_modularity

# --------------------------
# DSprites Dataset (kept)
# --------------------------
class DSpritesDataset(Dataset):
    def __init__(self, path="data/dsprites.npz", train=True, split_ratio=0.8, random_seed=42):
        data = np.load(path, allow_pickle=True, encoding="latin1")
        self.imgs = data["imgs"]
        self.latents_values = data["latents_values"]
        self.latents_classes = data["latents_classes"]  # discrete class labels

        # Split train/test deterministically
        np.random.seed(random_seed)
        n_samples = len(self.imgs)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * split_ratio)

        if train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        self.transform = T.Compose([
            T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.imgs[real_idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # Return both continuous values and discrete classes
        latent_values = torch.tensor(self.latents_values[real_idx], dtype=torch.float32)
        latent_classes = torch.tensor(self.latents_classes[real_idx], dtype=torch.long)

        return img, {"values": latent_values, "classes": latent_classes}


def get_dsprites_loader(batch_size: int, train: bool = True, path: str = "./data/dsprites.npz", num_workers: int = 2):
    dataset = DSpritesDataset(path=path, train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=False, num_workers=num_workers)


# --------------------------
# 3D Shapes Dataset
# --------------------------
class Shapes3DDataset(Dataset):
    def __init__(self, path="data/3dshapes.h5", train=True, split_ratio=0.8, random_seed=42):
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

        # Split train/test deterministically
        np.random.seed(random_seed)
        n_samples = len(self.imgs)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * split_ratio)

        if train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

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
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.imgs[real_idx]  # (64, 64, 3)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Return both continuous values and discrete classes
        latent_values = torch.tensor(self.latents_values[real_idx], dtype=torch.float32)
        latent_classes = torch.tensor(self.latents_classes[real_idx], dtype=torch.long)

        return img, {"values": latent_values, "classes": latent_classes}


def get_shapes3d_loader(batch_size: int, train: bool = True, path: str = "./data/3dshapes.h5", num_workers: int = 2):
    dataset = Shapes3DDataset(path=path, train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=False, num_workers=num_workers)


# --------------------------
# Helpers & factor mapping
# --------------------------
# DSprites factor ordering: [shape, scale, orientation, posX, posY]
DSPRITES_FACTORS = ["shape", "scale", "orientation", "pos_x", "pos_y"]

# 3D Shapes factor ordering: [floor_hue, wall_hue, object_hue, scale, shape, orientation]
SHAPES3D_FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]

def get_dataset_type(data_path: str):
    """Detect dataset type from file path"""
    if data_path.endswith('.h5') or '3dshapes' in data_path.lower() or 'shapes3d' in data_path.lower():
        return 'shapes3d'
    elif data_path.endswith('.npz') or 'dsprites' in data_path.lower():
        return 'dsprites'
    else:
        # Default to dsprites for backward compatibility
        return 'dsprites'

def factor_name_to_index(cfg, factor_name: str, dataset_type: str = None):
    """Map factor name to column index in latents"""
    # Auto-detect dataset type if not provided
    if dataset_type is None:
        data_path = cfg.get("data", {}).get("path", "./data/dsprites.npz")
        dataset_type = get_dataset_type(data_path)
    
    if dataset_type == 'shapes3d':
        factor_map = {
            "floor_hue": 0,
            "wall_hue": 1,
            "object_hue": 2,
            "scale": 3,
            "shape": 4,
            "orientation": 5,
            "floor": 0,  # aliases
            "wall": 1,
            "object": 2,
        }
    else:  # dsprites
        factor_map = {
            "shape": 1,       # note: index 0 sometimes reserved for color in some variants
            "scale": 2,
            "orientation": 3,
            "pos_x": 4,
            "pos_y": 5
        }

    normalized_name = factor_name.lower().replace(" ", "_").replace("-", "_")
    if normalized_name in factor_map:
        return factor_map[normalized_name]

    for key in factor_map:
        if normalized_name in key or key in normalized_name:
            return factor_map[key]

    raise ValueError(f"Unknown factor name: {factor_name} for dataset {dataset_type}. Available: {list(factor_map.keys())}")


def get_target_tensor(y_batch, cfg, device, factor_for_classification=None, factor_for_regression=None):
    """
    y_batch: dict with 'values' (continuous) and 'classes' (discrete) tensors
    Returns:
      class_target: LongTensor (B,) or None
      reg_target: FloatTensor (B, k) or None
    """
    class_target = None
    reg_target = None

    # Detect dataset type for factor mapping
    data_path = cfg.get("data", {}).get("path", "./data/dsprites.npz")
    dataset_type = get_dataset_type(data_path)
    
    if factor_for_classification:
        idx = factor_name_to_index(cfg, factor_for_classification, dataset_type)
        class_target = y_batch["classes"][:, idx].to(device)

    if factor_for_regression:
        if isinstance(factor_for_regression, (list, tuple)):
            idxs = [factor_name_to_index(cfg, f, dataset_type) for f in factor_for_regression]
            reg_target = y_batch["values"][:, idxs].to(device)
        else:
            idx = factor_name_to_index(cfg, factor_for_regression, dataset_type)
            reg_target = y_batch["values"][:, idx].unsqueeze(1).to(device)

    return class_target, reg_target


# --------------------------
# Model factory
# --------------------------
def build_model(cfg, pretrained_encoder=None):
    mtype = cfg["model"]["type"].lower()
    latent_dim = cfg["model"].get("latent_dim", 10)
    num_classes = cfg["model"].get("num_classes", 3)
    depth = cfg["model"].get("depth", 4)
    freeze_encoder = cfg["model"].get("freeze_encoder", False)
    input_size = cfg["data"].get("input_size", 64) if "data" in cfg else 64
    input_channels = cfg["data"].get("input_channels", 1) if "data" in cfg else 1

    if mtype == "autoencoder":
        return Autoencoder(latent_dim=latent_dim)
    elif mtype in ["classifier", "classification"]:
        return Classifier(latent_dim=latent_dim, num_classes=num_classes, input_size=input_size, input_channels=input_channels)
    elif mtype in ["multi_task", "multitask"]:
        num_reg = cfg["model"].get("num_reg", 2)
        return MultiTaskAE(latent_dim=latent_dim, num_classes=num_classes, num_reg=num_reg, input_size=input_size, input_channels=input_channels)
    elif mtype in ["regularized", "regularizedae"]:
        reg_type = cfg["model"].get("reg_type", "l1")
        return RegularizedAE(latent_dim=latent_dim, num_classes=num_classes, reg_type=reg_type, input_size=input_size, input_channels=input_channels)
    elif mtype in ["branched", "branchedae"]:
        num_subspaces = cfg["model"].get("num_subspaces", 2)
        return BranchedAE(latent_dim=latent_dim, num_classes=num_classes, num_subspaces=num_subspaces, input_size=input_size, input_channels=input_channels)
    elif mtype in ["taskfactorized", "task_factorized"]:
        num_reg = cfg["model"].get("num_reg", 2)
        return TaskFactorizedAE(latent_dim=latent_dim, num_classes=num_classes, num_reg=num_reg, input_size=input_size, input_channels=input_channels)
    elif mtype in ["scaled", "scaledae"]:
        width_multiplier = cfg["model"].get("width_multiplier", 1)
        depth = cfg["model"].get("depth", 4)
        input_size = cfg["data"].get("input_size", 64) if "data" in cfg else 64
        return ScaledAE(latent_dim=latent_dim, num_classes=num_classes,
                        width_multiplier=width_multiplier, depth=depth, input_size=input_size, input_channels=input_channels)
    elif mtype in ["transfer", "transferae"]:
        return TransferAE(latent_dim=latent_dim, num_classes=num_classes,
                         freeze_encoder=freeze_encoder, pretrained_encoder=pretrained_encoder, input_size=input_size, input_channels=input_channels)
    elif mtype in ["moe", "mixture_of_experts"]:
        num_experts = cfg["model"].get("num_experts", 3)
        return MoEBranchedAE(latent_dim=latent_dim, num_classes=num_classes, num_experts=num_experts, input_size=input_size, input_channels=input_channels)

    raise ValueError(f"Unknown model type: {mtype}")


# --------------------------
# Training step (keeps your structure)
# --------------------------
def train_epoch(model, loader, optimizer, cfg, device, epoch=0):
    model.train()
    total_loss = 0.0
    loss_components = {}

    it = tqdm(loader, desc=f"Training Epoch {epoch+1}", leave=False)

    for batch_idx, (x, y) in enumerate(it):
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}

        optimizer.zero_grad()

        # Get target configuration
        training_cfg = cfg.get("training", {})
        loss_cfg = cfg.get("loss", {})
        class_factor = training_cfg.get("class_target") or loss_cfg.get("class_target")
        reg_factor = training_cfg.get("reg_target") or loss_cfg.get("reg_target")

        class_target, reg_target = get_target_tensor(y, cfg, device, class_factor, reg_factor)

        # Get loss weights
        weights = loss_cfg.get("weights", {})
        w_recon = weights.get("recon", 1.0)
        w_class = weights.get("class", 1.0)
        w_reg = weights.get("reg", 1.0)
        # optional sparsity conf (not used in all models)
        w_sparse = weights.get("sparse", 0.0)

        loss = torch.tensor(0.0, device=device)

        # Convenience lambdas
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()

        # Handle model-specific training behaviours
        mtype = cfg["model"]["type"].lower()

        # Many model.forward calls return (logits, z, ...) or (x_recon, z) etc.
        # We follow the previous logic you had and keep explicit checks per class.

        if isinstance(model, Autoencoder):
            x_recon, z = model(x)
            recon_loss = mse(x_recon, x)
            loss = w_recon * recon_loss
            loss_components["recon"] = recon_loss.item()

        elif isinstance(model, Classifier):
            if class_target is None:
                raise ValueError("Classifier requires class_target in config")
            logits, z = model(x)
            class_loss = ce(logits, class_target)
            loss = w_class * class_loss
            loss_components["class"] = class_loss.item()

        elif isinstance(model, MultiTaskAE):
            if class_target is None:
                raise ValueError("MultiTaskAE requires class_target in config")

            use_reconstruction = training_cfg.get("use_reconstruction", False)

            if use_reconstruction:
                logits, z, reg_out, x_recon = model(x, return_reconstruction=True)
                recon_loss = mse(x_recon, x)
                loss_components["recon"] = recon_loss.item()
                loss += w_recon * recon_loss
            else:
                logits, z, reg_out = model(x, return_reconstruction=False)

            class_loss = ce(logits, class_target)
            loss_components["class"] = class_loss.item()
            loss += w_class * class_loss

            if reg_target is not None:
                reg_loss = mse(reg_out, reg_target)
                loss_components["reg"] = reg_loss.item()
                loss += w_reg * reg_loss

        elif isinstance(model, RegularizedAE):
            # if class_target is None:
            #     raise ValueError("RegularizedAE requires class_target in config")

            use_reconstruction = training_cfg.get("use_reconstruction", False)
            if use_reconstruction:
                _, z, x_recon = model(x, return_reconstruction=True)
                recon_loss = mse(x_recon, x)
                loss_components["recon"] = recon_loss.item()
                loss += w_recon * recon_loss
            else:
                _, z = model(x, return_reconstruction=False)

            # class_loss = ce(logits, class_target)
            # loss_components["class"] = class_loss.item()
            # loss += w_class * class_loss

            # Add regularization
            reg_loss = model.get_regularization_loss(z)
            reg_weight = cfg.get("regularization", {}).get("weight", 0.01)
            loss_components["reg"] = reg_loss.item()
            loss += reg_weight * reg_loss

        elif isinstance(model, BranchedAE):
            # if class_target is None:
            #     raise ValueError("BranchedAE requires class_target in config")

            use_reconstruction = training_cfg.get("use_reconstruction", False)
            if use_reconstruction:
                _, z, zs, x_recon = model(x, return_reconstruction=True)
                recon_loss = mse(x_recon, x)
                loss_components["recon"] = recon_loss.item()
                loss += w_recon * recon_loss
            else:
                _, z, zs = model(x, return_reconstruction=False)

            # class_loss = ce(logits, class_target)
            # loss_components["class"] = class_loss.item()
            # loss += w_class * class_loss

        elif isinstance(model, TaskFactorizedAE):
            if class_target is None:
                raise ValueError("TaskFactorizedAE requires class_target in config")

            use_reconstruction = training_cfg.get("use_reconstruction", False)
            if use_reconstruction:
                logits, z_tuple,reg_out, x_recon = model(x, return_reconstruction=True)
                recon_loss = mse(x_recon, x)
                loss_components["recon"] = recon_loss.item()
                loss += w_recon * recon_loss
            else:
                logits,z_tuple, reg_out  = model(x, return_reconstruction=False)

            class_loss = ce(logits, class_target)
            loss_components["class"] = class_loss.item()
            loss += w_class * class_loss

            if reg_target is not None:
                reg_loss = mse(reg_out, reg_target)
                loss_components["reg"] = reg_loss.item()
                loss += w_reg * reg_loss

        elif isinstance(model, ScaledAE):
            # if class_target is None:
            #     raise ValueError("ScaledAE requires class_target in config")

            use_reconstruction = training_cfg.get("use_reconstruction", False)
            if use_reconstruction:
                _, z, x_recon = model(x, return_reconstruction=True)
                recon_loss = mse(x_recon, x)
                loss_components["recon"] = recon_loss.item()
                loss += w_recon * recon_loss
            else:
                _, z = model(x, return_reconstruction=False)

            # class_loss = ce(logits, class_target)
            # loss_components["class"] = class_loss.item()
            # loss += w_class * class_loss

        elif isinstance(model, TransferAE):
            # if class_target is None:
            #     raise ValueError("TransferAE requires class_target in config")

            # Support causal interventions through config (optional)
            intervention_cfg = training_cfg.get("intervention", None)
            if intervention_cfg:
                # Expect mask and values to be provided in cfg or training code; skip if not provided
                mask = intervention_cfg.get("mask", None)
                values = intervention_cfg.get("values", None)
                if mask is not None and values is not None:
                    # Prepare tensors (broadcast if needed)
                    mask_t = torch.tensor(mask, device=device).float()
                    values_t = torch.tensor(values, device=device).float()
                    # Model returns (logits_post, z_post, causal_loss) when asked
                    _, z_post, causal_loss = model(x, intervention=values_t, mask=mask_t, return_causal_loss=True)
                    # class_loss = ce(logits, class_target)
                    # loss_components["class"] = class_loss.item()
                    loss += w_class * class_loss
                    loss_components["causal"] = causal_loss.item()
                    loss += training_cfg.get("causal_weight", 0.0) * causal_loss
                else:
                    # fall back to normal forward
                    use_reconstruction = training_cfg.get("use_reconstruction", False)
                    if use_reconstruction:
                        _, z, x_recon = model(x, return_reconstruction=True)
                        recon_loss = mse(x_recon, x)
                        loss_components["recon"] = recon_loss.item()
                        loss += w_recon * recon_loss
                    else:
                        _, z = model(x, return_reconstruction=False)
                    # class_loss = ce(logits, class_target)
                    # loss_components["class"] = class_loss.item()
                    # loss += w_class * class_loss
            else:
                use_reconstruction = training_cfg.get("use_reconstruction", False)
                if use_reconstruction:
                    _, z, x_recon = model(x, return_reconstruction=True)
                    recon_loss = mse(x_recon, x)
                    loss_components["recon"] = recon_loss.item()
                    loss += w_recon * recon_loss
                else:
                    _, z = model(x, return_reconstruction=False)
                # class_loss = ce(logits, class_target)
                # loss_components["class"] = class_loss.item()
                # loss += w_class * class_loss

        elif isinstance(model, MoEBranchedAE):
            if class_target is None:
                raise ValueError("MoEBranchedAE requires class_target in config")

            use_reconstruction = training_cfg.get("use_reconstruction", False)
            if use_reconstruction:
                logits, z, expert_outputs, gate_weights, x_recon = model(x, return_reconstruction=True)
                recon_loss = mse(x_recon, x)
                loss_components["recon"] = recon_loss.item()
                loss += w_recon * recon_loss
            else:
                logits, z, expert_outputs, gate_weights = model(x, return_reconstruction=False)

            class_loss = ce(logits, class_target)
            loss_components["class"] = class_loss.item()
            loss += w_class * class_loss

            # Add load balancing loss for MoE
            load_balance_weight = cfg.get("loss", {}).get("load_balance", 0.01)
            if load_balance_weight > 0 and gate_weights is not None:
                avg_gate_weights = torch.mean(gate_weights, dim=0)
                num_experts = gate_weights.size(1)
                target_usage = torch.full_like(avg_gate_weights, 1.0 / num_experts)
                load_balance_loss = nn.MSELoss()(avg_gate_weights, target_usage)
                loss_components["load_balance"] = load_balance_loss.item()
                loss += load_balance_weight * load_balance_loss

        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        # Backward pass
        if loss.item() > 0:
            loss.backward()

            # Gradient clipping
            max_grad_norm = training_cfg.get("max_grad_norm", None)
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        desc = f"Loss: {total_loss/(batch_idx+1):.4f}"
        if loss_components:
            component_str = " | ".join([f"{k}: {v:.3f}" for k, v in loss_components.items()])
            desc += f" [{component_str}]"
        it.set_description(desc)

    avg_loss = total_loss / len(loader)
    return avg_loss, loss_components


# --------------------------
# Evaluation helper (updated)
# --------------------------
def collect_latents(model, loader, device, max_samples=None):
    """Collect latent representations and ground truth factors"""
    model.eval()
    zs = []
    values = []
    classes = []

    with torch.no_grad():
        count = 0
        for x, y in loader:
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}

            # Extract latent representation using flexible API
            z = None
            if hasattr(model, "get_latent"):
                z = model.get_latent(x)
            elif hasattr(model, "encoder"):
                z = model.encoder(x)
            else:
                out = model(x)
                if isinstance(out, tuple):
                    # Find the latent tensor (usually 2D)
                    for item in out:
                        if isinstance(item, torch.Tensor) and item.ndim == 2 and item.size(1) < 100:
                            z = item
                            break
                    if z is None:
                        z = out[1] if len(out) > 1 else out[0]
                else:
                    z = out

            if z is None:
                raise RuntimeError("Could not extract latent representation from model")

            zs.append(z.cpu())
            values.append(y["values"].cpu())
            classes.append(y["classes"].cpu())

            count += x.size(0)
            if max_samples and count >= max_samples:
                break

    zs = torch.cat(zs, dim=0).numpy()
    values = torch.cat(values, dim=0).numpy()
    classes = torch.cat(classes, dim=0).numpy()

    return zs, {"values": values, "classes": classes}


# --------------------------
# Main experiment runner (updated with unique folders & visuals)
# --------------------------
def run(config_path: str, data_path: str = "./data/dsprites.npz"):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Create unique experiment id (exp_id + config-hash + timestamp)
    exp_base = cfg.get("experiment_id", Path(config_path).stem)
    cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_id = f"{exp_base}_{cfg_hash}_{timestamp}"
    print(f"→ Running experiment: {exp_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"→ Using device: {device}")

    # Data loaders - detect dataset type
    batch_size = cfg.get("training", {}).get("batch_size", 128)
    # num_workers = cfg.get("training", {}).get("num_workers", 1)
    num_workers = 1
    dataset_type = get_dataset_type(data_path)
    
    if dataset_type == 'shapes3d':
        print(f"→ Loading 3D Shapes dataset from {data_path}")
        train_loader = get_shapes3d_loader(batch_size, train=True, path=data_path, num_workers=num_workers)
        test_loader = get_shapes3d_loader(batch_size, train=False, path=data_path, num_workers=num_workers)
    else:
        print(f"→ Loading dSprites dataset from {data_path}")
        train_loader = get_dsprites_loader(batch_size, train=True, path=data_path, num_workers=num_workers)
        test_loader = get_dsprites_loader(batch_size, train=False, path=data_path, num_workers=num_workers)

    print(f"→ Data loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")

    # Handle transfer learning - load pretrained encoder if needed
    # --- correct encoder loader ---
    pretrained_encoder = None
    if cfg["model"]["type"].lower().startswith("transfer"):
        pretrained_path = cfg.get("model", {}).get("pretrained_path")
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"→ Loading pretrained encoder weights from {pretrained_path}")
            state = torch.load(pretrained_path, map_location=device)

            # make encoder and load weights
            encoder = ConvEncoder(cfg["model"]["latent_dim"]).to(device)

            # Case 1: encoder-only state dict (conv.*, fc.*)
            if any(k.startswith("conv.") or k.startswith("fc.") for k in state.keys()):
                print("↪ Detected encoder-only checkpoint ✅")
                encoder.load_state_dict(state, strict=True)

            # Case 2: full AE checkpoint (encoder.* prefix)
            else:
                print("↪ Detected full Autoencoder checkpoint — extracting encoder weights ✅")
                filtered = {k.replace("encoder.", ""): v
                            for k, v in state.items()
                            if k.startswith("encoder.")}
                encoder.load_state_dict(filtered, strict=True)

            pretrained_encoder = encoder


    # Model
    model = build_model(cfg, pretrained_encoder=pretrained_encoder).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"→ Model: {cfg['model']['type']} with {total_params} parameters")

    # Optimizer
    optim_cfg = cfg.get("optimizer", {"type": "adam", "lr": 1e-3})
    opt_name = optim_cfg.get("type", "adam").lower()
    lr = optim_cfg.get("lr", 1e-3)
    weight_decay = optim_cfg.get("weight_decay", 0.0)

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Scheduler
    scheduler = None
    scheduler_cfg = optim_cfg.get("scheduler", {})
    if scheduler_cfg.get("type") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("training", {}).get("epochs", 50)
        )
    elif scheduler_cfg.get("type") == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_cfg.get("step_size", 20),
            gamma=scheduler_cfg.get("gamma", 0.1)
        )

    # Prepare results and checkpoint directories (unique)
    results_dir = Path("results") / exp_id
    ckpt_dir = Path("checkpoints") / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, indent=2)

    # Training loop
    epochs = cfg.get("training", {}).get("epochs", 50)
    eval_interval = cfg.get("training", {}).get("eval_interval", 10)
    history = {"train_loss": [], "loss_components": [], "metrics": []}
    best_metric = float('inf')

    print(f"→ Training for {epochs} epochs...")

    for epoch in range(epochs):
        try:
            avg_loss, loss_components = train_epoch(model, train_loader, optimizer, cfg, device, epoch)
        except Exception as e:
            print(f"Error during training epoch {epoch+1}: {e}")
            traceback.print_exc()
            break

        history["train_loss"].append(avg_loss)
        history["loss_components"].append(loss_components)

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        print(f"[{exp_id}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - LR: {current_lr:.2e}")
        if loss_components:
            print(f"  Components: {loss_components}")

        # Periodic evaluation
        if eval_interval and (epoch + 1) % eval_interval == 0:
            print(f"→ Evaluating at epoch {epoch+1}...")
            zs_val, labels_val = collect_latents(model, test_loader, device, max_samples=2000)

            try:
                # Try calling new signature evaluate_all_metrics(model, reps, labels, device=device, cfg=cfg)
                try:
                    metrics = evaluate_all_metrics(model, torch.from_numpy(zs_val), labels_val, device=device, cfg=cfg)
                except TypeError:
                    # Fall back to older signature
                    metrics = evaluate_all_metrics(model, torch.from_numpy(zs_val), labels_val, device=device)

                metrics["epoch"] = epoch + 1
                metrics["train_loss"] = avg_loss

                # Save metrics with epoch in name so different hyperparams / epochs don't overwrite
                metrics_filename = results_dir / f"metrics_epoch{epoch+1}.json"
                with open(metrics_filename, "w") as f:
                    json.dump(metrics, f, indent=2)

                history["metrics"].append(metrics)
                print(f"  Metrics: {metrics}")

                # Save checkpoint if metric improved (use total_score if present)
                metric_value = metrics.get("total_score", avg_loss)
                if metric_value < best_metric:
                    best_metric = metric_value
                    best_ckpt = ckpt_dir / f"{exp_id}_best.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'metrics': metrics,
                        'config': cfg
                    }, best_ckpt)

                # Save visual diagnostics for this epoch (prefix by epoch)
                try:
                    save_prefix = f"epoch{epoch+1}"
                    visualize_functional_modularity(torch.from_numpy(zs_val), labels_val,
                                                    save_dir=str(results_dir / "visuals"),
                                                    prefix=save_prefix)
                except Exception as e:
                    print(f"  Visual diagnostics failed: {e}")

            except Exception as e:
                print(f"  Evaluation failed: {e}")
                traceback.print_exc()
                continue

        # Save periodic checkpoint
        if (epoch + 1) % max(1, cfg.get("logging", {}).get("save_interval", 25)) == 0:
            ckpt_path = ckpt_dir / f"{exp_id}_epoch{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'config': cfg
            }, ckpt_path)

    # Final evaluation
    print("→ Final evaluation...")
    try:
        zs_val, labels_val = collect_latents(model, test_loader, device)
        try:
            final_metrics = evaluate_all_metrics(model, torch.from_numpy(zs_val), labels_val, device=device, cfg=cfg)
        except TypeError:
            final_metrics = evaluate_all_metrics(model, torch.from_numpy(zs_val), labels_val, device=device)

        final_metrics["epoch"] = epochs
        print(f"Final metrics: {final_metrics}")

        # Save final artifacts
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epochs,
            'metrics': final_metrics,
            'config': cfg
        }, ckpt_dir / f"{exp_id}_final.pt")

        with open(results_dir / "final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        # Save final visuals
        # try:
        #     visualize_functional_modularity(torch.from_numpy(zs_val), labels_val,
        #                                     save_dir=str(results_dir / "visuals"),
        #                                     prefix="final")
        # except Exception as e:
        #     print(f"Final visual diagnostics failed: {e}")

    except Exception as e:
        print(f"Final evaluation failed: {e}")
        final_metrics = {"error": str(e)}
        traceback.print_exc()

    # Save training history
    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"→ Experiment {exp_id} completed! Results saved in: {results_dir}")
    return final_metrics, history


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Train modularity emergence experiments")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML experiment config")
    parser.add_argument("--data", type=str, default="./data/dsprites.npz",
                        help="Path to dsprites dataset")
    args = parser.parse_args()

    # --------------------------------------------------------
    # ✅ Run experiment with safe error handling
    # --------------------------------------------------------
    try:
        metrics, history = run(args.config, args.data)
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        traceback.print_exc()

