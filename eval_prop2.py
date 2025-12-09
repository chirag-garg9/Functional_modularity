# eval_modularity_suite.py
#
# Rigorous evaluation of 4 properties of functional modularity:
#   P1: Selective encoding (own vs complement vs random subspaces)
#   P2: Low interference (causal subspace ablations vs random)
#   P3: Factorwise contribution (factor heads + combiner vs baseline head)
#   P4: Localized adaptation (fine-tuning factor-specific outputs with gradient masking)
#
# Assumes dSprites-like factors and a models module with your architectures.
# Only mild hyperparameters are used (number of random subsets, epochs, etc.)
# and are declared in one place.

import os
import json
import yaml
import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision.transforms as T
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge

# Make CUDA errors synchronous for easier debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# -------------------------------------------------------------------------
# IMPORT YOUR MODELS
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# GLOBAL CONSTANTS / HYPERPARAMETERS
# -------------------------------------------------------------------------

FACTORS = ["shape", "scale", "rot", "pos_x", "pos_y"]
FACTOR_TYPES = {
    "shape": "categorical",
    "scale": "continuous",
    "rot": "continuous",
    "pos_x": "continuous",
    "pos_y": "continuous",
}
# index in cont vector for each continuous factor
CONTINUOUS_INDICES = {"scale": 0, "rot": 1, "pos_x": 2, "pos_y": 3}

# MI / NMI estimation
N_BINS = 10
NMI_JUNK_THRESHOLD = 1e-3  # below this max NMI: dimension treated as junk

# Randomization / statistics
N_RANDOM_SUBSPACES = 20  # for P1 random-subspace baseline
N_RANDOM_ABLATIONS = 20  # for P2 random ablation baseline
ASSIGN_FRACTION = 0.5    # fraction of samples used for assignment (subspace discovery)

# Localized adaptation
ADAPT_SIZE = 50000
ADAPT_TEST_SIZE = 10000
BASE_TEST_SIZE = 10000
ADAPT_EPOCHS = 5
ADAPT_LR = 1e-3
ADAPT_BATCH_SIZE = 512

# Multi-task heads (P3)
P3_EPOCHS = 20
P3_LR = 1e-3
P3_BATCH_SIZE = 512

# -------------------------------------------------------------------------
# DATASET: dSprites
# -------------------------------------------------------------------------

class DSpritesDataset(torch.utils.data.Dataset):
    """
    Expects dsprites.npz with:
      - imgs: (N, H, W)
      - latents_values: (N, 6) [color, shape, scale, orientation, posX, posY]
      - latents_classes: (N, 6)
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


# -------------------------------------------------------------------------
# DATASET: 3D Shapes
# -------------------------------------------------------------------------

class Shapes3DDataset(torch.utils.data.Dataset):
    """
    3D Shapes dataset from DeepMind.
    Expects 3dshapes.h5 with:
      - images: (N, 64, 64, 3) RGB images
      - labels: (N, 6) [floor_hue, wall_hue, object_hue, scale, shape, orientation]
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


# -------------------------------------------------------------------------
# Dataset type detection
# -------------------------------------------------------------------------
def get_dataset_type(data_path: str):
    """Detect dataset type from file path"""
    if data_path.endswith('.h5') or '3dshapes' in data_path.lower() or 'shapes3d' in data_path.lower():
        return 'shapes3d'
    elif data_path.endswith('.npz') or 'dsprites' in data_path.lower():
        return 'dsprites'
    else:
        # Default to dsprites for backward compatibility
        return 'dsprites'


# -------------------------------------------------------------------------
# MODEL BUILDING / LATENT EXTRACTION
# -------------------------------------------------------------------------

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
        input_size = cfg.get("data", {}).get("input_size", 64)
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


def collect_latents(model, loader, device, max_samples=None):
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

    return (
        torch.cat(zs).numpy(),
        {
            "values": torch.cat(vals).numpy(),
            "classes": torch.cat(cls).numpy(),
        },
    )


# -------------------------------------------------------------------------
# FACTOR HANDLING & NMI-BASED SUBSPACE DISCOVERY
# -------------------------------------------------------------------------

def build_factor_dict(labels):
    vals = labels["values"]  # (N, 6)
    cls = labels["classes"]  # (N, 6)
    return {
        "shape": cls[:, 1].astype(int),
        "scale": vals[:, 2],
        "rot": vals[:, 3],
        "pos_x": vals[:, 4],
        "pos_y": vals[:, 5],
    }


def discretize_continuous(arr, n_bins=N_BINS):
    arr = np.asarray(arr)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.quantile(arr, quantiles)
    bins[0] -= 1e-9
    bins[-1] += 1e-9
    dig = np.digitize(arr, bins[1:-1])
    return dig


def compute_nmi_matrix(latents, factors_dict):
    N, d = latents.shape
    nmi_matrix = np.zeros((d, len(FACTORS)), dtype=np.float32)

    disc_factors = {}
    for f in FACTORS:
        y = np.asarray(factors_dict[f])
        if FACTOR_TYPES[f] == "continuous":
            disc_factors[f] = discretize_continuous(y, n_bins=N_BINS)
        else:
            disc_factors[f] = y

    for i in range(d):
        z_i = discretize_continuous(latents[:, i], n_bins=N_BINS)
        for j, f in enumerate(FACTORS):
            nmi_matrix[i, j] = normalized_mutual_info_score(z_i, disc_factors[f])

    return nmi_matrix


def build_subspaces_from_nmi(nmi_matrix, threshold=NMI_JUNK_THRESHOLD):
    d, _ = nmi_matrix.shape
    subspaces = {f: [] for f in FACTORS}
    junk = []
    for i in range(d):
        col = nmi_matrix[i]
        max_val = col.max()
        if max_val < threshold:
            junk.append(i)
        else:
            j = int(col.argmax())
            subspaces[FACTORS[j]].append(i)
    return subspaces, junk


def split_assignment_eval(N, frac_assign=ASSIGN_FRACTION, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_assign = int(frac_assign * N)
    assign_idx = idx[:n_assign]
    eval_idx = idx[n_assign:]
    return assign_idx, eval_idx


# -------------------------------------------------------------------------
# SIMPLE PROBES (USED IN P1 & P2)
# -------------------------------------------------------------------------

def train_probe(X, y, factor_type):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X[:, None]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if factor_type == "categorical":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        return clf, {"acc": float(acc)}
    else:
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_val)
        mse = float(((pred - y_val) ** 2).mean())
        var = float(np.var(y_val) + 1e-8)
        r2 = float(1.0 - mse / var)
        return reg, {"mse": mse, "r2": r2}


# -------------------------------------------------------------------------
# PROPERTY 1: SELECTIVE ENCODING
# -------------------------------------------------------------------------

def evaluate_property1_selective_encoding(latents_eval, labels_eval, subspaces):
    """
    For each factor f:
      - Train probe on S_f, S_not_f, and |S_f|-dim random subsets.
      - Compute performance gaps and p-values vs random.
    """
    N, d = latents_eval.shape
    factors_dict = build_factor_dict(labels_eval)
    rng = np.random.RandomState(0)

    results = {}

    for f in FACTORS:
        S_f = subspaces.get(f, [])
        S_not_f = [i for i in range(d) if i not in S_f]

        if len(S_f) == 0 or len(S_not_f) == 0:
            # degenerate case
            results[f] = {
                "own_perf": None,
                "comp_perf": None,
                "rand_perf_mean": None,
                "rand_perf_std": None,
                "delta_own_vs_comp": 0.0,
                "delta_own_vs_rand": 0.0,
                "p_value_own_gt_rand": 1.0,
            }
            continue

        X_own = latents_eval[:, S_f]
        X_comp = latents_eval[:, S_not_f]
        y = factors_dict[f]
        factor_type = FACTOR_TYPES[f]

        # own-subspace probe
        _, stats_own = train_probe(X_own, y, factor_type)
        # complement-subspace probe
        _, stats_comp = train_probe(X_comp, y, factor_type)

        # performance scalar
        def perf(stats):
            if factor_type == "categorical":
                return stats["acc"]
            else:
                return stats["r2"]

        own_perf = perf(stats_own)
        comp_perf = perf(stats_comp)

        # random-subspace baseline
        rand_perfs = []
        for _ in range(N_RANDOM_SUBSPACES):
            rand_dims = rng.choice(d, size=len(S_f), replace=False)
            X_rand = latents_eval[:, rand_dims]
            _, stats_rand = train_probe(X_rand, y, factor_type)
            rand_perfs.append(perf(stats_rand))
        rand_perfs = np.array(rand_perfs)
        rand_mean = float(rand_perfs.mean())
        rand_std = float(rand_perfs.std() + 1e-8)

        delta_own_vs_comp = float(own_perf - comp_perf)
        delta_own_vs_rand = float(own_perf - rand_mean)
        p_val = float((rand_perfs >= own_perf).mean())

        results[f] = {
            "own_perf": float(own_perf),
            "comp_perf": float(comp_perf),
            "rand_perf_mean": rand_mean,
            "rand_perf_std": rand_std,
            "delta_own_vs_comp": delta_own_vs_comp,
            "delta_own_vs_rand": delta_own_vs_rand,
            "p_value_own_gt_rand": p_val,
            "factor_type": factor_type,
        }

    return {
        "selective_encoding_per_factor": results,
    }


# -------------------------------------------------------------------------
# PROPERTY 2: LOW INTERFERENCE (CAUSAL ABLATIONS)
# -------------------------------------------------------------------------

def evaluate_property2_interference(latents_eval, labels_eval, subspaces):
    """
    Interference matrix based on causal ablations of factor subspaces.
    Also includes random-ablation baseline to calibrate interference.
    """
    N, d = latents_eval.shape
    factors_dict = build_factor_dict(labels_eval)

    # 1) baseline probes on full z
    baseline_probes = {}
    baseline_perf = {}
    baseline_err = {}

    for f in FACTORS:
        probe, stats = train_probe(latents_eval, factors_dict[f], FACTOR_TYPES[f])
        baseline_probes[f] = probe
        if FACTOR_TYPES[f] == "categorical":
            acc = stats["acc"]
            baseline_perf[f] = acc
            baseline_err[f] = 1.0 - acc
        else:
            r2 = stats["r2"]
            baseline_perf[f] = r2
            baseline_err[f] = 1.0 - r2

    # 2) subspace ablation interference
    interference = {f: {} for f in FACTORS}
    for g in FACTORS:
        S_g = subspaces.get(g, [])
        z_ab = latents_eval.copy()
        if len(S_g) > 0:
            z_ab[:, S_g] = 0.0
        for f in FACTORS:
            probe = baseline_probes[f]
            y = factors_dict[f]
            if FACTOR_TYPES[f] == "categorical":
                pred = probe.predict(z_ab)
                acc = float((pred == y).mean())
                err_ab = 1.0 - acc
            else:
                pred = probe.predict(z_ab)
                mse = float(((pred - y) ** 2).mean())
                var = float(np.var(y) + 1e-8)
                r2 = 1.0 - mse / var
                err_ab = 1.0 - r2
            interference[f][g] = float(err_ab - baseline_err[f])

    # 3) random ablation baseline
    rng = np.random.RandomState(1)
    random_interference = {f: {g: [] for g in FACTORS} for f in FACTORS}

    for _ in range(N_RANDOM_ABLATIONS):
        for g in FACTORS:
            S_g = subspaces.get(g, [])
            k = len(S_g)
            if k == 0:
                continue
            rand_dims = rng.choice(d, size=k, replace=False)
            z_rand_ab = latents_eval.copy()
            z_rand_ab[:, rand_dims] = 0.0

            for f in FACTORS:
                probe = baseline_probes[f]
                y = factors_dict[f]
                if FACTOR_TYPES[f] == "categorical":
                    pred = probe.predict(z_rand_ab)
                    acc = float((pred == y).mean())
                    err_ab = 1.0 - acc
                else:
                    pred = probe.predict(z_rand_ab)
                    mse = float(((pred - y) ** 2).mean())
                    var = float(np.var(y) + 1e-8)
                    r2 = 1.0 - mse / var
                    err_ab = 1.0 - r2
                random_interference[f][g].append(err_ab - baseline_err[f])

    # now aggregate random baseline
    random_baseline_stats = {f: {} for f in FACTORS}
    for f in FACTORS:
        for g in FACTORS:
            vals = np.array(random_interference[f][g]) if len(random_interference[f][g]) > 0 else np.array([0.0])
            random_baseline_stats[f][g] = {
                "mean": float(vals.mean()),
                "std": float(vals.std() + 1e-8),
            }

    # 4) summary modularity score
    mat = np.zeros((len(FACTORS), len(FACTORS)), dtype=np.float32)
    for i, f in enumerate(FACTORS):
        for j, g in enumerate(FACTORS):
            mat[i, j] = float(interference[f][g])

    self_effect = float(np.diag(mat).mean())
    off_mask = np.ones_like(mat, dtype=bool)
    np.fill_diagonal(off_mask, False)
    cross_effect = float(mat[off_mask].mean()) if off_mask.sum() > 0 else 0.0
    interference_ratio = cross_effect / (self_effect + 1e-8) if self_effect > 0 else float("inf")
    modularity_score = 1.0 / (1.0 + max(interference_ratio, 0.0))

    summary = {
        "self_effect_mean": self_effect,
        "cross_effect_mean": cross_effect,
        "interference_ratio": interference_ratio,
        "modularity_score": modularity_score,
    }

    return {
        "interference_matrix": interference,
        "random_interference_baseline": random_baseline_stats,
        "summary": summary,
        "baseline_perf": baseline_perf,
    }


# -------------------------------------------------------------------------
# PROPERTY 3: FACTORWISE CONTRIBUTION (HEADS + COMBINER)
# -------------------------------------------------------------------------

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, n_shape_classes=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.shape_head = nn.Linear(hidden_dim, n_shape_classes)
        self.reg_head = nn.Linear(hidden_dim, 4)

    def forward(self, z):
        h = self.net(z)
        shape_logits = self.shape_head(h)
        cont = self.reg_head(h)
        return shape_logits, cont


class FactorEmbeddingHead(nn.Module):
    def __init__(self, input_dim, emb_dim=8, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CombinerHead(nn.Module):
    def __init__(self, input_dim, n_shape_classes=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.shape_head = nn.Linear(hidden_dim, n_shape_classes)
        self.reg_head = nn.Linear(hidden_dim, 4)

    def forward(self, emb_concat):
        h = self.net(emb_concat)
        shape_logits = self.shape_head(h)
        cont = self.reg_head(h)
        return shape_logits, cont


def build_latent_tensors(latents_eval, factors_dict_eval, device):
    z = torch.tensor(latents_eval, dtype=torch.float32, device=device)
    shape = torch.tensor(factors_dict_eval["shape"], dtype=torch.long, device=device)
    cont_vals = np.stack(
        [
            factors_dict_eval["scale"],
            factors_dict_eval["rot"],
            factors_dict_eval["pos_x"],
            factors_dict_eval["pos_y"],
        ],
        axis=-1,
    )
    cont = torch.tensor(cont_vals, dtype=torch.float32, device=device)
    return z, shape, cont


def split_train_val_test(labels_shape, val_ratio=0.2, test_ratio=0.2, seed=42):
    N = len(labels_shape)
    idx = np.arange(N)
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_ratio, stratify=labels_shape, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio / (1 - test_ratio),
        stratify=labels_shape[train_val_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def compute_multitask_losses(shape_logits, cont_pred, shape, cont):
    ce = nn.CrossEntropyLoss()(shape_logits, shape)
    mse = nn.MSELoss()(cont_pred, cont)
    return ce, mse, ce + mse


def eval_multitask_model(model, z, shape, cont, idx):
    model.eval()
    with torch.no_grad():
        out_shape, out_cont = model(z[idx])
        ce, mse, _ = compute_multitask_losses(out_shape, out_cont, shape[idx], cont[idx])
        pred_shape = out_shape.argmax(dim=-1)
        acc = (pred_shape == shape[idx]).float().mean().item()
        diff = out_cont - cont[idx]
        mse_per_dim = (diff ** 2).mean(dim=0).cpu().numpy()
        return {
            "shape_acc": float(acc),
            "shape_ce": float(ce.item()),
            "cont_mse": float(mse.item()),
            "cont_mse_per_dim": mse_per_dim.tolist(),
        }


def train_baseline_multitask_head(latents_eval, factors_dict_eval, device):
    z, shape, cont = build_latent_tensors(latents_eval, factors_dict_eval, device)
    N, d = z.shape
    train_idx, val_idx, test_idx = split_train_val_test(shape.cpu().numpy())

    n_classes = int(shape.max().item()) + 1
    model = MultiTaskMLP(d, n_shape_classes=n_classes, hidden_dim=128).to(device)
    opt = optim.Adam(model.parameters(), lr=P3_LR)

    train_ds = TensorDataset(z[train_idx], shape[train_idx], cont[train_idx])
    train_loader = DataLoader(train_ds, batch_size=P3_BATCH_SIZE, shuffle=True)

    best_val = float("inf")
    best_state = None

    for epoch in range(P3_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_z, batch_shape, batch_cont in train_loader:
            opt.zero_grad()
            out_shape, out_cont = model(batch_z)
            _, _, loss = compute_multitask_losses(out_shape, out_cont, batch_shape, batch_cont)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_z.size(0)

        val_stats = eval_multitask_model(model, z, shape, cont, val_idx)
        val_loss = val_stats["shape_ce"] + val_stats["cont_mse"]
        print(
            f"[P3 Baseline] Epoch {epoch+1}/{P3_EPOCHS} "
            f"TrainLoss={total_loss/len(train_idx):.4f} ValLoss={val_loss:.4f} "
            f"ValAcc={val_stats['shape_acc']:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_stats = eval_multitask_model(model, z, shape, cont, test_idx)

    return model, test_stats, (z, shape, cont, train_idx, val_idx, test_idx)


def evaluate_property3_factor_contribution(latents_eval, factors_dict_eval, subspaces, device):
    """
    Baseline MultiTaskMLP vs factorized heads+combiner + drop-expert.
    """
    print("\n[P3] Training baseline multi-task head...")
    baseline_model, baseline_test_stats, split_data = train_baseline_multitask_head(
        latents_eval, factors_dict_eval, device
    )
    z_all, shape_all, cont_all, train_idx, val_idx, test_idx = split_data
    N, d = z_all.shape

    # Build factor heads
    factor_heads = {}
    emb_dim = 8
    hidden_dim = 64
    for f in FACTORS:
        S_f = subspaces.get(f, [])
        input_dim = max(1, len(S_f))
        factor_heads[f] = FactorEmbeddingHead(input_dim, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)

    # Combiner
    comb_input_dim = len(FACTORS) * emb_dim
    n_classes = int(shape_all.max().item()) + 1
    combiner = CombinerHead(comb_input_dim, n_shape_classes=n_classes, hidden_dim=128).to(device)

    params = list(combiner.parameters())
    for f in FACTORS:
        params += list(factor_heads[f].parameters())
    opt = optim.Adam(params, lr=P3_LR)

    train_ds = TensorDataset(z_all[train_idx], shape_all[train_idx], cont_all[train_idx])
    train_loader = DataLoader(train_ds, batch_size=P3_BATCH_SIZE, shuffle=True)

    def forward_factorized(z):
        embs = []
        for f in FACTORS:
            S_f = subspaces.get(f, [])
            if len(S_f) == 0:
                emb = torch.zeros(z.size(0), emb_dim, device=z.device)
            else:
                z_f = z[:, S_f]
                emb = factor_heads[f](z_f)
            embs.append(emb)
        emb_concat = torch.cat(embs, dim=-1)
        return combiner(emb_concat)

    best_val = float("inf")
    best_state = None

    print("[P3] Training factorized heads + combiner...")
    for epoch in range(P3_EPOCHS):
        combiner.train()
        for f in FACTORS:
            factor_heads[f].train()

        total_loss = 0.0
        for batch_z, batch_shape, batch_cont in train_loader:
            opt.zero_grad()
            out_shape, out_cont = forward_factorized(batch_z)
            _, _, loss = compute_multitask_losses(out_shape, out_cont, batch_shape, batch_cont)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_z.size(0)

        # validation
        combiner.eval()
        for f in FACTORS:
            factor_heads[f].eval()
        with torch.no_grad():
            out_shape, out_cont = forward_factorized(z_all[val_idx])
            ce, mse, _ = compute_multitask_losses(
                out_shape, out_cont, shape_all[val_idx], cont_all[val_idx]
            )
            val_loss = ce.item() + mse.item()

        print(
            f"[P3 Factorized] Epoch {epoch+1}/{P3_EPOCHS} "
            f"TrainLoss={total_loss/len(train_idx):.4f} ValLoss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "combiner": {k: v.cpu() for k, v in combiner.state_dict().items()},
                "factor_heads": {
                    f: {k: v.cpu() for k, v in factor_heads[f].state_dict().items()}
                    for f in FACTORS
                },
            }

    if best_state is not None:
        combiner.load_state_dict({k: v.to(device) for k, v in best_state["combiner"].items()})
        for f in FACTORS:
            factor_heads[f].load_state_dict(
                {k: v.to(device) for k, v in best_state["factor_heads"][f].items()}
            )

    # Final factored performance on test
    combiner.eval()
    for f in FACTORS:
        factor_heads[f].eval()
    with torch.no_grad():
        out_shape_f, out_cont_f = forward_factorized(z_all[test_idx])
        ce_f, mse_f, _ = compute_multitask_losses(
            out_shape_f, out_cont_f, shape_all[test_idx], cont_all[test_idx]
        )
        pred_shape_f = out_shape_f.argmax(dim=-1)
        acc_f = (pred_shape_f == shape_all[test_idx]).float().mean().item()
        diff_f = out_cont_f - cont_all[test_idx]
        mse_per_dim = (diff_f ** 2).mean(dim=0).cpu().numpy()

    factored_test_stats = {
        "shape_acc": float(acc_f),
        "shape_ce": float(ce_f.item()),
        "cont_mse": float(mse_f.item()),
        "cont_mse_per_dim": mse_per_dim.tolist(),
    }

    # Recovery ratios
    recovery = {
        "shape_acc_ratio": factored_test_stats["shape_acc"] / max(1e-8, baseline_test_stats["shape_acc"]),
        "cont_mse_ratio": baseline_test_stats["cont_mse"] / max(1e-8, factored_test_stats["cont_mse"]),
    }

    # Drop-expert analysis on test set
    contribution_matrix = {t: {} for t in FACTORS}

    def eval_with_drop(drop_factor=None):
        combiner.eval()
        for f in FACTORS:
            factor_heads[f].eval()
        with torch.no_grad():
            embs = []
            for f in FACTORS:
                S_f = subspaces.get(f, [])
                if len(S_f) == 0:
                    emb = torch.zeros(z_all[test_idx].size(0), emb_dim, device=z_all.device)
                else:
                    z_f = z_all[test_idx][:, S_f]
                    if drop_factor is not None and f == drop_factor:
                        emb = torch.zeros(z_all[test_idx].size(0), emb_dim, device=z_all.device)
                    else:
                        emb = factor_heads[f](z_f)
                embs.append(emb)
            emb_concat = torch.cat(embs, dim=-1)
            out_shape, out_cont = combiner(emb_concat)
            ce, mse, _ = compute_multitask_losses(out_shape, out_cont, shape_all[test_idx], cont_all[test_idx])
            pred_shape = out_shape.argmax(dim=-1)
            acc = (pred_shape == shape_all[test_idx]).float().mean().item()
            diff = out_cont - cont_all[test_idx]
            mse_per_dim = (diff ** 2).mean(dim=0).cpu().numpy()
            cont_mse = float(mse_per_dim.mean())
        return {
            "shape_acc": float(acc),
            "shape_ce": float(ce.item()),
            "cont_mse": cont_mse,
        }

    full_stats = eval_with_drop(drop_factor=None)

    for drop_f in FACTORS:
        stats_drop = eval_with_drop(drop_factor=drop_f)
        for t in FACTORS:
            if t == "shape":
                loss_full = 1.0 - full_stats["shape_acc"]
                loss_drop = 1.0 - stats_drop["shape_acc"]
            else:
                loss_full = full_stats["cont_mse"] / 4.0
                loss_drop = stats_drop["cont_mse"] / 4.0
            contribution_matrix[t][drop_f] = float(loss_drop - loss_full)

    factor_contribution_scores = {}
    for f in FACTORS:
        self_effect = contribution_matrix[f][f]
        leak_effects = []
        for t in FACTORS:
            if t == f:
                continue
            leak_effects.append(max(0.0, contribution_matrix[t][f]))
        leak_mean = float(np.mean(leak_effects)) if len(leak_effects) > 0 else 0.0
        factor_contribution_scores[f] = {
            "self_contribution": float(self_effect),
            "leakage_mean": leak_mean,
        }

    return {
        "baseline_multitask_test": baseline_test_stats,
        "factored_multitask_test": factored_test_stats,
        "recovery": recovery,
        "drop_expert_contribution_matrix": contribution_matrix,
        "factor_contribution_scores": factor_contribution_scores,
    }, baseline_model


# -------------------------------------------------------------------------
# PROPERTY 4: LOCALIZED ADAPTATION
# -------------------------------------------------------------------------

def build_adaptation_indices(vals, factor_name, seed=123,
                             adapt_size=ADAPT_SIZE, test_size=ADAPT_TEST_SIZE,
                             base_size=BASE_TEST_SIZE):
    """
    Build indices for adaptation train, adaptation-test, and base-test sets,
    conditioned on factor-specific shifts.
    """
    N = vals.shape[0]
    rng = np.random.RandomState(seed)
    idx_all = np.arange(N)

    if factor_name == "shape":
        shape = vals[:, 1].astype(int)
        uniq = np.unique(shape)
        target = uniq[-1]
        mask = shape == target
    elif factor_name == "scale":
        v = vals[:, 2]
        q = np.quantile(v, 0.66)
        mask = v >= q
    elif factor_name == "rot":
        v = vals[:, 3]
        q = np.quantile(v, 0.5)
        mask = v >= q
    elif factor_name == "pos_x":
        v = vals[:, 4]
        q = np.quantile(v, 0.5)
        mask = v >= q
    else:  # pos_y
        v = vals[:, 5]
        q = np.quantile(v, 0.5)
        mask = v >= q

    idx_factor = idx_all[mask]
    if len(idx_factor) == 0:
        idx_factor = idx_all.copy()
    rng.shuffle(idx_factor)

    adapt_size = min(adapt_size, len(idx_factor))
    adapt_idx = idx_factor[:adapt_size]

    remaining = idx_factor[adapt_size:]
    rng.shuffle(remaining)
    adapt_test_size = min(test_size, len(remaining))
    if adapt_test_size == 0:
        adapt_test_idx = adapt_idx.copy()
    else:
        adapt_test_idx = remaining[:adapt_test_size]

    rng.shuffle(idx_all)
    base_test_size = min(base_size, len(idx_all))
    base_test_idx = idx_all[:base_test_size]

    return adapt_idx, adapt_test_idx, base_test_idx


def evaluate_property4_localized_adaptation(latents_eval, labels_eval,
                                            factors_dict_eval, base_model,
                                            device):
    """
    Compare localized adaptation (output-slice-only) vs unconstrained adaptation.
    """
    print("\n[P4] Localized adaptation experiments...")

    z_all, shape_all, cont_all = build_latent_tensors(latents_eval, factors_dict_eval, device)
    vals_all = labels_eval["values"]
    N, d = z_all.shape
    n_classes = int(shape_all.max().item()) + 1

    def eval_model(model, idx):
        return eval_multitask_model(model, z_all, shape_all, cont_all, idx)

    adaptation_results = {}

    for f in FACTORS:
        print(f"[P4] Factor {f}...")
        adapt_idx, adapt_test_idx, base_test_idx = build_adaptation_indices(vals_all, f)

        # Base stats (same for both regimes before adaptation)
        stats_before_adapt = eval_model(base_model, adapt_test_idx)
        stats_before_base = eval_model(base_model, base_test_idx)

        # Two copies: localized vs unconstrained
        # 1) localized (gradient masking on outputs)
        model_loc = MultiTaskMLP(d, n_shape_classes=n_classes, hidden_dim=128).to(device)
        model_loc.load_state_dict(base_model.state_dict())
        opt_loc = optim.Adam(model_loc.parameters(), lr=ADAPT_LR)

        # 2) unconstrained
        model_uncon = MultiTaskMLP(d, n_shape_classes=n_classes, hidden_dim=128).to(device)
        model_uncon.load_state_dict(base_model.state_dict())
        opt_uncon = optim.Adam(model_uncon.parameters(), lr=ADAPT_LR)

        adapt_ds = TensorDataset(z_all[adapt_idx], shape_all[adapt_idx], cont_all[adapt_idx])
        adapt_loader = DataLoader(adapt_ds, batch_size=ADAPT_BATCH_SIZE, shuffle=True)

        idx_cont = CONTINUOUS_INDICES.get(f, None)

        # training loop
        for epoch in range(ADAPT_EPOCHS):
            # localized
            model_loc.train()
            for batch_z, batch_shape, batch_cont in adapt_loader:
                opt_loc.zero_grad()
                out_shape, out_cont = model_loc(batch_z)
                if f == "shape":
                    loss = nn.CrossEntropyLoss()(out_shape, batch_shape)
                else:
                    loss = nn.MSELoss()(out_cont[:, idx_cont], batch_cont[:, idx_cont])
                loss.backward()

                # gradient masking for localized adaptation
                if f == "shape":
                    if model_loc.reg_head.weight.grad is not None:
                        model_loc.reg_head.weight.grad.zero_()
                    if model_loc.reg_head.bias.grad is not None:
                        model_loc.reg_head.bias.grad.zero_()
                else:
                    if model_loc.reg_head.weight.grad is not None:
                        gW = model_loc.reg_head.weight.grad
                        mask = torch.zeros_like(gW)
                        mask[idx_cont, :] = 1.0
                        gW.mul_(mask)
                    if model_loc.reg_head.bias.grad is not None:
                        gb = model_loc.reg_head.bias.grad
                        maskb = torch.zeros_like(gb)
                        maskb[idx_cont] = 1.0
                        gb.mul_(maskb)
                    if model_loc.shape_head.weight.grad is not None:
                        model_loc.shape_head.weight.grad.zero_()
                    if model_loc.shape_head.bias.grad is not None:
                        model_loc.shape_head.bias.grad.zero_()

                opt_loc.step()

            # unconstrained
            model_uncon.train()
            for batch_z, batch_shape, batch_cont in adapt_loader:
                opt_uncon.zero_grad()
                out_shape, out_cont = model_uncon(batch_z)
                if f == "shape":
                    loss = nn.CrossEntropyLoss()(out_shape, batch_shape)
                else:
                    loss = nn.MSELoss()(out_cont[:, idx_cont], batch_cont[:, idx_cont])
                loss.backward()
                opt_uncon.step()

        # after adaptation
        stats_after_adapt_loc = eval_model(model_loc, adapt_test_idx)
        stats_after_base_loc = eval_model(model_loc, base_test_idx)
        stats_after_adapt_uncon = eval_model(model_uncon, adapt_test_idx)
        stats_after_base_uncon = eval_model(model_uncon, base_test_idx)

        def compute_gain_and_leak(factor, stats_before_adapt, stats_after_adapt,
                                  stats_before_base, stats_after_base):
            if factor == "shape":
                gain = stats_after_adapt["shape_acc"] - stats_before_adapt["shape_acc"]
                leak = max(0.0, stats_after_base["cont_mse"] - stats_before_base["cont_mse"])
            else:
                k = CONTINUOUS_INDICES[factor]
                before_mse_k = stats_before_adapt["cont_mse_per_dim"][k]
                after_mse_k = stats_after_adapt["cont_mse_per_dim"][k]
                gain = before_mse_k - after_mse_k
                # leakage on base: shape + other continuous dims
                err_before_shape = 1.0 - stats_before_base["shape_acc"]
                err_after_shape = 1.0 - stats_after_base["shape_acc"]
                leak_shape = max(0.0, err_after_shape - err_before_shape)

                before_mse_other = np.array(stats_before_base["cont_mse_per_dim"])
                after_mse_other = np.array(stats_after_base["cont_mse_per_dim"])
                mask = np.ones(4, dtype=bool)
                mask[k] = False
                diff_mse = after_mse_other[mask] - before_mse_other[mask]
                leak_other = float(np.mean(np.maximum(diff_mse, 0.0)))
                leak = leak_shape + leak_other
            la_score = gain / (gain + leak + 1e-8) if gain > 0 else 0.0
            return float(gain), float(leak), float(la_score)

        G_loc, leak_loc, LA_loc = compute_gain_and_leak(
            f, stats_before_adapt, stats_after_adapt_loc,
            stats_before_base, stats_after_base_loc
        )
        G_uncon, leak_uncon, LA_uncon = compute_gain_and_leak(
            f, stats_before_adapt, stats_after_adapt_uncon,
            stats_before_base, stats_after_base_uncon
        )

        adaptation_results[f] = {
            "localized": {
                "G_target": G_loc,
                "leak": leak_loc,
                "LA_score": LA_loc,
                "before_adapt_stats": stats_before_adapt,
                "after_adapt_stats": stats_after_adapt_loc,
                "before_base_stats": stats_before_base,
                "after_base_stats": stats_after_base_loc,
            },
            "unconstrained": {
                "G_target": G_uncon,
                "leak": leak_uncon,
                "LA_score": LA_uncon,
                "before_adapt_stats": stats_before_adapt,
                "after_adapt_stats": stats_after_adapt_uncon,
                "before_base_stats": stats_before_base,
                "after_base_stats": stats_after_base_uncon,
            },
        }

    overall_loc = float(np.mean([v["localized"]["LA_score"] for v in adaptation_results.values()]))
    overall_uncon = float(np.mean([v["unconstrained"]["LA_score"] for v in adaptation_results.values()]))

    return {
        "per_factor": adaptation_results,
        "overall_LA_localized": overall_loc,
        "overall_LA_unconstrained": overall_uncon,
    }


# -------------------------------------------------------------------------
# MAIN EVAL PIPELINE
# -------------------------------------------------------------------------

def run_eval(config_path, ckpt_path, data_path, device=None, max_samples=None):
    cfg = yaml.safe_load(open(config_path))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bs = cfg.get("training", {}).get("batch_size", 128)
    dataset_type = get_dataset_type(data_path)
    
    if dataset_type == 'shapes3d':
        print(f"→ Loading 3D Shapes dataset from {data_path}")
        loader = get_shapes3d_loader(bs, data_path, shuffle=False)
    else:
        print(f"→ Loading dSprites dataset from {data_path}")
        loader = get_dsprites_loader(bs, data_path, shuffle=False)

    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    exp_name = Path(config_path).stem
    out_dir = Path(f"eval_results/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[0] Extracting latents...")
    all_latents, all_labels = collect_latents(model, loader, device, max_samples=max_samples)
    N_total = all_latents.shape[0]
    print(f"Collected {N_total} latent samples")

    assign_idx, eval_idx = split_assignment_eval(N_total, frac_assign=ASSIGN_FRACTION, seed=0)
    latents_assign = all_latents[assign_idx]
    labels_assign = {
        "values": all_labels["values"][assign_idx],
        "classes": all_labels["classes"][assign_idx],
    }
    latents_eval = all_latents[eval_idx]
    labels_eval = {
        "values": all_labels["values"][eval_idx],
        "classes": all_labels["classes"][eval_idx],
    }

    # Subspace discovery (assignment split)
    print("\n[1] Subspace discovery on assignment split...")
    factors_assign = build_factor_dict(labels_assign)
    nmi_assign = compute_nmi_matrix(latents_assign, factors_assign)
    subspaces, junk_assign = build_subspaces_from_nmi(nmi_assign, threshold=NMI_JUNK_THRESHOLD)

    with open(out_dir / "subspaces_assignment.json", "w") as f:
        json.dump(
            {
                "nmi_matrix_assignment": nmi_assign.tolist(),
                "subspaces": subspaces,
                "junk_dims_assignment": junk_assign,
            },
            f,
            indent=2,
        )

    factors_eval = build_factor_dict(labels_eval)

    # P1: selective encoding
    print("\n[2] Property 1: Selective encoding...")
    p1_results = evaluate_property1_selective_encoding(latents_eval, labels_eval, subspaces)
    with open(out_dir / "property1_selective_encoding.json", "w") as f:
        json.dump(p1_results, f, indent=2)

    # P2: low interference
    print("\n[3] Property 2: Low interference...")
    p2_results = evaluate_property2_interference(latents_eval, labels_eval, subspaces)
    with open(out_dir / "property2_interference.json", "w") as f:
        json.dump(p2_results, f, indent=2)

    # P3: factorwise contribution
    print("\n[4] Property 3: Factorwise contribution...")
    p3_results, baseline_head = evaluate_property3_factor_contribution(
        latents_eval, factors_eval, subspaces, device
    )
    with open(out_dir / "property3_factor_contribution.json", "w") as f:
        json.dump(p3_results, f, indent=2)

    # P4: localized adaptation
    print("\n[5] Property 4: Localized adaptation...")
    p4_results = evaluate_property4_localized_adaptation(
        latents_eval, labels_eval, factors_eval, baseline_head, device
    )
    with open(out_dir / "property4_localized_adaptation.json", "w") as f:
        json.dump(p4_results, f, indent=2)

    # Minimal summary
    summary = {
        "P1": p1_results["selective_encoding_per_factor"],
        "P2_modularity_score": p2_results["summary"]["modularity_score"],
        "P3_recovery": p3_results["recovery"],
        "P4_overall_LA_localized": p4_results["overall_LA_localized"],
        "P4_overall_LA_unconstrained": p4_results["overall_LA_unconstrained"],
    }
    with open(out_dir / "summary_modularity_suite.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved all modularity evaluation results to: {out_dir}")
    print(json.dumps(summary, indent=2))


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full modularity property evaluation suite")
    parser.add_argument("--config", required=True, help="YAML config used for training")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint with model_state_dict")
    parser.add_argument("--data", default="./data/dsprites.npz", help="Path to dsprites npz")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional limit on number of dsprites samples for evaluation")
    args = parser.parse_args()

    try:
        dev = torch.device(args.device) if args.device else None
        run_eval(args.config, args.checkpoint, args.data,
                 device=dev, max_samples=args.max_samples)
    except Exception as e:
        print("❌ Eval failed:", e)
        traceback.print_exc()
