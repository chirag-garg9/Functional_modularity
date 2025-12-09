import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------
# Utility Conv Blocks
# -------------------
def conv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, dropout=0.0):
    """Conv -> BatchNorm -> ReLU with optional dropout"""
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1):
    """Deconv -> BatchNorm -> ReLU"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

# -------------------
# Flexible Decoder (matches encoder depth/channels)
# -------------------
class FlexibleConvDecoder(nn.Module):
    def __init__(self, latent_dim, channels, final_size, out_channels=1):
        """
        channels: list of encoder channel sizes (e.g. [32,64,128,256]) -- decoder will invert them
        final_size: spatial size at start of decoder (e.g. 4 for 64x64 with depth=4)
        out_channels: #channels in reconstructed image (1 for dSprites, 3 for 3D Shapes)
        """
        super().__init__()
        self.final_size = final_size
        self.latent_dim = latent_dim
        self.channels = channels
        self.start_ch = channels[-1]
        self.out_channels = out_channels

        self.fc = nn.Linear(latent_dim, self.start_ch * final_size * final_size)

        # reverse channels to deconv sequence (except last -> output)
        deconv_layers = []
        rev = channels[::-1]
        for i in range(len(rev) - 1):
            deconv_layers.append(deconv_block(rev[i], rev[i + 1]))
        # final transpose conv to output 'out_channels'
        deconv_layers.append(nn.ConvTranspose2d(rev[-1], out_channels, 4, 2, 1))
        deconv_layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, z):
        h = self.fc(z).view(-1, self.start_ch, self.final_size, self.final_size)
        return self.deconv(h)

# -------------------
# Base Encoder / Decoder (Fixed for 64x64 input) - kept for compatibility
# -------------------
class ConvEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=10,
        input_size=64,
        dropout=0.0,
        channels=None,
        depth=4,
        in_channels=1,   # <-- NEW: configurable input channels
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        # default channels if not provided
        if channels is None:
            channels = [32, 64, 128, 256][:depth]
        self.channels = channels

        assert len(channels) == depth, "channels length must equal depth"

        convs = []
        in_ch = in_channels
        for ch in channels:
            convs.append(conv_block(in_ch, ch, dropout=dropout))
            in_ch = ch

        self.conv = nn.Sequential(*convs)
        final_size = input_size // (2 ** depth)
        self.final_size = final_size
        self.fc = nn.Linear(channels[-1] * final_size * final_size, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class ConvDecoder(nn.Module):
    """
    Backwards-compatible decoder that assumes a 4-layer encoder with last channel 256,
    for 64x64 outputs. Now supports configurable out_channels.
    """
    def __init__(self, latent_dim=10, output_size=64, out_channels=1):
        super().__init__()
        self.output_size = output_size
        self.out_channels = out_channels
        final_size = output_size // (2 ** 4)  # Should be 4 for 64x64
        self.fc = nn.Linear(latent_dim, 256 * final_size * final_size)
        self.final_size = final_size

        self.deconv = nn.Sequential(
            deconv_block(256, 128),                         # 4 -> 8
            deconv_block(128, 64),                          # 8 -> 16
            deconv_block(64, 32),                           # 16 -> 32
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, self.final_size, self.final_size)
        return self.deconv(h)

# -------------------
# E1: Baseline Autoencoder
# -------------------
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=10, input_channels=1, output_channels=1, input_size=64):
        """
        input_channels: 1 for dSprites, 3 for 3D Shapes
        output_channels: usually same as input_channels
        """
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, input_size=input_size, in_channels=input_channels)
        self.decoder = ConvDecoder(latent_dim, output_size=input_size, out_channels=output_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# -------------------
# E1: Baseline Classifier
# -------------------
class Classifier(nn.Module):
    def __init__(self, latent_dim=10, num_classes=3, input_channels=1, input_size=64):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, input_size=input_size, in_channels=input_channels)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z

# -------------------
# E2: Multi-task
# -------------------
class MultiTaskAE(nn.Module):
    """
    Multi-task autoencoder with:
      - shared encoder
      - reconstruction decoder
      - multiple classification heads (for discrete factors)
      - multiple regression heads (for continuous factors)

    Two ways to use:

    (a) Backwards-compatible (single head each):
        MultiTaskAE(latent_dim=10, num_classes=3, num_reg=2, ...)

        -> one classifier head: "cls"
           one regression head: "reg"

    (b) New, factor-aware mode:
        MultiTaskAE(
            latent_dim=10,
            class_factors={"shape": 3, "color": 8},
            reg_factors=["scale", "pos_x", "pos_y"],
            ...
        )

        -> classifier heads: "shape", "color"
           regression heads: "scale", "pos_x", "pos_y"
    """

    def __init__(
        self,
        latent_dim=10,
        # old-style:
        num_classes=None,
        num_reg=None,
        # new-style:
        class_factors=None,   # dict: name -> num_classes
        reg_factors=None,     # list of names for scalar regression
        input_channels=1,
        output_channels=1,
        input_size=64,
        hidden_cls=None,
        hidden_reg=None,
        dropout_cls=0.3,
        dropout_reg=0.3,
    ):
        super().__init__()

        # ---------- Encoder / decoder ----------
        self.encoder = ConvEncoder(
            latent_dim,
            input_size=input_size,
            in_channels=input_channels
        )
        self.decoder = ConvDecoder(
            latent_dim,
            output_size=input_size,
            out_channels=output_channels
        )

        # ---------- Factor configuration ----------
        # Backward-compatible single-head mode
        if class_factors is None and num_classes is not None:
            class_factors = {"cls": num_classes}
        if reg_factors is None and num_reg is not None:
            reg_factors = ["reg"] * num_reg  # treat as num_reg scalar factors

        # If still None, just no heads of that type
        if class_factors is None:
            class_factors = {}
        if reg_factors is None:
            reg_factors = []

        self.class_factors = class_factors       # dict name -> num_classes
        self.reg_factors = reg_factors           # list of names (scalar targets)

        # default hidden dims
        if hidden_cls is None:
            hidden_cls = max(16, latent_dim // 2)
        if hidden_reg is None:
            hidden_reg = max(16, latent_dim // 2)

        # ---------- Build classification heads ----------
        self.class_heads = nn.ModuleDict()
        for name, n_cls in self.class_factors.items():
            self.class_heads[name] = nn.Sequential(
                nn.Linear(latent_dim, hidden_cls),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_cls),
                nn.Linear(hidden_cls, n_cls),
            )

        # ---------- Build regression heads ----------
        self.reg_heads = nn.ModuleDict()
        for name in self.reg_factors:
            self.reg_heads[name] = nn.Sequential(
                nn.Linear(latent_dim, hidden_reg),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_reg),
                nn.Linear(hidden_reg, 1),  # scalar per reg factor
            )

    def forward(self, x, return_reconstruction=False):
        """
        Returns:
          class_outputs: dict name -> logits (B, n_classes)
          reg_outputs:   dict name -> (B, 1)
          z: latent (B, D)
          (optional) x_recon: (B, C, H, W)
        """
        z = self.encoder(x)

        # classification heads
        class_outputs = {}
        for name, head in self.class_heads.items():
            class_outputs[name] = head(z)

        # regression heads
        reg_outputs = {}
        for name, head in self.reg_heads.items():
            reg_outputs[name] = head(z)

        if return_reconstruction:
            x_recon = self.decoder(z)
            return class_outputs, z, reg_outputs, x_recon

        return class_outputs, z, reg_outputs


# -------------------
# E3: Regularized latent
# -------------------
class RegularizedAE(nn.Module):
    def __init__(
        self,
        latent_dim=10,
        num_classes=3,
        reg_type='l1',
        input_channels=1,
        output_channels=1,
        input_size=64,
    ):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, input_size=input_size, dropout=0.1, in_channels=input_channels)
        self.decoder = ConvDecoder(latent_dim, output_size=input_size, out_channels=output_channels)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(latent_dim // 2, num_classes)
        )
        # reg_type can be 'l1', 'l2', 'orthogonal', 'sparsity', 'variance', 'tc'
        self.reg_type = reg_type

    def forward(self, x, return_reconstruction=False):
        z = self.encoder(x)

        if return_reconstruction:
            x_recon = self.decoder(z)
            return None, z, x_recon
        return None, z

    def get_regularization_loss(self, z):
        """Compute various regularization losses on latent representations z (batch, latent_dim)."""
        if self.reg_type == 'l1':
            return torch.mean(torch.abs(z))
        elif self.reg_type == 'l2':
            return torch.mean(z ** 2)
        elif self.reg_type == 'orthogonal':
            z_norm = F.normalize(z, dim=0)
            correlation_matrix = torch.mm(z_norm.T, z_norm)  # [D, D]
            mask = ~torch.eye(correlation_matrix.size(0), dtype=torch.bool, device=z.device)
            return torch.mean(correlation_matrix[mask] ** 2)
        elif self.reg_type == 'sparsity':
            return torch.mean(torch.abs(z))
        elif self.reg_type == 'variance':
            var = torch.var(z, dim=0, unbiased=False) + 1e-8
            return torch.mean((var - 1.0) ** 2)
        elif self.reg_type == 'tc':
            z_centered = z - torch.mean(z, dim=0, keepdim=True)
            cov = (z_centered.T @ z_centered) / (z_centered.size(0))  # [D,D]
            mask = ~torch.eye(cov.size(0), dtype=torch.bool, device=z.device)
            return torch.mean(cov[mask] ** 2)
        else:
            return torch.tensor(0.0, device=z.device)

# -------------------
# E4: Subspace Modular
# -------------------
class BranchedAE(nn.Module):
    def __init__(
        self,
        latent_dim=10,
        num_classes=3,
        num_subspaces=2,
        input_channels=1,
        output_channels=1,
        input_size=64,
    ):
        super().__init__()
        assert latent_dim % num_subspaces == 0, "latent_dim must be divisible by num_subspaces"

        self.num_subspaces = num_subspaces
        self.subspace_dim = latent_dim // num_subspaces

        # Multiple encoders for different subspaces
        self.encoders = nn.ModuleList([
            ConvEncoder(self.subspace_dim, input_size=input_size, in_channels=input_channels)
            for _ in range(num_subspaces)
        ])

        # Shared decoder
        self.decoder = ConvDecoder(latent_dim, output_size=input_size, out_channels=output_channels)

        # Task-specific heads
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, x, return_reconstruction=False):
        zs = [enc(x) for enc in self.encoders]
        z = torch.cat(zs, dim=1)

        if return_reconstruction:
            x_recon = self.decoder(z)
            return None, z, zs, x_recon
        return None, z, zs

# -------------------
# E5: Task Factorization (base encoder + private encoders)
# -------------------
class TaskFactorizedAE(nn.Module):
    """
    Task-factorized AE:

    - Shared (base) encoder -> z_shared
    - Private encoder for all classification tasks -> z_class
    - Private encoder for all regression tasks    -> z_reg

    - Classification heads use [z_class, z_shared]
    - Regression heads use    [z_reg,   z_shared]

    Supports:
      class_factors: dict name -> num_classes
      reg_factors:   list of names (scalar)

    Backwards compatible:
      if class_factors is None and num_classes is provided -> one head "cls"
      if reg_factors   is None and num_reg is provided     -> num_reg copies of "reg"
    """
    def __init__(
        self,
        latent_dim=10,
        # backward compatible:
        num_classes=None,
        num_reg=None,
        # new:
        class_factors=None,   # dict: name -> num_classes
        reg_factors=None,     # list of names (scalar regression)
        private_ratio=0.4,
        shared_ratio=0.2,
        input_size=64,
        input_channels=1,
        output_channels=1,
        hidden_cls=None,
        hidden_reg=None,
    ):
        super().__init__()

        # ---------- Factor configuration ----------
        if class_factors is None and num_classes is not None:
            class_factors = {"cls": num_classes}
        if reg_factors is None and num_reg is not None:
            reg_factors = ["reg"] * num_reg

        if class_factors is None:
            class_factors = {}
        if reg_factors is None:
            reg_factors = []

        self.class_factors = class_factors
        self.reg_factors = reg_factors

        # ---------- Latent partitioning ----------
        latent_dim = int(latent_dim)
        shared_dim = max(1, int(latent_dim * shared_ratio))
        remaining = latent_dim - shared_dim
        private_each = max(1, remaining // 2)
        class_private = private_each
        reg_private = remaining - private_each

        self.dim_class = class_private
        self.dim_reg = reg_private
        self.dim_shared = shared_dim
        self.latent_dim = class_private + reg_private + shared_dim

        # ---------- Encoders ----------
        self.base_encoder = ConvEncoder(
            self.dim_shared,
            input_size=input_size,
            in_channels=input_channels,
        )
        self.enc_class = ConvEncoder(
            self.dim_class,
            input_size=input_size,
            in_channels=input_channels,
        )
        self.enc_reg = ConvEncoder(
            self.dim_reg,
            input_size=input_size,
            in_channels=input_channels,
        )

        # ---------- Decoder ----------
        self.decoder = ConvDecoder(
            self.latent_dim,
            output_size=input_size,
            out_channels=output_channels,
        )

        # ---------- Heads ----------
        if hidden_cls is None:
            hidden_cls = max(16, (self.dim_class + self.dim_shared) // 2)
        if hidden_reg is None:
            hidden_reg = max(16, (self.dim_reg + self.dim_shared) // 2)

        # classification: each head sees [z_class, z_shared]
        self.class_heads = nn.ModuleDict()
        cls_input_dim = self.dim_class + self.dim_shared
        for name, n_cls in self.class_factors.items():
            self.class_heads[name] = nn.Sequential(
                nn.Linear(cls_input_dim, hidden_cls),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_cls, n_cls),
            )

        # regression: each head sees [z_reg, z_shared]
        self.reg_heads = nn.ModuleDict()
        reg_input_dim = self.dim_reg + self.dim_shared
        for name in self.reg_factors:
            self.reg_heads[name] = nn.Sequential(
                nn.Linear(reg_input_dim, hidden_reg),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_reg, 1),
            )

    def forward(self, x, return_reconstruction=False):
        """
        Returns:
          class_outputs: dict name -> logits (B, n_classes)
          reg_outputs:   dict name -> (B, 1)
          z_all: full latent (B, latent_dim_total)
          (optional) x_recon: (B, C, H, W)
        """
        z_shared = self.base_encoder(x)
        z_class = self.enc_class(x)
        z_reg = self.enc_reg(x)

        z_for_class = torch.cat([z_class, z_shared], dim=1)
        z_for_reg = torch.cat([z_reg, z_shared], dim=1)
        z_all = torch.cat([z_class, z_reg, z_shared], dim=1)

        class_outputs = {}
        for name, head in self.class_heads.items():
            class_outputs[name] = head(z_for_class)

        reg_outputs = {}
        for name, head in self.reg_heads.items():
            reg_outputs[name] = head(z_for_reg)

        if return_reconstruction:
            x_recon = self.decoder(z_all)
            return class_outputs, z_all, reg_outputs, x_recon

        return class_outputs, z_all, reg_outputs


# -------------------
# E6: Scaled Capacity (depth scaling + flexible decoder)
# -------------------
class ScaledAE(nn.Module):
    def __init__(
        self,
        latent_dim=10,
        num_classes=3,
        width_multiplier=1.0,
        depth=4,
        input_size=64,
        input_channels=1,
        output_channels=1,
    ):
        """
        depth: number of conv downsampling layers (>=1)
        width_multiplier: scales channel counts
        """
        super().__init__()
        assert depth >= 1
        self.depth = depth
        # base channels sequence (can be extended/shrunk with depth)
        base_channels = [32, 64, 128, 256]
        channels = base_channels[:depth]
        channels = [max(8, int(ch * width_multiplier)) for ch in channels]
        self.channels = channels

        # Build encoder
        layers = []
        in_ch = input_channels
        for ch in channels:
            layers.append(conv_block(in_ch, ch))
            in_ch = ch
        self.encoder_conv = nn.Sequential(*layers)

        final_size = input_size // (2 ** depth)
        self.final_size = final_size
        self.fc_latent = nn.Linear(channels[-1] * final_size * final_size, latent_dim)

        # Decoder: use FlexibleConvDecoder to match channels & depth
        self.decoder = FlexibleConvDecoder(latent_dim, channels, final_size, out_channels=output_channels)

    def forward(self, x, return_reconstruction=False):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc_latent(h)

        if return_reconstruction:
            x_recon = self.decoder(z)
            return None, z, x_recon
        return None, z

# -------------------
# E7: Transfer Learning (with causal intervention)
# -------------------
class TransferAE(nn.Module):
    def __init__(
        self,
        latent_dim=10,
        num_classes=3,
        freeze_encoder=True,
        pretrained_encoder=None,
        pretrained_path=None,
        input_size=64,
        input_channels=1,
        output_channels=1,
    ):
        super().__init__()

        # ------------------------------
        # Init encoder
        # ------------------------------
        self.encoder = ConvEncoder(latent_dim, input_size=input_size, in_channels=input_channels)

        # ------------------------------
        # Load pretrained encoder weights
        # ------------------------------
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")

            # if state_dict was saved from model.encoder (no "encoder." prefix)
            if any(k.startswith("conv.") or k.startswith("fc.") for k in state.keys()):
                print("→ Loading encoder-only state dict")
                self.encoder.load_state_dict(state)
            else:
                # Full model dict with prefix (fallback)
                print("→ Loading full-model state dict (prefix expected)")
                self.encoder.load_state_dict({
                    k.replace("encoder.", ""): v for k, v in state.items()
                    if k.startswith("encoder.")
                })

        elif isinstance(pretrained_encoder, nn.Module):
            print("→ Using provided pretrained encoder module")
            self.encoder = pretrained_encoder

        # ------------------------------
        # Freeze if needed
        # ------------------------------
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ------------------------------
        # Decoder (for FT + reconstruction)
        # ------------------------------
        self.decoder = ConvDecoder(latent_dim, output_size=input_size, out_channels=output_channels)

    def forward(self, x, intervention=None, mask=None,
                return_reconstruction=False, return_causal_loss=False):
        z = self.encoder(x)
        causal_loss = None
        z_post = None

        if intervention is not None and mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(z.size(0), -1)
            if intervention.dim() == 1:
                intervention = intervention.unsqueeze(0).expand(z.size(0), -1)

            z_post = z * (1.0 - mask) + intervention * mask
            causal_loss = F.mse_loss(z * (1 - mask), z_post * (1 - mask))

            if return_reconstruction:
                x_recon = self.decoder(z_post)
                return (None, z_post, causal_loss, x_recon) if return_causal_loss else (None, z_post, x_recon)
            return (None, z_post, causal_loss) if return_causal_loss else (None, z_post)

        if return_reconstruction:
            x_recon = self.decoder(z)
            return None, z, x_recon
        return None, z

    def unfreeze_encoder(self, layers_to_unfreeze=-1):
        if layers_to_unfreeze == -1:
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            layers = list(self.encoder.conv.children())
            for layer in layers[-layers_to_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True

# -------------------
# Mixture of Experts Extension for E4
# -------------------
class MoEBranchedAE(nn.Module):
    def __init__(
        self,
        latent_dim=10,
        num_classes=3,
        num_experts=3,
        expert_dim=None,
        input_channels=1,
        output_channels=1,
        input_size=64,
    ):
        super().__init__()
        if expert_dim is None:
            expert_dim = latent_dim // num_experts

        self.num_experts = num_experts
        self.expert_dim = expert_dim

        # Multiple expert encoders
        self.experts = nn.ModuleList([
            ConvEncoder(expert_dim, input_size=input_size, in_channels=input_channels)
            for _ in range(num_experts)
        ])

        # Gating network - produce logits over experts from a small conv encoder + fc
        self.gate_encoder = ConvEncoder(num_experts, input_size=input_size, in_channels=input_channels)
        self.gate_softmax = nn.Softmax(dim=1)

        # Final representation + decoder
        total_dim = num_experts * expert_dim
        self.decoder = ConvDecoder(total_dim, output_size=input_size, out_channels=output_channels)

    def forward(self, x, return_reconstruction=False):
        # Get expert representations
        expert_outputs = [expert(x) for expert in self.experts]  # list of [B, expert_dim]

        # Get gating weights
        gate_logits = self.gate_encoder(x)  # [B, num_experts]
        gate_weights = self.gate_softmax(gate_logits)

        # Weighted combination of expert outputs
        weighted_outputs = []
        for i, expert_out in enumerate(expert_outputs):
            weight = gate_weights[:, i:i+1]  # [batch_size, 1]
            weighted_outputs.append(weight * expert_out)

        z = torch.cat(weighted_outputs, dim=1)

        if return_reconstruction:
            x_recon = self.decoder(z)
            return None, z, expert_outputs, gate_weights, x_recon

        return None, z, expert_outputs, gate_weights
