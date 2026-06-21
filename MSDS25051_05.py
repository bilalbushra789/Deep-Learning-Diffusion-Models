"""
MSDS25051_05.py
=================
Image Generation Using Diffusion Models — single self-contained script.

This file contains everything required
    SECTION 1: Data Loader              (reads images, builds the training set)
    SECTION 2: Forward Diffusion Process (image -> noise, Algorithm 1 ingredients)
    SECTION 3: Denoising Model (U-Net)   (eps_theta(x_t, t))
    SECTION 4: Training Loop (Algorithm 1) + CLI entry point

"""

import argparse
import json
import math
import os
import random
import time
from glob import glob

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image


# =====================================================================
# SECTION 1: DATA LOADER
# =====================================================================
#
# Reads images from a folder structured as:
#     root/
#         ClassA/
#             img1.jpg
#             img2.jpg
#         ClassB/
#             ...
#
# Picks a subset of classes (default 5) and a subset of images per class
# (default 20).
#
# Images are resized and scaled to [-1, 1], the standard input range
# used for DDPM training (matches the Gaussian prior N(0, I) used in
# the forward process).

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_available_classes(root_dir):
    """Return sorted list of class (sub-folder) names found in root_dir."""
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


class AnimalDiffusionDataset(Dataset):
    """
    Picks `num_classes` classes and `images_per_class` images per class
    from `root_dir`, and serves them as normalized tensors in [-1, 1].

    Parameters
    ----------
    root_dir : str
        Path to the folder that contains one sub-folder per animal class.
    image_size : int
        Output (square) resolution fed to the model.
    num_classes : int
        How many classes to sample from (assignment default: 5).
    images_per_class : int
        How many images to take from each chosen class (assignment default: 20).
    selected_classes : list[str] or None
        If given, use exactly these class names instead of randomly picking
        `num_classes` of them. Useful for reproducibility.
    seed : int
        Random seed used when randomly choosing classes/images.
    """

    def __init__(
        self,
        root_dir,
        image_size=64,
        num_classes=5,
        images_per_class=20,
        selected_classes=None,
        seed=42,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        rng = random.Random(seed)

        all_classes = list_available_classes(root_dir)
        if len(all_classes) == 0:
            raise RuntimeError(f"No class folders found inside {root_dir}")

        if selected_classes is not None:
            self.classes = list(selected_classes)
        else:
            self.classes = sorted(rng.sample(all_classes, k=min(num_classes, len(all_classes))))

        self.image_paths = []
        self.labels = []
        for class_idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            files = [
                f for f in sorted(glob(os.path.join(class_dir, "*")))
                if f.lower().endswith(IMG_EXTENSIONS)
            ]
            rng.shuffle(files)
            chosen = files[:images_per_class]
            if len(chosen) < images_per_class:
                print(
                    f"[WARN] class '{cls}' only has {len(chosen)} images "
                    f"(< requested {images_per_class})."
                )
            self.image_paths.extend(chosen)
            self.labels.extend([class_idx] * len(chosen))

        if len(self.image_paths) == 0:
            raise RuntimeError("No images collected — check root_dir / class names.")

        # Necessary transformations before feeding data to the diffusion process:
        #   1. Resize to a fixed square size
        #   2. Convert to tensor in [0, 1]
        #   3. Normalize to [-1, 1] (mean=0.5, std=0.5 per channel)
        #   4. Light augmentation (random horizontal flip) since training set is tiny
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def class_name(self, label_idx):
        return self.classes[label_idx]


def denormalize(tensor):
    """Map a tensor from [-1, 1] back to [0, 1] for visualization/saving."""
    return (tensor.clamp(-1, 1) + 1) / 2


# =====================================================================
# SECTION 2: FORWARD DIFFUSION PROCESS (Algorithm 1 ingredients)
# =====================================================================
#
# IMPORTANT (per assignment): noise is NEVER applied to the image
# directly in a naive "img + noise" sense. Instead we use the
# closed-form forward process:
#
#     x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
#
# which is mathematically equivalent to injecting Gaussian noise t
# times sequentially (Markov chain q(x_t | x_t-1)), but computed in one
# shot.

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    """Standard linear noise schedule used in the original DDPM paper."""
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
    """
    Wraps the noise schedule and exposes:
      - q_sample:      forward process x_0 -> x_t  (used in training)
      - training_loss: Algorithm 1 (one gradient step)
      - p_sample:      one reverse step x_t -> x_t-1 (used in sampling)
      - p_sample_loop: full reverse process x_T -> x_0 (used in testing)
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Precompute terms reused at every step (standard DDPM bookkeeping)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior variance for the reverse process q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    @staticmethod
    def _extract(values, t, shape):
        """Gather values at timestep indices t and reshape for broadcasting
        against an image batch of shape `shape` = (B, C, H, W)."""
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    # ---------- Forward process: q(x_t | x_0) ----------
    def q_sample(self, x0, t, noise=None):
        """
        Closed-form forward diffusion (Algorithm 1, line 5 ingredients):
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        This represents T sequential noise-injection steps collapsed into
        one equation — NOT raw "image + noise".
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_ac * x0 + sqrt_omac * noise, noise

    def forward_diffusion_sequence(self, x0, num_shown=10):
        """
        Produce a sequence of progressively noisier versions of x0, evenly
        spaced across [0, T-1], purely for visualization (Figure 1 in the
        assignment PDF: 'Sample Images of adding noise on T steps').
        Does not affect training.
        """
        device = x0.device
        steps = torch.linspace(0, self.timesteps - 1, num_shown).long().to(device)
        imgs = []
        for t_val in steps:
            t = torch.full((x0.shape[0],), t_val.item(), device=device, dtype=torch.long)
            x_t, _ = self.q_sample(x0, t)
            imgs.append(x_t)
        return imgs, steps.tolist()

    # ---------- Training objective: Algorithm 1 ----------
    def training_loss(self, model, x0, t, loss_type="l2"):
        """
        One iteration of Algorithm 1:
            1. x0 ~ q(x0)                      (given, a real batch)
            2. t ~ Uniform({1, ..., T})         (given as input)
            3. eps ~ N(0, I)
            4. x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps
            5. loss = || eps - eps_theta(x_t, t) ||^2
        """
        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t, noise=noise)
        predicted_noise = model(x_t, t)

        if loss_type == "l1":
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == "l2":
            loss = F.mse_loss(predicted_noise, noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        return loss

    # ---------- Reverse process: p_theta(x_{t-1} | x_t) ----------
    @torch.no_grad()
    def p_sample(self, model, x_t, t, t_index):
        """One reverse-diffusion step (denoising), used inside the sampling loop."""
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_omac_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        predicted_noise = model(x_t, t)

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_omac_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_var_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, return_all_steps=False):
        """
        Full reverse process: start from pure Gaussian noise x_T and
        iteratively denoise down to x_0. This is the "test function which
        will accept noise and create an image from it" required by the
        assignment.
        """
        img = torch.randn(shape, device=device)
        imgs = [img.cpu()]

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            if return_all_steps:
                imgs.append(img.cpu())

        if return_all_steps:
            return img, imgs
        return img

    @torch.no_grad()
    def sample_from_given_noise(self, model, noise, return_all_steps=False):
        """
        Same as p_sample_loop, but starts from a user-provided noise tensor
        instead of sampling a fresh one. Used by test_single_sample.ipynb so
        the user can literally hand in noise and get an image back, as the
        assignment specifies.
        """
        device = noise.device
        img = noise
        imgs = [img.cpu()]

        for i in reversed(range(self.timesteps)):
            t = torch.full((img.shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            if return_all_steps:
                imgs.append(img.cpu())

        if return_all_steps:
            return img, imgs
        return img


# =====================================================================
# SECTION 3: DENOISING MODEL — U-Net (eps_theta(x_t, t))
# =====================================================================
#
# A compact U-Net with sinusoidal timestep embeddings, residual blocks,
# GroupNorm + SiLU activations, and self-attention at the bottleneck.
# This follows the standard DDPM backbone design (Ho et al. 2020) scaled
# down to fit a small dataset (5 classes x 20 images) and a single T4
# GPU on Colab.
#
# Every design choice is commented since the assignment explicitly
# states: "Every step of the model chooses carefully. Which layer,
# activation function is used (when and where). "

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes the integer timestep t into a continuous vector using sine/cosine
    functions of different frequencies (same idea as Transformer positional
    encodings). This lets the network distinguish between e.g. t=5 (almost
    no noise) and t=950 (almost pure noise) and condition its denoising
    behaviour accordingly.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Two 3x3 conv layers with GroupNorm + SiLU (the activation combo used in
    the original DDPM U-Net; SiLU/Swish trains more stably than ReLU for
    diffusion models because it's smooth everywhere). The timestep embedding
    is injected by adding a learned projection of it to the feature map
    after the first conv, so every block "knows" the current noise level.
    A residual (skip) connection helps gradient flow given how deep the
    effective computation graph becomes over many diffusion steps.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_bias = self.time_mlp(t_emb)
        h = h + time_bias[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class SelfAttention(nn.Module):
    """
    Standard non-local self-attention block applied at the bottleneck
    (lowest spatial resolution), where it's computationally cheap and most
    useful for capturing global structure (overall animal shape/pose)
    rather than local texture.
    """

    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.softmax((q.transpose(1, 2) @ k) / math.sqrt(C), dim=-1)
        out = v @ attn.transpose(1, 2)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    """
    U-Net used as eps_theta(x_t, t): predicts the noise that was added to
    produce x_t from x_0, conditioned on the timestep t.

    Architecture (for image_size=64, base_channels=64):
        Encoder: 64 -> 128 -> 256   (with downsampling between stages)
        Bottleneck: 256 channels + self-attention
        Decoder: 256 -> 128 -> 64   (with upsampling + skip connections)
        Output: 1x1 conv back to 3 channels (predicted noise, same shape as input)

    """

    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256, channel_mults=(1, 2, 4)):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * m for m in channel_mults]  # e.g. [64, 128, 256]

        # ---------------- Encoder ----------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            self.down_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
            self.downsamples.append(
                Downsample(out_ch) if i < len(channels) - 1 else nn.Identity()
            )
            in_ch = out_ch

        # ---------------- Bottleneck ----------------
        self.bottleneck1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.bottleneck_attn = SelfAttention(in_ch)
        self.bottleneck2 = ResidualBlock(in_ch, in_ch, time_emb_dim)

        # ---------------- Decoder ----------------
        # `cur_ch` tracks the channel count flowing through the decoder
        # (starts at the bottleneck's channel count). At each stage we
        # concatenate the matching encoder skip connection (out_ch channels)
        # before the residual block, then project down to out_ch.
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        cur_ch = in_ch  # = channels[-1], the bottleneck width
        for i, out_ch in enumerate(reversed_channels):
            # input channels = current decoder width + skip-connection channels (concat)
            self.up_blocks.append(ResidualBlock(cur_ch + out_ch, out_ch, time_emb_dim))
            self.upsamples.append(
                Upsample(out_ch) if i < len(reversed_channels) - 1 else nn.Identity()
            )
            cur_ch = out_ch

        self.final_block = nn.Sequential(
            nn.GroupNorm(8, base_channels * channel_mults[0]),
            nn.SiLU(),
            nn.Conv2d(base_channels * channel_mults[0], in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            x = block(x, t_emb)
            skips.append(x)
            x = down(x)

        x = self.bottleneck1(x, t_emb)
        x = self.bottleneck_attn(x)
        x = self.bottleneck2(x, t_emb)

        for block, up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
            x = up(x)

        return self.final_block(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# SECTION 4: TRAINING LOOP (Algorithm 1) + CLI ENTRY POINT
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train a DDPM-style diffusion model on animal images.")
    p.add_argument("--data_root", type=str, required=True,
                    help="Path to folder containing one sub-folder per animal class.")
    p.add_argument("--output_dir", type=str, default="./outputs",
                    help="Where to save checkpoints, samples, and logs.")
    p.add_argument("--num_classes", type=int, default=5,
                    help="How many animal classes to train on.")
    p.add_argument("--images_per_class", type=int, default=20,
                    help="How many images per chosen class to use for training.")
    p.add_argument("--selected_classes", type=str, default=None,
                    help="Comma-separated explicit class names, e.g. 'Bear,Lion,Tiger,Deer,Bird'. "
                         "Overrides --num_classes random selection if given.")
    p.add_argument("--image_size", type=int, default=64,
                    help="Square resolution images are resized to.")
    p.add_argument("--timesteps", type=int, default=1000,
                    help="Number of diffusion steps T.")
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2", "huber"])
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--sample_every", type=int, default=25,
                    help="Generate a sample grid every N epochs.")
    p.add_argument("--num_samples", type=int, default=8,
                    help="How many images to generate at each sampling checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_noise_progression_figure(dataset, diffusion, output_dir, num_shown=10):
    """Figure 1 equivalent: show one real image getting progressively noisier."""
    img, _ = dataset[0]
    img = img.unsqueeze(0).to(diffusion.device)
    imgs, steps = diffusion.forward_diffusion_sequence(img, num_shown=num_shown)

    fig, axes = plt.subplots(1, num_shown, figsize=(num_shown * 1.6, 2))
    for ax, x_t, t_val in zip(axes, imgs, steps):
        vis = denormalize(x_t[0]).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        ax.imshow(vis)
        ax.set_title(f"t={t_val}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Forward diffusion: image -> noise")
    plt.tight_layout()
    path = os.path.join(output_dir, "forward_noise_progression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved forward-noise figure to {path}")


def save_loss_curve(losses, output_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training loss over time")
    plt.grid(alpha=0.3)
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------------- Data ----------------
    selected_classes = (
        [c.strip() for c in args.selected_classes.split(",")]
        if args.selected_classes else None
    )
    dataset = AnimalDiffusionDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        selected_classes=selected_classes,
        seed=args.seed,
    )
    print(f"Dataset: {len(dataset)} images across classes: {dataset.classes}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    # ---------------- Diffusion + Model ----------------
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    model = UNet(base_channels=args.base_channels).to(device)
    print(f"Model parameter count: {count_parameters(model):,}")

    save_noise_progression_figure(dataset, diffusion, args.output_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------------- Training loop (Algorithm 1) ----------------
    all_losses = []
    epoch_losses = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        num_batches = 0

        for x0, _ in dataloader:
            x0 = x0.to(device)
            batch_size = x0.shape[0]

            # Algorithm 1, line 3: t ~ Uniform({1, ..., T})
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

            loss = diffusion.training_loss(model, x0, t, loss_type=args.loss_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())
            running_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = running_loss / max(num_batches, 1)
        epoch_losses.append(avg_epoch_loss)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}/{args.epochs}] avg_loss={avg_epoch_loss:.5f} elapsed={elapsed/60:.1f}min")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            shape = (args.num_samples, 3, args.image_size, args.image_size)
            samples = diffusion.p_sample_loop(model, shape, device)
            samples = denormalize(samples)
            grid = make_grid(samples, nrow=4)
            save_image(grid, os.path.join(samples_dir, f"epoch_{epoch:04d}.png"))
            model.train()
            print(f"  Saved sample grid for epoch {epoch}")

    # ---------------- Save artifacts ----------------
    save_loss_curve(all_losses, args.output_dir)

    ckpt_path = os.path.join(args.output_dir, "diffusion_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "image_size": args.image_size,
            "timesteps": args.timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "base_channels": args.base_channels,
            "classes": dataset.classes,
        },
    }, ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path}")

    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump({
            "epoch_losses": epoch_losses,
            "all_step_losses": all_losses,
            "classes_used": dataset.classes,
            "num_images": len(dataset),
            "args": vars(args),
        }, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
