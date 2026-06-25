"""MSDS25051_05
============================================================================


SECTION 1: Data Loader
SECTION 2: Forward / Reverse Diffusion Process
SECTION 3: Denoising Model (U-Net)
SECTION 4: Training Loop + CLI entry point
"""

import argparse
import copy
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

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_available_classes(root_dir):
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


class AnimalDiffusionDataset(Dataset):
    def __init__(self, root_dir, image_size=64, num_classes=5,
                 images_per_class=20, selected_classes=None, seed=42):
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
                print(f"[WARN] class '{cls}' only has {len(chosen)} images.")
            self.image_paths.extend(chosen)
            self.labels.extend([class_idx] * len(chosen))
        if len(self.image_paths) == 0:
            raise RuntimeError("No images collected — check root_dir / class names.")

        
        pre_size = int(image_size * 1.15)  # resize a bit larger, then crop down
        self.transform = transforms.Compose([
            transforms.Resize((pre_size, pre_size)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]

    def class_name(self, label_idx):
        return self.classes[label_idx]


def denormalize(tensor):
    """Map tensor from [-1, 1] -> [0, 1] for visualization."""
    return (tensor.clamp(-1, 1) + 1) / 2


# =====================================================================
# SECTION 2: FORWARD / REVERSE DIFFUSION PROCESS
# =====================================================================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
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
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    @staticmethod
    def _extract(values, t, shape):
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise, noise

    def forward_diffusion_sequence(self, x0, num_shown=10):
        device = x0.device
        steps = torch.linspace(0, self.timesteps - 1, num_shown).long().to(device)
        imgs = []
        for t_val in steps:
            t = torch.full((x0.shape[0],), t_val.item(), device=device, dtype=torch.long)
            x_t, _ = self.q_sample(x0, t)
            imgs.append(x_t)
        return imgs, steps.tolist()

    def training_loss(self, model, x0, t, loss_type="l2"):
        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t, noise=noise)
        predicted_noise = model(x_t, t)
        if loss_type == "l1":
            return F.l1_loss(predicted_noise, noise)
        elif loss_type == "l2":
            return F.mse_loss(predicted_noise, noise)
        elif loss_type == "huber":
            return F.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    @torch.no_grad()
    def p_sample(self, model, x_t, t, t_index):
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
        img = torch.randn(shape, device=device)
        imgs = [img.cpu()]
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            img = torch.clamp(img, -1.0, 1.0)
            if return_all_steps:
                imgs.append(img.cpu())
        if return_all_steps:
            return img, imgs
        return img

    @torch.no_grad()
    def sample_from_given_noise(self, model, noise, return_all_steps=False):
        device = noise.device
        img = noise
        imgs = [img.cpu()]
        for i in reversed(range(self.timesteps)):
            t = torch.full((img.shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            img = torch.clamp(img, -1.0, 1.0)
            if return_all_steps:
                imgs.append(img.cpu())
        if return_all_steps:
            return img, imgs
        return img


# =====================================================================
# SECTION 3: DENOISING MODEL — U-Net
# =====================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class SelfAttention(nn.Module):
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
        out = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
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
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256,
                 channel_mults=(1, 2, 4)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        channels = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            self.down_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
            self.downsamples.append(
                Downsample(out_ch) if i < len(channels) - 1 else nn.Identity()
            )
            in_ch = out_ch
        self.bottleneck1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.bottleneck_attn = SelfAttention(in_ch)
        self.bottleneck2 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        cur_ch = in_ch
        for i, out_ch in enumerate(reversed_channels):
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
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
            x = up(x)
        return self.final_block(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# CHANGE 2: EMA (Exponential Moving Average) of model weights
# =====================================================================
# Standard trick from the original DDPM paper (Ho et al., 2020) and
# nearly every diffusion model since. 

class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


# =====================================================================
# SECTION 4: TRAINING LOOP + CLI ENTRY POINT
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--images_per_class", type=int, default=20)
    p.add_argument("--selected_classes", type=str, default=None)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2", "huber"])
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--sample_every", type=int, default=25)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    # CHANGE 2 (cont.): EMA decay rate, off by setting to 0
    p.add_argument("--ema_decay", type=float, default=0.995)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_noise_progression_figure(dataset, diffusion, output_dir, num_shown=10):
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
    print(f"Saved forward-noise figure -> {path}")


def save_loss_curve(epoch_losses, output_dir):
    """
    epoch_losses : list of per-epoch average loss values  (len = num_epochs)
    X-axis -> epoch number (1, 2, 3 ...)
    """
    epochs = list(range(1, len(epoch_losses) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, epoch_losses, linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss over time")
    plt.grid(alpha=0.3)
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve -> {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    model = UNet(base_channels=args.base_channels).to(device)
    print(f"Model parameter count: {count_parameters(model):,}")

    # CHANGE 2 (cont.): create EMA shadow model
    ema = EMA(model, decay=args.ema_decay)

    save_noise_progression_figure(dataset, diffusion, args.output_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # CHANGE 3: cosine LR decay across the full training run, instead
    # of a constant LR. This lets the model take large steps early on
    # and small, fine refinement steps later — which directly addresses
    # the noisy, non-decreasing loss plateau seen in the original run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    epoch_losses = []
    all_losses   = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for x0, _ in dataloader:
            x0 = x0.to(device)
            t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device).long()
            loss = diffusion.training_loss(model, x0, t, loss_type=args.loss_type)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)  # CHANGE 2 (cont.): update EMA weights every step
            all_losses.append(loss.item())
            running_loss += loss.item()
            num_batches += 1

        scheduler.step()  # CHANGE 3 (cont.): decay LR once per epoch

        avg_epoch_loss = running_loss / max(num_batches, 1)
        epoch_losses.append(avg_epoch_loss)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch}/{args.epochs}]  avg_loss={avg_epoch_loss:.5f}  "
              f"lr={current_lr:.2e}  elapsed={elapsed/60:.1f}min")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            # CHANGE 2 (cont.): sample using the EMA model, not the raw model
            ema.ema_model.eval()
            shape = (args.num_samples, 3, args.image_size, args.image_size)
            samples = diffusion.p_sample_loop(ema.ema_model, shape, device)
            samples = denormalize(samples)
            grid = make_grid(samples, nrow=4)
            save_image(grid, os.path.join(samples_dir, f"epoch_{epoch:04d}.png"))
            print(f"  Saved EMA sample grid for epoch {epoch}")

    save_loss_curve(epoch_losses, args.output_dir)

    ckpt_path = os.path.join(args.output_dir, "diffusion_model.pt")
    torch.save({
        # CHANGE 2 (cont.): save EMA weights as the primary checkpoint
        # since they produce sharper samples. Raw weights kept too,
        # in case you want to inspect/compare them.
        "model_state_dict": ema.ema_model.state_dict(),
        "raw_model_state_dict": model.state_dict(),
        "config": {
            "image_size": args.image_size,
            "timesteps": args.timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "base_channels": args.base_channels,
            "classes": dataset.classes,
        },
    }, ckpt_path)
    print(f"Saved model checkpoint (EMA weights) -> {ckpt_path}")

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
