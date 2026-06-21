# =====================================================================
# SECTION 3: DENOISING MODEL — U-Net (eps_theta(x_t, t))
# =====================================================================
#
# A compact U-Net with sinusoidal timestep embeddings, residual blocks,
# GroupNorm + SiLU activations, and self-attention at the bottleneck.
# This follows the standard DDPM backbone design (Ho et al. 2020) scaled
# down to fit a small dataset (5 classes x 20 images).
#
# Every design choice is commented since the assignment explicitly
# states: "Every step of the model chooses carefully. Which layer,
# activation function is used (when and where).

import math

import torch
import torch.nn as nn


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

    Skip connections (U-Net's defining feature) let fine spatial detail
    from the encoder reach the decoder directly, which is essential for
    reconstructing sharp edges/texture rather than blurry blobs.
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
