# Image Generation Using Diffusion Models
**Assignment 5 — Deep Learning Spring 2026**
**Name: Bilal Bushra**
**Roll Number: MSDS25051**

---

## Project Structure

```
MSDS25051_05/
│
├── MSDS25051_05.py              ← Main training script (run this to train)
├── MSDS25051_05_allCode.py      ← All code concatenated 
├── test_single_sample.ipynb     ← Load model & generate images from noise
├── Report.pdf                   ← Findings, results, loss graphs, noise samples
├── README.md                    ← This file
│
└── saved_models/
    └── diffusion_model.pt       ← Trained model checkpoint
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `MSDS25051_05.py` | CLI training entry point — parses arguments, runs Algorithm 1 training loop |
`set_seed` |
| `test_single_sample.ipynb` | Loads checkpoint from `saved_models/`, generates images from noise |
| `MSDS25051_05_allCode.py` | Concatenation of all `.py` files (required submission item) |

---

## Requirements

```
Python 3.9+
torch
torchvision
matplotlib
Pillow
```

Install with:
```bash
pip install torch torchvision matplotlib pillow
```

---

## Dataset Setup

Unzip animal dataset so the folder looks like:

```
animal_data/
    Bear/
        Bear_1.jpg
        Bear_2.jpg
        ...
    Bird/
        ...
    Cat/
        ...
    ... (15 class folders total)
```

The data loader picks 5 classes and 20 images per class.
Use `--selected_classes` to fix which 5 classes are used.

```

## How to Run — Command Line (Local)

From inside the folder containing all `.py` files:

```bash
python MSDS25051_05.py \
    --data_root /path/to/animal_data \
    --output_dir ./outputs \
    --selected_classes "Bear,Lion,Tiger,Deer,Bird" \
    --image_size 64 \
    --timesteps 1000 \
    --epochs 500 \
    --batch_size 16 \
    --lr 2e-4 \
    --loss_type l2
```

### All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | *(required)* | Path to folder with one sub-folder per animal class |
| `--output_dir` | `./outputs` | Where to save checkpoint, samples, loss curve |
| `--num_classes` | `5` | How many animal classes to train on |
| `--images_per_class` | `20` | Images per class |
| `--selected_classes` | `None` | Comma-separated class names e.g. `"Bear,Lion,Tiger,Deer,Bird"` |
| `--image_size` | `64` | Square resolution to resize images to |
| `--timesteps` | `1000` | Number of diffusion steps T |
| `--epochs` | `500` | Training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `2e-4` | Learning rate |
| `--loss_type` | `l2` | Loss: `l1`, `l2`, or `huber` |
| `--base_channels` | `64` | U-Net base width (~8.6M params at 64) |
| `--sample_every` | `25` | Save generated sample grid every N epochs |
| `--num_samples` | `8` | Number of images in each sample grid |
| `--seed` | `42` | Random seed |

---

## Training Outputs

After training, `--output_dir` will contain:

```
outputs/
    diffusion_model.pt               ← Model checkpoint (copy to saved_models/)
    loss_curve.png                   ← Training loss over all steps
    forward_noise_progression.png    ← Bear image noised across T steps (Figure 1)
    training_log.json                ← All losses + config used
    samples/
        epoch_0025.png               ← Generated sample grid at epoch 25
        epoch_0050.png
        ...
        epoch_0300.png               ← Final generated samples
```

---

## How to Test — Generate Images from Noise


Then **Run all cells**. The notebook will:
1. Load the trained model from the checkpoint
2. Generate a single image starting from pure Gaussian noise
3. Show the full reverse trajectory (noise → image across all T steps)
4. Generate a grid of 8 samples
5. Demonstrate reproducible generation from a fixed seed

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Lower `--batch_size` to 8, or `--base_channels` to 32 |
| `RuntimeError: device mismatch (CPU vs CUDA)` | Already fixed — `img.to(diffusion.device)` is applied before any diffusion call |
| `No class folders found` | Check `--data_root` points to the folder that *contains* class sub-folders, not above it |
| Notebook can't import from `MSDS25051_05.py` | Make sure the file is uploaded to the same Colab session and `sys.path.append(...)` points to its folder |
| Samples look like noise after training | Diffusion models need many epochs on small data — ensure you ran 200+ epochs with `--timesteps 1000` |

---

## Model Summary

The denoising model `eps_theta(x_t, t)` is a **U-Net** (~8.6M parameters at `base_channels=64`):

```
Input: noisy image x_t (3 × 64 × 64) + timestep t
  │
  ├─ Sinusoidal timestep embedding → 256-dim vector
  ├─ Encoder: 64ch → 128ch → 256ch  (ResidualBlock + Downsample)
  ├─ Bottleneck: 256ch + Self-Attention
  ├─ Decoder: 256ch → 128ch → 64ch  (ResidualBlock + Upsample + skip connections)
  └─ Output conv: 64ch → 3ch
Output: predicted noise eps (3 × 64 × 64)
```

**Key design choices:**
- **SiLU activations** — smooth everywhere, better than ReLU for diffusion models
- **GroupNorm** — stable with small batch sizes (16), unlike BatchNorm
- **Skip connections** — preserve spatial detail from encoder to decoder
- **Self-attention at bottleneck only** — captures global structure cheaply at 16×16 resolution
- **Sinusoidal time embeddings** — lets every layer condition on the exact noise level
