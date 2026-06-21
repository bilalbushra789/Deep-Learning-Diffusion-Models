import os

import matplotlib.pyplot as plt
import torch


def denormalize(tensor):
    """Map a tensor from [-1, 1] back to [0, 1] for visualization/saving."""
    return (tensor.clamp(-1, 1) + 1) / 2


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_noise_progression_figure(dataset, diffusion, output_dir, num_shown=10):
    """Figure 1 equivalent: show one real image getting progressively noisier."""
    img, _ = dataset[0]
    img = img.unsqueeze(0)
    imgs, steps = diffusion.forward_diffusion_sequence(img, num_shown=num_shown)

    fig, axes = plt.subplots(1, num_shown, figsize=(num_shown * 1.6, 2))
    for ax, x_t, t_val in zip(axes, imgs, steps):
        vis = denormalize(x_t[0]).permute(1, 2, 0).clamp(0, 1).numpy()
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
