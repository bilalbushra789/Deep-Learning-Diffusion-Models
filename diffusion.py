# =====================================================================
# SECTION 2: FORWARD DIFFUSION PROCESS (Algorithm 1 ingredients)
# =====================================================================
#
# IMPORTANT : noise is NEVER applied to the image
# directly in a naive "img + noise" sense. Instead I use the
# closed-form forward process:
#
#     x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
#
# which is mathematically equivalent to injecting Gaussian noise t
# times sequentially (Markov chain q(x_t | x_t-1)), but computed in one
# shot.

import torch
import torch.nn.functional as F


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
        the user can literally hand in noise and get an image back.
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
