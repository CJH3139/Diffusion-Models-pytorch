import argparse
import os
import torch

from modules import UNet, UNet_conditional
from utils import save_images

CIFAR10_CLASSES = [
    "airplane", "auto", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def _predict_noise(self, model, x, t, labels, cfg_scale):
        if labels is None:
            return model(x, t)
        predicted = model(x, t, labels)
        if cfg_scale > 0:
            uncond = model(x, t, None)
            predicted = torch.lerp(uncond, predicted, cfg_scale)
        return predicted

    @torch.no_grad()
    def sample(self, model, n, labels=None, cfg_scale=3):
        from tqdm import tqdm
        model.eval()
        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)
        for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1):
            t = (torch.ones(n, device=self.device) * i).long()
            predicted_noise = self._predict_noise(model, x, t, labels, cfg_scale)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        return (x * 255).type(torch.uint8)

    @torch.no_grad()
    def sample_ddim(self, model, n, steps=50, labels=None, cfg_scale=3, eta=0.0):
        from tqdm import tqdm
        model.eval()
        # Evenly spaced subset of the 1..noise_steps-1 range, descending.
        timesteps = torch.linspace(self.noise_steps - 1, 1, steps, device=self.device).long()
        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)
        for idx, t_cur in enumerate(tqdm(timesteps)):
            t = torch.full((n,), t_cur.item(), device=self.device, dtype=torch.long)
            predicted_noise = self._predict_noise(model, x, t, labels, cfg_scale)
            a_t = self.alpha_hat[t][:, None, None, None]
            if idx < len(timesteps) - 1:
                t_next = timesteps[idx + 1]
                a_next = self.alpha_hat[t_next][None, None, None, None]
            else:
                a_next = torch.ones_like(a_t)
            x0_pred = (x - torch.sqrt(1 - a_t) * predicted_noise) / torch.sqrt(a_t)
            sigma = eta * torch.sqrt((1 - a_next) / (1 - a_t) * (1 - a_t / a_next))
            dir_xt = torch.sqrt((1 - a_next - sigma ** 2).clamp(min=0)) * predicted_noise
            noise = torch.randn_like(x) if eta > 0 and idx < len(timesteps) - 1 else 0.0
            x = torch.sqrt(a_next) * x0_pred + dir_xt + sigma * noise
        x = (x.clamp(-1, 1) + 1) / 2
        return (x * 255).type(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the .pt checkpoint")
    parser.add_argument("--mode", choices=["uncond", "cond"], default="uncond")
    parser.add_argument("--n", type=int, default=8, help="Number of images to generate")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--class-idx", type=int, default=6,
                        help="CIFAR-10 class index for conditional mode (0-9)")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim",
                        help="ddim is ~20x faster with comparable quality")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of denoising steps for DDIM (ignored for DDPM)")
    parser.add_argument("--out", default=r"C:\Users\roych\dm\outputs\sample.png")
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detected if omitted")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "uncond":
        model = UNet(device=device).to(device)
        labels = None
    else:
        model = UNet_conditional(num_classes=10, device=device).to(device)
        labels = torch.full((args.n,), args.class_idx, dtype=torch.long, device=device)
        print(f"Sampling class {args.class_idx} ({CIFAR10_CLASSES[args.class_idx]})")

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    diffusion = Diffusion(img_size=args.img_size, device=device)
    if args.sampler == "ddim":
        print(f"DDIM sampler, {args.steps} steps")
        imgs = diffusion.sample_ddim(model, args.n, steps=args.steps,
                                     labels=labels, cfg_scale=args.cfg_scale)
    else:
        print("DDPM sampler, 999 steps")
        imgs = diffusion.sample(model, args.n, labels=labels, cfg_scale=args.cfg_scale)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_images(imgs, args.out)
    print(f"Saved {args.n} images -> {args.out}")


if __name__ == "__main__":
    main()
