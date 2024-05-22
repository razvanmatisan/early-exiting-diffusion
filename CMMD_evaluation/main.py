import argparse
import os
import sys

import distance
import embedding
import io_util
import numpy as np
import torch
from absl import app, flags
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.cifar10 import get_cifar10_dataloader
from models.early_exit import EarlyExitUViT, OldEarlyExitUViT
from models.uvit import UViT
from utils.train_utils import (
    get_noise_scheduler,
    seed_everything,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def get_args():
    parser = argparse.ArgumentParser(description="Sampling parameters")
    # Default parameters from https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py

    parser.add_argument("--start_seed", type=int, default=1, help="Start seed")
    parser.add_argument("--end_seed", type=int, default=9, help="End seed")
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of timesteps"
    )
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--amp_dtype", type=str, default="bf16", help="AMP data type")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )

    parser.add_argument(
        "--sample_height",
        type=int,
        default=32,
        help="Height of the images sampled for logging",
    )
    parser.add_argument(
        "--sample_width",
        type=int,
        default=32,
        help="Width of the images sampled for logging",
    )

    # Checkpointing
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for loading the training state",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="deediff_uvit",
        choices=["uvit", "deediff_uvit"],
        help="Model name",
    )

    parser.add_argument(
        "--exit_threshold",
        type=float,
        default=0.1,
        help="Early exit threshold",
    )

    parser.add_argument(
        "--load_from_folder", action="store_true", help="Load from folder"
    )

    # Benchmarking
    parser.add_argument(
        "--benchmarking",
        action="store_true",
        help="True if we want to benchmark the sampler",
    )

    # CMMD parameters
    parser.add_argument(
        "--cmmd_batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )

    parser.add_argument(
        "--cmmd_max_count",
        type=int,
        default=5,
        help="Maximum number of images to read from each directory.",
    )

    return parser.parse_args()

device = get_device()
betas = torch.linspace(1e-4, 0.02, 1000).to(device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_previous = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
betas_tilde = betas * (1 - alphas_bar_previous) / (1 - alphas_bar)

def sample(threshold, model):
    bs = 8
    torch.manual_seed(0)

    x = torch.randn(bs, 3, 32, 32).to(device)
    error_prediction_by_timestep = torch.zeros(1000, 13)
    indices_by_timestep = torch.zeros(1000, bs)
    for t in tqdm(range(1000, 0, -1)):
        with torch.no_grad():
            time_tensor = t / 1000 * torch.ones(bs, device=device)
            epsilon, classifier_outputs, outputs = model(x, time_tensor)

        outputs = torch.stack(outputs + [epsilon])
        classifier_outputs = torch.stack(
            classifier_outputs + [torch.zeros_like(classifier_outputs[0])]
        )

        # Simulate early exit with a global threshold
        indices = torch.argmax((classifier_outputs <= threshold).int(), dim=0)
        epsilon = outputs[indices, torch.arange(bs)]

        # Log for visualization
        error_prediction_by_timestep[t - 1] = classifier_outputs.mean(axis=1)[:13]
        indices_by_timestep[t - 1, :] = indices

        alpha_t = alphas[t - 1]
        alpha_bar_t = alphas_bar[t - 1]
        sigma_t = torch.sqrt(betas_tilde[t - 1])

        z = torch.randn_like(x) if t > 1 else 0
        x = (
            torch.sqrt(1 / alpha_t)
            * (x - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * epsilon)
        ) + sigma_t * z

    return x

def save_cifar10_images(directory):
    args = get_args()
    # Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)

    dataloader = get_cifar10_dataloader(batch_size=1, seed=0)

    for seed in range(args.start_seed, args.end_seed):
        x, _ = next(iter(dataloader))  # x.shape = [1, 3, 32, 32]
        # File path to save the image, e.g., 'original_data/0.png'
        filename = os.path.join(directory, f"{seed}.png")
        # Save image; 'save_image' expects a batch dimension, so use 'unsqueeze(0)'
        save_image(x, filename)

def save_cifar10_sampled_images(directory):
    args = get_args()
    device = get_device()

    uvit = UViT(
        img_size=32,
        patch_size=2,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    model = OldEarlyExitUViT(uvit=uvit, classifier_type="attention_probe", exit_threshold=args.exit_threshold)

    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path, device)
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
    else:
        print("The loaded checkpoint path is wrong or not provided!")
        exit(1)

    model = model.eval()
    model = model.to(device)

    noise_scheduler = get_noise_scheduler(args)

    os.makedirs(directory, exist_ok=True)

    for seed in range(args.start_seed, args.end_seed):
        x = sample(args.exit_threshold, model)
        x = (x + 1) / 2

        # File path to save the image, e.g., 'generated_data/0.png'
        filename = os.path.join(directory, f"{seed}.png")
        save_image(x, filename)


def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError(
            "`ref_dir` and `ref_embed_file` both cannot be set at the same time."
        )
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(
            ref_dir, embedding_model, batch_size, max_count
        ).astype("float32")
    eval_embs = io_util.compute_embeddings_for_dir(
        eval_dir, embedding_model, batch_size, max_count
    ).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()


if __name__ == "__main__":
    args = get_args()

    # Directory for original images
    output_dir_original = "./Generated_samples/CIFAR10/original_data"

    # Directory for sampled images
    output_dir_model = "./Generated_samples/CIFAR10/generated_data"

    if args.load_from_folder == False:
        save_cifar10_images(output_dir_original)
        print(f"All CIFAR10 images have been saved in '{output_dir_original}'.")

        save_cifar10_sampled_images(output_dir_model)
        print(f"All CIFAR10 images have been saved in '{output_dir_original}'.")

    print(
        "The CMMD value is: "
        f" {compute_cmmd(output_dir_original, output_dir_model, batch_size=args.cmmd_batch_size, max_count=args.cmmd_max_count):.3f}"
    )