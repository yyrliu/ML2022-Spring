import argparse
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import torch
import torchvision

import config as cfg
from model import Generator
from utils import fix_random_seed, load_config, setup_logger


def inference(
    checkpoint=None, n_generate=1000, n_output=100, view=False, overwrite=False
):
    """
    1. G_path is the path for Generator ckpt
    2. You can use this function to generate final answer
    """

    logger = setup_logger("hw6.gan", use_wandb=False)

    output_dir = Path(cfg.config.workspace_dir, "output")

    try:
        Path(output_dir).mkdir()
    except FileExistsError as e:
        if not overwrite:
            logger.error("Output directory exists! Inference aborted.")
            raise e
        logger.warning("Output will be overwritten...")
        rmtree(output_dir)
        Path(output_dir).mkdir()

    if checkpoint is None:
        checkpoint = sorted(
            Path(cfg.config.ckpt_dir).glob("G_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1]),
            reverse=True,
        )[0]

    logger.info(f"Loading checkpoint from {checkpoint}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator(cfg.config.z_dim).to(device)
    generator.load_state_dict(torch.load(checkpoint))
    generator.eval()
    z_samples = torch.randn(n_generate, cfg.config.z_dim).to(device)
    with torch.no_grad():
        imgs = (generator(z_samples) + 1) / 2.0

    for i, img in enumerate(imgs):
        torchvision.utils.save_image(img, Path(output_dir, f"{i:03}.jpg"))

    if view:
        row, col = n_output // 10, 10
        grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
        plt.figure(figsize=(row, col))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    fix_random_seed(2022)
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument(
        "-n", "--samples", help="Number of samples to generate", type=int, default=1000
    )
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-v", "--view", action="store_true")
    parser.add_argument(
        "--model", help="Path to Generator checkpoint", type=str, default=None
    )
    args = parser.parse_args()

    load_config(args.config)
    inference(
        checkpoint=args.model,
        n_generate=args.samples,
        view=args.view,
        overwrite=args.force,
    )
