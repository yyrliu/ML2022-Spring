import argparse
import glob
import logging
import os
from pathlib import Path

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb

from dataloader import get_dataset
from loss_fn import get_loss_fn
from model import Discriminator, Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import fix_random_seed, load_config, setup_logger


def discriminator_train_one_step(
    discriminator, generator, r_imgs, opt, loss_fn, device, stats
):
    z = torch.randn(cfg.config.batch_size, cfg.config.z_dim).to(device)
    f_imgs = generator(z)
    discriminator.train()
    r_imgs = r_imgs.to(device)

    r_logit = discriminator(r_imgs)
    f_logit = discriminator(f_imgs)

    loss = loss_fn(r_logit, f_logit)
    discriminator.zero_grad()
    loss.backward()
    opt.step()
    stats["d_loss"] = loss.item()
    return stats


def generator_train_one_step(generator, discriminator, opt, loss_fn, device, stats):
    generator.train()
    z = torch.randn(cfg.config.batch_size, cfg.config.z_dim).to(device)
    f_imgs = generator(z)
    f_logit = discriminator(f_imgs)
    loss = loss_fn(f_logit)
    generator.zero_grad()
    loss.backward()
    opt.step()
    stats["g_loss"] = loss.item()
    return stats


def gen_samples(generator, z_samples):
    generator.eval()
    with torch.no_grad():
        imgs = (generator(z_samples) + 1) / 2.0
    return imgs


def save_checkpoint(generator, discriminator, ckpt_dir, epoch):
    torch.save(generator.state_dict(), os.path.join(ckpt_dir, f"G_{epoch+1:02d}.pt"))
    torch.save(
        discriminator.state_dict(), os.path.join(ckpt_dir, f"D_{epoch+1:02d}.pt")
    )


def train(overwrite=False):
    logger = setup_logger("hw6.gan")
    logger.info(f"Training with config: {Path(cfg.config.workspace_dir)}/config.py")

    Path(cfg.config.ckpt_dir).mkdir(exist_ok=True)
    logger.info(f"Checkpoints will be saved in {cfg.config.ckpt_dir}")

    if glob.glob(f"{cfg.config.ckpt_dir}/*.pt"):
        if not overwrite:
            raise ValueError(f"Existing checkpoints found, training aborted.")
        else:
            logger.warning(
                f"Existing checkpoints found, training will overwrite checkpoints."
            )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    dataset = get_dataset(cfg.config.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=cfg.config.batch_size, shuffle=True, num_workers=8
    )

    generator = Generator(100)
    discriminator = Discriminator(3)

    loss_fn_d, loss_fn_g = get_loss_fn(cfg.config.model_type)

    opt_D = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.config.lr, betas=(0.5, 0.999)
    )
    opt_G = torch.optim.Adam(
        generator.parameters(), lr=cfg.config.lr, betas=(0.5, 0.999)
    )

    
    z_samples = torch.randn(100, cfg.config.z_dim).to(device)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    step = 0
    stats = { "d_loss": None, "g_loss": None}

    for epoch in range(cfg.config.n_epoch):

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)
        for data in progress_bar:
            stats = discriminator_train_one_step(
                discriminator, generator, data, opt_D, loss_fn_d, device, stats
            )
            if step % cfg.config.n_critic == 0:
                stats = generator_train_one_step(
                    generator, discriminator, opt_G, loss_fn_g, device, stats
                )
            progress_bar.set_postfix(loss_d=stats["d_loss"], loss_g=stats["g_loss"])
            if cfg.config.use_wandb and step % cfg.config.log_step == 0:
                wandb.log(stats)
            step += 1

        logger.info(
            f"Epoch {epoch+1:02d} done: D_loss: {stats['d_loss']:.4f}, G_loss: {stats['g_loss']:.4f}"
        )

        img_sameple_path = f"{cfg.config.workspace_dir}/epoch_{epoch+1:02d}.jpg"
        f_imgs_sample = gen_samples(generator, z_samples)
        torchvision.utils.save_image(
            f_imgs_sample,
            img_sameple_path,
            nrow=10,
        )

        if cfg.config.use_wandb:
            image = wandb.Image(img_sameple_path, caption=f"epoch_{epoch+1:02d}")
            wandb.log({"image": image}, step=step)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_checkpoint(generator, discriminator, cfg.config.ckpt_dir, epoch)


def intercept_and_show_img(imgs):
    grid_img = torchvision.utils.make_grid(imgs.cpu(), nrow=8)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    fix_random_seed(2022)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, default="config.py")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_config(args.config)
    train(overwrite=args.force)
