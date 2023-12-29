import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ChainDataset
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
import os
import glob
import random
from datetime import datetime
from pathlib import Path

import argparse
import config as cfg
from model import Generator, Discriminator
from dataloader import get_dataset
from utils import load_config, fix_random_seed
from loss_fn import get_loss_fn

def discriminator_train_one_step(discriminator, generator, r_imgs, opt, loss_fn, device):
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

    return loss.item()

def generator_train_one_step(generator, discriminator, opt, loss_fn, device):
    generator.train()
    z = torch.randn(cfg.config.batch_size, cfg.config.z_dim).to(device)
    f_imgs = generator(z)
   
    f_logit = discriminator(f_imgs)

    loss = loss_fn(f_logit)
    generator.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()

def gen_samples(generator, z_samples):
    generator.eval()
    with torch.no_grad():
        imgs = (generator(z_samples) + 1) / 2.0
    return imgs

def save_checkpoint(generator, discriminator, ckpt_dir, epoch):
    torch.save(generator.state_dict(), os.path.join(ckpt_dir, f'G_{epoch+1:02d}.pt'))
    torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, f'D_{epoch+1:02d}.pt'))

def train(overwrite=False):

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M')
    
    Path(cfg.config.ckpt_dir).mkdir(exist_ok=True)
    
    if glob.glob(f'{cfg.config.ckpt_dir}/*.pt'):
        if not overwrite:
            raise ValueError(f"Existing checkpoints found, training aborted.")
        else:
            logging.warning(f"Existing checkpoints found, training will overwrite checkpoints.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(cfg.config.data_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.config.batch_size, shuffle=True, num_workers=8)

    generator = Generator(100)
    discriminator = Discriminator(3)
    
    loss_fn_d, loss_fn_g = get_loss_fn(cfg.config.model_type)

    opt_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.config.lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(generator.parameters(), lr=cfg.config.lr, betas=(0.5, 0.999))

    steps = 0
    z_samples = torch.randn(100, cfg.config.z_dim).to(device)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(cfg.config.n_epoch):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)
        for data in progress_bar:
            loss_d = discriminator_train_one_step(discriminator, generator, data, opt_D, loss_fn_d, device)
            if steps % cfg.config.n_critic == 0:
                loss_g = generator_train_one_step(generator, discriminator, opt_G, loss_fn_g, device)
            steps += 1
            progress_bar.set_postfix(loss_d=loss_d, loss_g=loss_g)

        f_imgs_sample = gen_samples(generator, z_samples)
        torchvision.utils.save_image(f_imgs_sample, f"{cfg.config.workspace_dir}/epoch_{epoch+1:02d}.jpg", nrow=10)

        if (epoch+1) % 5 == 0 or epoch == 0:
            save_checkpoint(generator, discriminator, cfg.config.ckpt_dir, epoch)

def intercept_and_show_img(imgs):
    grid_img = torchvision.utils.make_grid(imgs.cpu(), nrow=8)
    plt.figure(figsize=(10,10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    
if __name__ == '__main__':
    fix_random_seed(2022)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, default="config.py")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_config(args.config)
    train(overwrite=args.force)
