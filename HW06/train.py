import argparse
from numbers import Number
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
import wandb
from dataloader import get_dataset
from inference import inference_during_train
from opt import get_opt
from loss_fn import get_loss_fn
from model import Discriminator, Generator
from utils import fix_random_seed, load_config, setup_logger, transpose_dict_to_list, comfirm_overwrite


def discriminator_train_one_step(
    discriminator, generator, r_imgs, opt, loss_fn, device, stats
):
    z = torch.randn(cfg.config.batch_size, cfg.config.z_dim).to(device)
    with torch.no_grad():
        f_imgs = generator(z)
    discriminator.train()
    r_imgs = r_imgs.to(device)
    r_logit = discriminator(r_imgs)
    f_logit = discriminator(f_imgs)
    loss, (r_acc, f_acc, score) = loss_fn(r_logit, f_logit)
    discriminator.zero_grad()
    loss.backward()
    opt.step()

    if cfg.config.model_type == "WGAN":
        with torch.no_grad():
            for p in discriminator.parameters():
                p.clamp_(-cfg.config.weight_clip, cfg.config.weight_clip)
            
    stats["dis/loss"] = loss.item()
    stats["dis/r_acc"] = r_acc
    stats["dis/f_acc"] = f_acc
    stats["dis/score"] = score
    stats["dis/r_dist"] = r_logit.detach().cpu().numpy()
    stats["dis/f_dist"] = f_logit.detach().cpu().numpy()
    return stats


def generator_train_one_step(generator, discriminator, opt, loss_fn, device, stats):
    generator.train()
    z = torch.randn(cfg.config.batch_size, cfg.config.z_dim).to(device)
    f_imgs = generator(z)
    f_logit = discriminator(f_imgs)
    loss, score = loss_fn(f_logit)
    generator.zero_grad()
    loss.backward()
    opt.step()
    stats["gen/loss"] = loss.item()
    stats["gen/score"] = score
    stats["gen/f_dist"] = f_logit.detach().cpu().numpy()
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


def train(overwrite=False, inference=False):
    logger = setup_logger("hw6.gan")
    logger.info(f"Training with config: {Path(cfg.config.workspace_dir)}/config.py")

    comfirm_overwrite([ f"{cfg.config.ckpt_dir}/*.pt", f"{cfg.config.workspace_dir}/epoch_*.jpg" ], overwrite)

    Path(cfg.config.ckpt_dir).mkdir(exist_ok=True)
    logger.info(f"Checkpoints will be saved in {cfg.config.ckpt_dir}")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    dataset = get_dataset(cfg.config.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=cfg.config.batch_size, shuffle=True, num_workers=8
    )

    generator = Generator(cfg.config.z_dim, bias=cfg.arch_config.d_conv_bias, norm=cfg.arch_config.d_conv_norm)
    discriminator = Discriminator()

    loss_fn_d, loss_fn_g = get_loss_fn(cfg.config.model_type)

    opt_D, opt_G = get_opt(cfg.config.model_type, discriminator.parameters(), generator.parameters(), cfg.config.lr)

    z_samples = torch.randn(100, cfg.config.z_dim).to(device)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    step = 0
    stats = {
        "dis/loss": None,
        "dis/r_acc": None,
        "dis/f_acc": None,
        "gen/loss": None,
        "gen/score": None,
    }

    inference_stats = {
        "gen/fid": [],
        "gen/kid": [],
        "gen/afd": [],
    }

    for epoch in range(cfg.config.n_epoch):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}", leave=True) as pbar:
            data_iter = iter(dataloader)
            while True:
                stop = False
                for _ in range(cfg.config.n_critic):
                    try:
                        stats = discriminator_train_one_step(
                            discriminator, generator, next(data_iter), opt_D, loss_fn_d, device, stats
                        )
                    except StopIteration:
                        stop = True
                
                stats = generator_train_one_step(
                    generator, discriminator, opt_G, loss_fn_g, device, stats
                )
                    
                pbar.set_postfix(
                    d_loss=stats["dis/loss"],
                    g_loss=stats["gen/loss"],
                    step=step,
                    refresh=False,
                )

                if cfg.config.use_wandb and step % cfg.config.log_step == 0:
                    hist_stats = { k: wandb.Histogram(v) for k, v in stats.items() if k.endswith("_dist") }
                    scalar_stats = { k: v for k, v in stats.items() if isinstance(v, Number) }
                    wandb.log(hist_stats, step=step, commit=False)
                    wandb.log(scalar_stats, step=step)
                step += cfg.config.n_critic

                if stop:
                    break

                pbar.update(cfg.config.n_critic)


        logger.info(
            f"Epoch {epoch+1:02d} done: D_loss: {stats['dis/loss']:.4f}, G_loss: {stats['gen/loss']:.4f}"
        )

        img_sameple_path = f"{cfg.config.workspace_dir}/epoch_{epoch+1:02d}.jpg"
        f_imgs_sample = gen_samples(generator, z_samples)
        torchvision.utils.save_image(
            f_imgs_sample,
            img_sameple_path,
            nrow=10,
        )

        if inference:
            inference_stats = inference_during_train(generator, step, device, inference_stats)

        if cfg.config.use_wandb:
            if inference:
                last_inference_stats = transpose_dict_to_list(inference_stats)[-1]
                wandb.log(last_inference_stats, step=step, commit=False)
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
    parser.add_argument("config", type=str, default="config.py")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-i", "--inference", action="store_true")
    args = parser.parse_args()

    load_config(args.config)
    train(overwrite=args.force, inference=args.inference)
