import torch
from torch.nn import functional as F

"""
NOTE

FOR SETTING DISCRIMINATOR LOSS:

GAN: 
    loss_D = (r_loss + f_loss)/2
WGAN: 
    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
WGAN-GP: 
    gradient_penalty = self.gp(r_imgs, f_imgs)
    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty

FOR SETTING GENERATOR LOSS:

GAN: 
    loss_G = self.loss(f_logit, r_label)
WGAN:
    loss_G = -torch.mean(self.D(f_imgs))
WGAN-GP:
    loss_G = -torch.mean(self.D(f_imgs))

"""


def gan_loss_fn_d(real_logit, fake_logit):
    real_loss = F.binary_cross_entropy(
        real_logit, torch.ones_like(real_logit, device=real_logit.device)
    )
    fake_loss = F.binary_cross_entropy(
        fake_logit, torch.zeros_like(fake_logit, device=real_logit.device)
    )
    loss = (real_loss + fake_loss) / 2
    r_acc = (real_logit > 0.5).float().mean().item()
    f_acc = (real_logit < 0.5).float().mean().item()
    score = real_logit.mean() - fake_logit.mean().item()
    return loss, (r_acc, f_acc, score)


def gan_loss_fn_g(fake_logit):
    loss = F.binary_cross_entropy(
        fake_logit, torch.ones_like(fake_logit, device=fake_logit.device)
    )
    score = fake_logit.mean().item()
    return loss, score

def wgan_loss_fn_d(real_logit, fake_logit):
    loss = -torch.mean(real_logit) + torch.mean(fake_logit)
    
    logit_mean = torch.cat([real_logit, fake_logit]).mean().item()
    r_acc = ((real_logit - logit_mean) > 0).float().mean().item()
    f_acc = ((fake_logit - logit_mean) < 0).float().mean().item()
    return loss, (r_acc, f_acc, None)

def wgan_loss_fn_g(fake_logit):
    loss = -torch.mean(fake_logit)
    score = torch.mean(fake_logit).item()
    return loss, score

def get_loss_fn(loss_type):
    if loss_type == "GAN":
        return gan_loss_fn_d, gan_loss_fn_g
    elif loss_type == "WGAN":
        return wgan_loss_fn_d, wgan_loss_fn_g
    else:
        raise NotImplementedError(f"loss_type {loss_type} not implemented")
