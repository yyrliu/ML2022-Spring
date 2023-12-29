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
    return (real_loss + fake_loss) / 2


def gan_loss_fn_g(fake_logit):
    loss = F.binary_cross_entropy(
        fake_logit, torch.ones_like(fake_logit, device=fake_logit.device)
    )
    return loss


def get_loss_fn(loss_type):
    if loss_type == "GAN":
        return gan_loss_fn_d, gan_loss_fn_g
    else:
        raise NotImplementedError(f"loss_type {loss_type} not implemented")
