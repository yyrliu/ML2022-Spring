import torch

"""
NOTE FOR SETTING OPTIMIZER:

GAN: use Adam optimizer
WGAN: use RMSprop optimizer
WGAN-GP: use Adam optimizer 
"""


def get_opt(opt_type, d_params, g_params, lr):
    if opt_type == "GAN":
        return (torch.optim.Adam(d_params, lr=lr, betas=(0.5, 0.999)),
                torch.optim.Adam(g_params, lr=lr, betas=(0.5, 0.999)))
    # elif opt_type == "WGAN":
    #     return (torch.optim.RMSprop(d_params, lr=lr), 
    #             torch.optim.Adam(g_params, lr=lr, betas=(0.5, 0.999)))
    elif opt_type == "WGAN":
        return (torch.optim.RMSprop(d_params, lr=lr), 
                torch.optim.RMSprop(g_params, lr=lr))
    else:
        raise NotImplementedError(f"opt_type {opt_type} not implemented")
