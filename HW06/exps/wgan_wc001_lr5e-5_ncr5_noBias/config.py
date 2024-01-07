config = {
    "model_type": "WGAN",
    "batch_size": 64,
    "lr": 5e-5,
    "weight_clip": 0.01,
    "n_epoch": 50,
    "n_critic": 5,
    "z_dim": 100,
    "workspace_dir": None,
    "data_dir": "./data/faces",
    "use_wandb": True,
    "log_step": 50,
    "valid_fid_set": "animeface_2000",
    "valid_afd_thres": 0.895,
}

arch_args = {
    "d_conv_bias": False,
}
