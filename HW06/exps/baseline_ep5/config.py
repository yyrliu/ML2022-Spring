config = {
    "model_type": "GAN",
    "batch_size": 64,
    "lr": 1e-4,
    "n_epoch": 5,
    "n_critic": 1,
    "z_dim": 100,
    "workspace_dir": None,
    "data_dir": "./data/faces",
    "use_wandb": True,
    "log_step": 50,
    "valid_fid_set": "animeface_2000",
    "valid_afd_thres": 0.895,
}

arch_args = {
    "d_last_activation": "sigmoid",
}
