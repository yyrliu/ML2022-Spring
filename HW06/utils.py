import logging
import random
from argparse import Namespace
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

import config as cfg


def fix_random_seed(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    config_path = Path(config_path).resolve().relative_to(Path.cwd())
    config = import_module(str(config_path).replace(".py", "").replace("/", "."))
    cfg.config = Namespace(**config.config)
    cfg.arch_args = Namespace(**config.arch_args)
    cfg.config.workspace_dir = str(config_path.parent)
    cfg.config.ckpt_dir = str(Path(cfg.config.workspace_dir, "checkpoints"))


def setup_logger(proj, use_wandb=True):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(proj)
    if cfg.config.use_wandb and use_wandb:
        import wandb

        wandb.init(
            project=proj, name=Path(cfg.config.workspace_dir).stem, config=cfg.config
        )
        wandb.config.update(vars(cfg.arch_args))

    return logger
