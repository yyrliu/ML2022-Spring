import logging
import random
from argparse import Namespace
from importlib import import_module
from pathlib import Path
import glob
import logging


import numpy as np
import torch

import config as cfg

logger = logging.getLogger(__name__)

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
    cfg.arch_args = Namespace(**(cfg.default_arch_args | config.arch_args))
    cfg.config.workspace_dir = str(config_path.parent)
    cfg.config.ckpt_dir = str(Path(cfg.config.workspace_dir, "checkpoints"))

def comfirm_overwrite(patterns, overwrite):
    if not isinstance(patterns, list):
        patterns = [patterns]

    patterns = [str(pattern) if isinstance(pattern, Path) else pattern for pattern in patterns ]
    
    old_records = [ Path(path) for pattern in patterns for path in glob.glob(pattern) ]
    if len(old_records) > 0:
        if not overwrite:
            raise FileExistsError(f"Existing files in {[pattern for pattern in patterns]} found, Will not overwrite.")
        else:
            logger.warning(
                f"Existing files in {[pattern for pattern in patterns]} will be removed."
            )
            old_records.sort(key=lambda x: len(x.parents), reverse=True)
            for path in old_records:
                path.rmdir() if path.is_dir() else path.unlink()
    
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

def transpose_dict_to_list(d):
    return [dict(zip(d, col)) for col in zip(*d.values())]
