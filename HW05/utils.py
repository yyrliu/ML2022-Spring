from argparse import Namespace
import logging
from importlib import import_module
import sys
from pathlib import Path
import config as cfg

def setup_logger(use_wandb=True):

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO", # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )

    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)
    if cfg.config.use_wandb and use_wandb:
        import wandb
        wandb.init(project=proj, name=Path(cfg.config.savedir).stem, config=cfg.config)

    return logger

class Config(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # store keys/attributes that are not defined in original config
        self.lagacy_keys = [
            "lr_decay",
        ]

    def __getattr__(self, item):
        if item in self.lagacy_keys:
            return None
        else:
            raise AttributeError

def load_config(config_path):
    config_path = Path(config_path).resolve().relative_to(Path.cwd())
    config = import_module(str(config_path).replace(".py", "").replace("/", "."))
    config.config.savedir = str(config_path.parent)
    cfg.config = Config(**vars(config.config))
    cfg.arch_args = Config(**vars(config.arch_args))
    return config
