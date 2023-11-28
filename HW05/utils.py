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

def load_config(config_path):
    config_path = Path(config_path).resolve().relative_to(Path.cwd())
    config = import_module(str(config_path).replace(".py", "").replace("/", "."))
    config.config.savedir = str(config_path.parent)
    cfg.config = config.config
    cfg.arch_args = config.arch_args
    return config
