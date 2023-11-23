import logging
import sys
from pathlib import Path
import config as cfg

def setup_logger():

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO", # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )

    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)
    if cfg.config.use_wandb:
        import wandb
        wandb.init(project=proj, name=Path(cfg.config.savedir).stem, config=cfg.config)

    return logger
