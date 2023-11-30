import argparse
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from importlib import import_module
from pathlib import Path
import torch
from torch import nn
import random
import numpy as np
import wandb

from dataloader import load_data_iterator
from train import train_one_epoch, validate_and_save, try_load_checkpoint
from optimizer import NoamOpt
from utils import setup_logger, load_config
from model import build_model
import config as cfg
from config import seed

from loss_fn import LabelSmoothedCrossEntropyCriterion



def main():

    logger = setup_logger()
    logger.info(f'Loading experiment settings from {Path(cfg.config.savedir, "config.py")}')

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    task_cfg = TranslationConfig(
        data=cfg.config.datadir,
        source_lang=cfg.config.source_lang,
        target_lang=cfg.config.target_lang,
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
    task = TranslationTask.setup_task(task_cfg)
    # combine if you have back-translation data.
    task.load_dataset(split="train", epoch=1, combine=True)
    task.load_dataset(split="valid", epoch=1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if cfg.config.use_wandb:
        wandb.config.update(vars(cfg.arch_args))

    model = build_model(cfg.arch_args, task)
    logger.info(model)

    criterion = nn.CrossEntropyLoss(
        reduction="sum",
        label_smoothing=0.1,
        ignore_index=task.target_dictionary.pad()
    )

    optimizer = NoamOpt(
        model_size=cfg.arch_args.encoder_embed_dim, 
        factor=cfg.config.lr_factor, 
        warmup=cfg.config.lr_warmup, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

    model = model.to(device=device)
    criterion = criterion.to(device=device)
    sequence_generator = task.build_generator([model], cfg.config)

    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    logger.info(f"max tokens per batch = {cfg.config.max_tokens}, accumulate steps = {cfg.config.accum_steps}")

    epoch_itr = load_data_iterator(task, "train", cfg.config.start_epoch, cfg.config.max_tokens, cfg.config.num_workers)

    try_load_checkpoint(model, logger, optimizer, name=cfg.config.resume)
    while epoch_itr.next_epoch_idx <= cfg.config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, criterion, optimizer, device, logger, cfg.config.accum_steps)
        valid_iter = load_data_iterator(task, "valid", 1, cfg.config.max_tokens, cfg.config.num_workers).next_epoch_itr(shuffle=False)
        validate_and_save(valid_iter, model, task, criterion, optimizer, sequence_generator, logger, device, epoch=epoch_itr.epoch)
        logger.info("end of epoch {}".format(epoch_itr.epoch))    
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, cfg.config.max_tokens, cfg.config.num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, default="config.py")
    args = parser.parse_args()

    load_config(args.config)
    main()
