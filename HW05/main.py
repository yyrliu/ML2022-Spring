from fairseq.tasks.translation import TranslationConfig, TranslationTask
import torch
import random
import numpy as np
import wandb

from dataloader import load_data_iterator
from train import train_one_epoch, validate_and_save, try_load_checkpoint
from optimizer import NoamOpt
from utils import setup_logger
from model import build_model
from config import config, arch_args, seed

from loss_fn import LabelSmoothedCrossEntropyCriterion


if config.use_wandb:
    wandb.config.update(vars(arch_args))

def main():

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    task_cfg = TranslationConfig(
        data=config.datadir,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
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
    logger = setup_logger()

    if config.use_wandb:
        wandb.config.update(vars(arch_args))

    model = build_model(arch_args, task)
    logger.info(model)

    # generally, 0.1 is good enough
    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )

    optimizer = NoamOpt(
        model_size=arch_args.encoder_embed_dim, 
        factor=config.lr_factor, 
        warmup=config.lr_warmup, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

    model = model.to(device=device)
    criterion = criterion.to(device=device)
    sequence_generator = task.build_generator([model], config)

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
    logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")

    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)

    try_load_checkpoint(model, logger, optimizer, name=config.resume)
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, criterion, optimizer, device, logger, config.accum_steps)
        valid_iter = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
        stats = validate_and_save(valid_iter, model, task, criterion, optimizer, sequence_generator, logger, device, epoch=epoch_itr.epoch)
        logger.info("end of epoch {}".format(epoch_itr.epoch))    
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)

if __name__ == "__main__":
    main()
