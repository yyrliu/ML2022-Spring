import argparse
from importlib import import_module
from fairseq import utils
from fairseq.tasks.translation import TranslationConfig, TranslationTask
import sacrebleu
import subprocess
from pathlib import Path
import torch
import tqdm
import numpy as np
 
from dataloader import load_data_iterator
from model import build_model
from train import inference_step, try_load_checkpoint
from utils import setup_logger

import config as cfg

def avg_checkpoints(model_path, logger, num_to_avg=5):

    input_path = Path(model_path).resolve()
    output_path = Path(f"{model_path}/avg_last_{num_to_avg}_checkpoint.pt").resolve()

    if Path(output_path).exists():
        logger.info(f"{output_path} exists, will not overwrite!")
        return

    command = f"\
        python fairseq/scripts/average_checkpoints.py\
        --inputs {input_path}\
        --num-epoch-checkpoints {num_to_avg}\
        --output {output_path}\
        "

    logger.info(command)

    subprocess.run(command, shell=True)

def generate_prediction(model, task, sequence_generator, device, logger, split="test", outfile="./prediction.txt", prediction_only=False):    
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, cfg.config.max_tokens, cfg.config.num_workers).next_epoch_itr(shuffle=False)
    
    idxs = []
    srcs = []
    hyps = []
    refs = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)

            # do inference
            s, h, r = inference_step(sample, model, task, sequence_generator)
            
            srcs.extend(s)
            refs.extend(r)
            hyps.extend(h)
            idxs.extend(list(sample['id']))

    if prediction_only:
        path_base = Path(outfile)
        with open(path_base.with_stem(f"{path_base.stem}-{task.cfg.target_lang}"), "w") as f:
            for hyp in hyps:
                f.write(f"{hyp}\n")

        with open(path_base.with_stem(f"{path_base.stem}-{task.cfg.source_lang}"), "w") as f:
            for src in srcs:
                f.write(f"{src}\n")

    else:
        tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
        showid = np.random.randint(len(hyps))
        logger.info("example source: " + srcs[showid])
        logger.info("example hypothesis: " + hyps[showid])
        logger.info("example reference: " + refs[showid])
        logger.info(bleu.format())
        preds = [x for _, *x in sorted(zip(idxs, srcs, hyps))]
        with open(outfile, "w") as f:
            f.write(bleu.format()+"\n")
            for s, h in preds:
                f.write(f"{s}\t{h}\n")

        return bleu.format()
    
def main(checkpoint, prediction_only=False):

    logger = setup_logger(use_wandb=False)
    logger.info(f'Loading experiment settings from {Path(cfg.config.savedir, "config.py")}')

    if checkpoint == "avg5":
        avg_checkpoints(cfg.config.savedir, logger, num_to_avg=5)
        checkpoint_name = "avg_last_5_checkpoint.pt"
    elif checkpoint == "best":
        checkpoint_name = "checkpoint_best.pt"
    elif checkpoint == "last":
        checkpoint_name = "checkpoint_last.pt"
    else:
        if Path(f"{cfg.config.savedir}/{checkpoint}").exists():
            checkpoint_name = checkpoint
        else:
            raise ValueError(f"checkpoint {checkpoint} not found!")

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg.arch_args, task)
    model = model.to(device=device)
    # checkpoint_last.pt : latest epoch
    # checkpoint_best.pt : highest validation bleu
    # avg_last_5_checkpoint.pt:　the average of last 5 epochs
    try_load_checkpoint(model, logger, name=checkpoint_name)
    sequence_generator = task.build_generator([model], cfg.config)

    if prediction_only:
        return generate_prediction(model, task, sequence_generator, device, logger, split="mono", outfile=f"{cfg.config.savedir}/prediction-only-{checkpoint}.txt", prediction_only=prediction_only)
    else:
        return generate_prediction(model, task, sequence_generator, device, logger, split="test", outfile=f"{cfg.config.savedir}/prediction-{checkpoint}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, default="config.py")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--prediction_only", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve().relative_to(Path.cwd())
    config = import_module(str(config_path).replace(".py", "").replace("/", "."))
    config.config.savedir = str(config_path.parent)

    cfg.config = config.config
    cfg.arch_args = config.arch_args

    if args.prediction_only:
        if args.checkpoint:
            main(args.checkpoint, prediction_only=True)
        else:
            main("avg5", prediction_only=True)

    else:
        if args.checkpoint:
            checkpoints = [args.checkpoint]
        else:
            checkpoints = ["avg5", "best", "last"]
        
        bleus = []
        
        for checkpoint in checkpoints:
            bleus.append(main(checkpoint))

        for policy, bleu in zip(checkpoint, bleus):
            print(f"{policy}: {bleu}")
