from fairseq import utils
from fairseq.tasks.translation import TranslationConfig, TranslationTask
import pprint

from config import seed
import config as cfg

from utils import setup_logger

## setup task

def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond 
        # first call of this method has no effect. 
    )
    return batch_iterator

def main():
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

    logger = setup_logger()

    logger.info("loading data for epoch 1")
    task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
    task.load_dataset(split="valid", epoch=1)

    sample = task.dataset("valid")[1]
    pprint.pprint(sample)
    pprint.pprint(
        "Source: " + \
        task.source_dictionary.string(
            sample['source'],
            cfg.config.post_process,
        )
    )
    pprint.pprint(
        "Target: " + \
        task.target_dictionary.string(
            sample['target'],
            cfg.config.post_process,
        )
    )


    demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
    demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
    sample = next(demo_iter)
    pprint.pprint(sample)

if __name__ == "__main__":
    main()
