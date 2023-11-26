### Env config divert from original setup

Due to [Dataclass error while importing Fairseq in Python 3.11](https://github.com/facebookresearch/fairseq/issues/5012), a modified version of `Fairseq` is installed with `pip install git+https://github.com/One-sixth/fairseq.git`

### Preprocessing of private test dataset

Since the data pair for `test` is not available, [TED Talks 2013](https://object.pouta.csc.fi/OPUS-TED2013/v1.1/moses/en-zh.txt.zip) from [OPUS](https://opus.nlpl.eu/index.php) is used for my personal performance evaluation.

Preprocessing for `TED Talks 2013` 

1. Run script `zh_cn_to_zh_tw.py` to convert `data/downloaded/test/TED2013.en-zh.zh` from **Simplified Chinese** to **Traditional Chinese  (Taiwan Standard) with Taiwanese idiom** by [OpenCC](https://github.com/BYVoid/OpenCC). The proceesed file is saved to `data/processed/test.raw.zh`
2. Create copy of the correspomding english dataset `cp data/downloaded/test/TED2013.en-zh.en data/processed/test.raw.en`.
3. Run `preprocessing.py` (some modifications were made, see below)

Additional rules added to `HW05/preprocessing.py`
- Remove lines containing URLs (`match_url()`)
- Unify interpuncts symbol "•" `s = s.replace('·', '•')` and `s = s.replace('‧', '•')`
- Truncate test datasets, leaving only the first 4000 lines since that will be sufficient for model performance evaluation.

### Model performance evaluation

`generate_prediction()` in `optimizer` module uses `sacrebleu.corpus_bleu` to evaluate model performance against reference `test.clean.zh` described in [Preprocessing of private test dataset](#preprocessing-of-private-test-dataset)

### Results

| Entry      | BLEU_AVG_Last_5 | BLEU_Best | BLEU_Last |`path`|
|------------|-----------------|-----------|-----------|------|
| [Base Line](#base-line) | 15.16 | 14.59 | 14.34 |`checkpoints/rnn`|
| [LR scheduling](#lr-scheduling) (`lr_factor=2`) | 15.10 | 15.33 | 15.33 |`lr_scheduler` |
| [LR scheduling + train longer](#lr-scheduling--train-longer) (`lr_factor=2`) | 17.27 | 17.06 | 17.06 |`lr_scheduler_30ep` |
| [LR scheduling](#lr-scheduling) (`lr_factor=1`) | 17.15 | 17.55 | 17.55 |`lr_scheduler` |
| [LR scheduling + train longer](#lr-scheduling--train-longer) (`lr_factor=1`) | 18.10 | 17.90 | 17.85 |`lr_scheduler_30ep` |
| [Transformer](#transformer) | 22.18 | 22.15 | 21.73 | Listed in [Transformer](#transformer) |

### Experimental details

#### Base line
Default training of template code

#### LR scheduling
- Add scheduler for learing rate
$lr\_rate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$

#### LR scheduling + train longer
- $lr\_rate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$

- Train for 30 epoches instead of 15 epochs (default)

#### Transformer
- Switch to transformer model

- Fine tune transformer architecture

| Entry           | encoder_layers | dropout | BLEU_AVG_Last_5 | `path` |
|-----------------|----------------|---------| --------------- | ------ |
| Default         | 1 | 0.3 | 15.46 | `transformer` |
| layers=4        | 4 | 0.1 | 22.18 | `transformer_layers4` |


