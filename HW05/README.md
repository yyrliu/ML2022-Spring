## Setup & Preprocessing

### Env config divert from original setup

Due to [Dataclass error while importing Fairseq in Python 3.11](https://github.com/facebookresearch/fairseq/issues/5012), a modified version of `Fairseq` is installed with `pip install git+https://github.com/One-sixth/fairseq.git`

### Preprocessing of private test dataset

Since the data pair for `test` is not available, [TED Talks 2013](https://object.pouta.csc.fi/OPUS-TED2013/v1.1/moses/en-zh.txt.zip) from [OPUS](https://opus.nlpl.eu/index.php) is used for my personal performance evaluation.

Preprocessing of `TED Talks 2013` 

1. Run script `zh_cn_to_zh_tw.py` to convert `data/downloaded/test/TED2013.en-zh.zh` from **Simplified Chinese** to **Traditional Chinese  (Taiwan Standard) with Taiwanese idiom** by [OpenCC](https://github.com/BYVoid/OpenCC). The proceesed file is saved to `data/processed/test.raw.zh`
2. Create copy of the correspomding english dataset `cp data/downloaded/test/TED2013.en-zh.en data/processed/test.raw.en`.
3. Run `preprocessing.py` (some modifications were made, see below)

Additional rules added to `HW05/preprocessing.py`
- Remove lines containing URLs (`match_url()`)
- Unify interpuncts symbol "•" `s = s.replace('·', '•')` and `s = s.replace('‧', '•')`
- Truncate test datasets, leaving only the first 4000 lines since that will be sufficient for model performance evaluation.

### Setup back-translation and train with back-translated (synthetic) data

1. Dataset download from https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted_zh_corpus.deduped.gz

2. Unzip and create a copy of `ted_zh_corpus.deduped` in `data/processed/` and rename the file to `back_translation.raw.zh`

3. Run `python preprocessing.py backtranslate` to clean up and prepare the binary dataset

4. Create a config with `source_lang = "zh"` & `target_lang = "en"` and train the model

5. Run `python predict.py --prediction_only --config {YOUR_CONFIG_FILE}` to generate back-translations.

6. Edit `synthetic_en_path` & `synthetic_zh_path` in `synthetic()` of `preprocessing.py` to point to your back-translations generated in last step.

7. Now you can train with original and synthetic datasets by pointing `datadir` in config to `data/bin/synthetic/`

### Model performance evaluation

`generate_prediction()` in `optimizer` module uses `sacrebleu.corpus_bleu` to evaluate model performance against reference `test.clean.zh` described in [Preprocessing of private test dataset](#preprocessing-of-private-test-dataset)

## Results

| Entry      | BLEU_AVG_Last_5 | BLEU_Best | BLEU_Last |`path`|
|------------|-----------------|-----------|-----------|------|
| [Base Line](#base-line) | 15.16 | 14.59 | 14.34 |`checkpoints/rnn`|
| [LR scheduling](#lr-scheduling) (`lr_factor=2`) | 15.10 | 15.33 | 15.33 |`lr_scheduler` |
| [LR scheduling + train longer](#lr-scheduling--train-longer) (`lr_factor=2`) | 17.27 | 17.06 | 17.06 |`lr_scheduler_30ep` |
| [LR scheduling](#lr-scheduling) (`lr_factor=1`) | 17.15 | 17.55 | 17.55 |`lr_scheduler` |
| [LR scheduling + train longer](#lr-scheduling--train-longer) (`lr_factor=1`) | 18.10 | 17.90 | 17.85 |`lr_scheduler_30ep` |
| [Transformer](#transformer) (best) | 26.61 | 24.33 | 25.80 | Listed in [Transformer](#transformer) |
| [Trained with synthesized data](#training-with-additional-synthesized-training-data) (best) | 27.30 | 26.67 | 26.90 | Listed in [Synthetic](#performance-of-model-trained-with-oringinal--synthesized-data) |

## Experimental details

Training logs are available on [wandb](https://wandb.ai/yyrliu/hw5.seq2seq/overview?workspace=user-yyrliu).

### Base line
Default training of template code

### LR scheduling
- Add scheduler for learing rate
$`lr\_rate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})`$

### LR scheduling + train longer
- $`lr\_rate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})`$

- Train for 30 epoches instead of 15 epochs (default)

### Transformer
- Switch to transformer model
- Train for 30 epoches
- Fine tune transformer architecture

Sign of overfitting has been observed in `tf_base` with increasing `valid_loss` while `train_loss` continued to decrease, therefore higher dropoff ratio is used in `tf_base, drop=0.2` and the issue was resolved. However, during BLEU test, `tf_base` showed best result regardless the overfitting issue, same phenomenon observed during [back-translation](#back-translation-model-evaluation).

| Entry           | encoder_layers | heads | d_encoder | d_encoder_ffn | d_encoder | d_decoder_ffn | dropout | lr_factor | lr_decay | BLEU_AVG_Last_5 | `path` |
|-----------------|----------------|-------|-----------|---------------|-----------|---------------|---------|-----------|----------|-----------------|--------|
| default         | 1 | 4 | 256 | 512 | 256 | 1024 | 0.3 | 1.0 | 0.5 | 15.46 | `transformer` |
| layers=4        | 4 | | | | | | 0.1 | | | 22.18 | `transformer_layers4` |
| ffn=1024        | 4 | | | 1024 | | | 0.1 | | | 22.92 | `transformer_ffn1024` |
| dim=128         | 4 | | 128 | | 128 | 512 | 0.1 | | | 19.63 | `transformer_d128` |
| heads=6         | 4 | 6 | 288 | 1024 | 288 | | 0.1 | | | 23.26 | `transformer_head6` |
| heads=6, lr=1.5 | 4 | 6 | 288 | 1024 | 288 | | 0.1 | 1.5 | | 23.50 | `transformer_head6_lr15` |
| heads=6, lr=1.5, lr_decay=0.54 | 4 | 6 | 288 | 1024 | 288 | | 0.1 | 1.5 | 0.54 | 23.26 | `transformer_head6_lr15_decay-54` |
| tf_base         | 6 | 8 | 512 | 2048 | 512 | 2048 | 0.1 | | | 26.61 | `transformer_base` |
| tf_base, drop=0.2 | 6 | 8 | 512 | 2048 | 512 | 2048 | 0.2 | | | 25.50 | `transformer_base_drop02` |

*Unlisted values are identical to those of the default model.

### Training with Additional Synthesized Training Data

#### Back-translation Model Evaluation

- Train for 30 epoches

Sign of overfitting has been observed in `tf_base` with increasing `valid_loss` while `train_loss` continued to decrease, therefore higher dropoff ratio is used in `tf_base, drop=0.2` and the issue was resolved. However, during BLEU test, `tf_base` showed best result regardless the overfitting issue, therefore both synthetic datasets were applied [below](#performance-of-model-trained-with-oringinal--synthesized-data).

**Note: The `BLEU_AVG_Last_5` is not compareable to other tables since it evaluates the back translation (zh to en) performance**

| Entry            | encoder_layers | heads | d_encoder | d_encoder_ffn | d_encoder | d_decoder_ffn | dropout | lr_factor | lr_decay | BLEU_AVG_Last_5 | `path` |
|------------------|----------------|-------|-----------|---------------|-----------|---------------|---------|-----------|----------|-----------------|--------|
| default          | 4 | 6 | 288 | 1024 | 288 | 1024 | 0.1 | 1.5 | 0.5 | 18.64 | `back_translate` |
| tf_base          | 6 | 8 | 512 | 2048 | 512 | 2048 | | | | 23.79 | `back_translate_base` |
| tf_base, drop=0.2 | 6 | 8 | 512 | 2048 | 512 | 2048 | 0.2 | | | 23.50 | `back_translate_base_drop02` |

*Unlisted values are identical to those of the default model.

**`tf_base` represents the base model architecture in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 

### Performance of Model Trained with Oringinal & Synthesized Data

- Oringinal and synthetic dataset created from back-translation were combined to train the model.
- Train for 30 epoches

| Entry              | encoder_layers | heads | d_encoder | d_encoder_ffn | d_encoder | d_decoder_ffn | dropout | lr_factor | lr_decay | synthetic_data_model | BLEU_AVG_Last_5 | `path` |
|--------------------|----------------|-------|-----------|---------------|-----------|---------------|---------|-----------|----------|----------------------|-----------------|--------|
| default            | 4 | 6 | 288 | 1024 | 288 | 1024 | 0.1 | 1.5 | 0.5 | `back_translate` | 23.79 | `synthetic` |
| lr=0.1             | | | | | | | | 0.1 | | `back_translate` | 23.86 | `synthetic_lr10` |
| ffn=1280           | | | | 1280 | | 1280 | | | | `back_translate` | 24.62 | `synthetic_ffn1280` |
| layers=6           | 6 | | | | | | | | | `back_translate` | 24.37 | `synthetic_layers6` |
| layers=6, ffn=1280 | 6 | | | 1280 | | 1280 | | | | `back_translate` | 24.62 | `synthetic_layers6` |
| tf_base            | 6 | 8 | 512 | 2048 | 512 | 2048 | | | | `back_translate` | 26.56 | `synthetic_base` |
| tf_base, from_back_translate_tf_base | 6 | 8 | 512 | 2048 | 512 | 2048 | | | | `back_translate_base` | 27.30 | `synthetic_base_from_bt_base` |
| tf_base, from_back_translate_tf_base_drop=0.2 | 6 | 8 | 512 | 2048 | 512 | 2048 | | | | `back_translate_base_drop02` | 27.02 | `synthetic_base_from_bt_base_drop02` |

*Unlisted values are identical to those of the default model.

**`tf_base` represents the base model architecture in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 
