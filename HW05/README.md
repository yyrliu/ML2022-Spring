### Preprocessing of private test dataset

Since the data pair for `test` is not available, [TED Talks 2013](https://object.pouta.csc.fi/OPUS-TED2013/v1.1/moses/en-zh.txt.zip) from [OPUS](https://opus.nlpl.eu/index.php) is used for my personal performance evaluation.

Due to [Dataclass error while importing Fairseq in Python 3.11](https://github.com/facebookresearch/fairseq/issues/5012), a modified version of `Fairseq` is installed with `pip install git+https://github.com/One-sixth/fairseq.git`

Preprocessing for `TED Talks 2013` 
1. Rename `TED2013.en-zh.zh` and `ED2013.en-zh.en` to `test.raw.zh` and `test.raw.en`
2. Use script `HW05/data/zh_cn_to_zh_tw.py` to convert Simplified Chinese to Traditional Chinese (Taiwan Standard) with Taiwanese idiom with [OpenCC](https://github.com/BYVoid/OpenCC)
3. Run `HW05/preprocessing.py` (some modifications were made, see below)
4. Truncate each file and leaving the first 4000 lines with `echo "$(tail -4000 test.clean.en)" > test.clean.en` 

Additional rules added to `HW05/preprocessing.py`
- Remove lines containing URLs (`match_url()`)
- Unify interpuncts symbol "•" `s = s.replace('·', '•')` and `s = s.replace('‧', '•')`
