from pathlib import Path
import glob
import sentencepiece as spm
import subprocess
import random
import re

def strQ2B(ustring):
    """Full width -> half width"""
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full width space: direct conversion
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # Full width chars (except space) conversion
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)
                
def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace('-', '') # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s) # keep punctuation
    elif lang == 'zh':
        s = strQ2B(s) # Q2B
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = s.replace('·', '•')
        s = s.replace('‧', '•')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s) # keep punctuation
    s = ' '.join(s.strip().split())
    return s

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())

def match_url(s):
    return re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', s)

def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    # *.raw.* -> *.clean.*
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.raw.{l1}', 'r') as l1_in_f:
        with open(f'{prefix}.raw.{l2}', 'r') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0: # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0: # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0: # remove by ratio of length
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        if match_url(s1) or match_url(s2): # remove sentence with url
                            continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)

def split_dataset(prefix, langs, train_ratio=0.99):
    file_existed = [
        Path(f'{prefix}.train.clean.{lang}').exists()\
        and Path(f'{prefix}.valid.clean.{lang}').exists()
        for lang in langs
    ]
    if all(file_existed):
        print('train/valid splits exists. skipping split.')
        return
    
    line_num = sum(1 for _ in open(f'{data_prefix}.clean.{langs[0]}'))
    labels = list(range(line_num))
    random.shuffle(labels)
    for lang in langs:
        train_f = open(f'{prefix}.train.clean.{lang}', 'w')
        valid_f = open(f'{prefix}.valid.clean.{lang}', 'w')
        count = 0
        for line in open(f'{prefix}.clean.{lang}', 'r'):
            if labels[count]/line_num < train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()

def train_subwords(prefix, langs, vocab_size=8000):
    if (Path(f'{prefix}.spm{vocab_size}.model')).exists():
        print(f'{prefix}.spm{vocab_size}.model exists. skipping spm_train.')
        return
    
    inputs = [ 
        f'{prefix}.{split}.clean.{lang}' for lang in langs for split in ['train', 'valid']
    ]

    print(f'training spm with {inputs}')

    spm.SentencePieceTrainer.train(
        input=','.join(inputs),
        model_prefix=f'{prefix}.spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 'bpe' works as well
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )

def tokenizer(dir, in_tag, langs, model_file):
    spm_model = spm.SentencePieceProcessor(model_file=model_file)
    for split in in_tag.keys():
        for lang in langs:
            out_path = Path(f'{dir}/{split}.{lang}')
            if out_path.exists():
                print(f"{out_path} exists. skipping spm_encode.")
            else:
                with open(f'{dir}/{split}.{lang}', 'w') as out_f:
                    with open(f'{dir}/{in_tag[split]}.{lang}', 'r') as in_f:
                        for line in in_f:
                            line = line.strip()
                            tok = spm_model.encode(line, out_type=str)
                            print(' '.join(tok), file=out_f)

def binarize(source_prefix, bin_path, src_lang, tgt_lang):
    binpath = Path(bin_path)
    if binpath.exists():
        print(binpath, "exists, will not overwrite!")
    else:
        command = f"\
            python -m fairseq_cli.preprocess \
            --source-lang {src_lang}\
            --target-lang {tgt_lang}\
            --trainpref '{source_prefix}/train'\
            --validpref '{source_prefix}/valid'\
            --testpref '{source_prefix}/test'\
            --destdir {bin_path}\
            --joined-dictionary\
            --workers 2\
            "
        print(command)
        subprocess.run(command, shell=True)

def preprocess():
    seed = 73
    random.seed(seed)
    src_lang = 'en'
    tgt_lang = 'zh'
    data_dir = 'data/processed'
    data_prefix = f'{data_dir}/train_dev'
    test_prefix = f'{data_dir}/test'    
    in_tag = {
        'train': 'train_dev.train.clean',
        'valid': 'train_dev.valid.clean',
        'test':  'test.clean',
    }

    # train_dev.raw.en -> train_dev.clean.en
    # train_dev.raw.zh -> train_dev.clean.zh
    clean_corpus(data_prefix, src_lang, tgt_lang)

    # test.raw.en -> test.clean.en
    # test.raw.zh -> test.clean.zh
    clean_corpus(test_prefix, src_lang, tgt_lang, min_len=5, max_len=-1)

    # truncate tests 4000 lines
    subprocess.run(f'echo "$(tail -4000 {test_prefix}.clean.en)" > {test_prefix}.clean.en', shell=True)
    subprocess.run(f'echo "$(tail -4000 {test_prefix}.clean.zh)" > {test_prefix}.clean.zh', shell=True)

    # train_dev.clean.en -> train_dev.train.clean.en + train_dev.valid.clean.en
    # train_dev.clean.zh -> train_dev.train.clean.zh + train_dev.valid.clean.zh
    split_dataset(data_prefix, [src_lang, tgt_lang])

    # trained model saved as train_dev.spm8000.model
    train_subwords(data_prefix, [src_lang, tgt_lang])

    # train_dev.{train,valid,test}.clean.{en,zh} -> train_dev.{train,valid,test}.{en,zh}
    tokenizer(data_dir, in_tag, [src_lang, tgt_lang], f'{data_prefix}.spm8000.model')

    # data/processed/{train,valid,test}.{en,zh} -> data/bin/{train,valid,test}.en-zh.{en,zh}.bin
    binarize(data_dir, 'data/bin', src_lang, tgt_lang)

def back_translate_binarize(source_prefix, bin_path, src_lang, tgt_lang):
    src_dict_file = 'data/bin/dict.en.txt'
    tgt_dict_file = src_dict_file

    bin_files = glob.glob(f'{source_prefix}/mono.zh-en.*.bin')

    if len(bin_files) > 0:
        print(f'{bin_files} exists. skipping back_translation.')
    else:
        command = f"\
            python -m fairseq_cli.preprocess\
            --source-lang {src_lang}\
            --target-lang {tgt_lang}\
            --trainpref '{source_prefix}/back_translation'\
            --destdir {bin_path}\
            --srcdict {src_dict_file}\
            --tgtdict {tgt_dict_file}\
            --workers 2\
            "
        print(command)
        subprocess.run(command, shell=True)

def backtranslate():
    src_lang = 'zh'
    tgt_lang = 'en'
    data_dir = 'data/processed'
    data_prefix = f'{data_dir}/back_translation'
    tokenizer_model = f'data/processed/train_dev.spm8000.model'
    in_tag = {
        'back_translation': 'back_translation.clean',
    }

    num_lines = sum(1 for _ in open(f"{data_prefix}.raw.{src_lang}", 'r'))
    if Path(f"{data_prefix}.raw.{tgt_lang}").exists():
        print(f"{data_prefix}.raw.{tgt_lang} exists. skipping dummy file creation.")
    else:
        with open(f"{data_prefix}.raw.{tgt_lang}", 'w') as tgt_f:
            for _ in range(num_lines):
                tgt_f.write('.\n')

    # back_translation.raw.{en,zh} -> back_translation.{train,valid}.clean.{zh,en}
    clean_corpus(data_prefix, src_lang, tgt_lang, ratio=-1, max_len=1000, min_len=1)
    # back_translation.{train,valid}.clean.{zh,en} -> back_translation.{zh,en}
    tokenizer(data_dir, in_tag, [src_lang, tgt_lang], tokenizer_model)
    # data/processed/back_translation.{zh,en} -> data/bin/train.zh-en.{zh,en}.bin
    back_translate_binarize(data_dir, 'data/bin', src_lang, tgt_lang)
    # data/bin/train.zh-en.{zh,en}.bin -> data/bin/mono.zh-en.{zh,en}.bin
    for bin in glob.glob(f'data/bin/train.zh-en.*'):
        Path(bin).rename(bin.replace('train', 'mono'))

if __name__ == '__main__':
    # preprocess()
    backtranslate()
