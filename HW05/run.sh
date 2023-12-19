#!/bin/bash

DATE=`date '+%Y-%m-%d-%H-%M-%S'`

exec > >(tee "run_${DATE}.log")

# Example for training -> backtranslation -> training with synthetic data
# The backtranslation result path need to be sepicified in preprocessing.py manually:
# in synthetic(): 
#   synthetic_en_path = $BACK_TRANSLATE_CONFIG_DIR/prediction-only-avg5-en.txt"
#   synthetic_zh_path = $BACK_TRANSLATE_CONFIG_DIR/prediction-only-avg5-zh.txt"

# Define config paths
TRANSFORMER_CONFIG="./checkpoints/transformer_base/config.py"
BACK_TRANSLATE_CONFIG="./checkpoints/back_translate_base/config.py"
SYNTHETIC_CONFIG="./checkpoints/synthetic_base_from_bt_base_drop02/config.py"

# train with original data
python main.py --config $TRANSFORMER_CONFIG
python predict.py --config $TRANSFORMER_CONFIG

# train backtranslation model and generate backtranslation result
python main.py --config $BACK_TRANSLATE_CONFIG
python predict.py --config $BACK_TRANSLATE_CONFIG
python predict.py --prediction_only --config $BACK_TRANSLATE_CONFIG

# preprocess synthetic data
python preprocessing.py synthetic

# train with original & synthetic data
python main.py --config $SYNTHETIC_CONFIG
python predict.py --config $SYNTHETIC_CONFIG
