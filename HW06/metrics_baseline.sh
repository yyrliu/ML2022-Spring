#!/bin/bash

# Get baselines for the metrics with untrained model and real data

LOG_FILE=metrics_baseline.txt

UNTRAINED_OUTPUT=exps/baseline_untrained/output
CRYPTO=data/faces
ANIMEFACE=data/images
ANIMEFACE_2000=data/validate_set

echo "<<< Matrics of untrained model >>>\n" > $LOG_FILE
echo "---------- FID/KID ----------" >> $LOG_FILE
python clean_fid.py -q $UNTRAINED_OUTPUT >> $LOG_FILE
echo "--- Amine Face Dectection ---" >> $LOG_FILE
python yolov5_anime.py -q $UNTRAINED_OUTPUT >> $LOG_FILE

echo "\n<<< Matrics of real pictures >>>" >> $LOG_FILE

echo "\n========== crypko ==========" >> $LOG_FILE
echo "---------- FID/KID ----------" >> $LOG_FILE
python clean_fid.py -q $CRYPTO >> $LOG_FILE
echo "--- Amine Face Dectection ---" >> $LOG_FILE
python yolov5_anime.py -q $CRYPTO >> $LOG_FILE

echo "\n========== animeface ==========" >> $LOG_FILE
echo "---------- FID/KID ----------" >> $LOG_FILE
python clean_fid.py -q $ANIMEFACE >> $LOG_FILE
echo "--- Amine Face Dectection ---" >> $LOG_FILE
python yolov5_anime.py -q $ANIMEFACE >> $LOG_FILE

echo "\n========== animeface, n=2000, cropped to 64x64 ==========" >> $LOG_FILE
echo "---------- FID/KID ----------" >> $LOG_FILE
python clean_fid.py -q $ANIMEFACE_2000 >> $LOG_FILE
echo "--- Amine Face Dectection ---" >> $LOG_FILE
python yolov5_anime.py -q $ANIMEFACE_2000 >> $LOG_FILE
