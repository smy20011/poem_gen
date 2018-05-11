#!/bin/bash
USR_DIR=`pwd`
PROBLEM=s1_chat
# PROBLEM=poetry_lines
DATA_DIR=$USR_DIR/data
TMP_DIR=/tmp/t2t_datagen
MODEL=transformer
HPARAMS=transformer_small
TRAIN_DIR=$USR_DIR/data/train/$PROBLEM
SCHEDULE=train

mkdir -p $DATA_DIR $TRAIN_DIR

cp *.txt $DATA_DIR

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams='batch_size=128' \
  --hparams_set=$HPARAMS \
  --schedule=$SCHEDULE \
  --output_dir=$TRAIN_DIR

DECODE_FILE=$DATA_DIR/test_input.txt
# python tokenlize.py < $DATA_DIR/test_input.txt > $DECODE_FILE
BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$DATA_DIR/test_output.txt

cat $DATA_DIR/test_output.txt