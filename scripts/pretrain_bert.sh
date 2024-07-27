#!/bin/bash

# Set variables for data preparation and training
TRAIN_FILE="./data/csv/train.csv"
TEST_FILE="./data/csv/test.csv"
TOKENIZER_PATH="./tokenizer/tokenizer_30k"
MODEL_OUTPUT_DIR="./pretrained_bert"
MAX_LENGTH=512
TRUNCATE=False
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16
LOGGING_STEPS=1000
SAVE_STEPS=1000


# Run the combined script
echo "Running data preparation and model training..."
python3 bert/pretrain_bert.py \
  --train_file $TRAIN_FILE \
  --test_file $TEST_FILE \
  --tokenizer_path $TOKENIZER_PATH \
  --model_output_dir $MODEL_OUTPUT_DIR \
  --max_length $MAX_LENGTH \
  --truncate $TRUNCATE \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
  --logging_steps $LOGGING_STEPS \
  --save_steps $SAVE_STEPS \
  
