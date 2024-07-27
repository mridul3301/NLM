#!/bin/bash

# Define the paths and parameters
CSV_DATA_PATH="./data/csv"
TEXT_DATA_PATH="./data/text"
TOKENIZER_PATH="./tokenizer/bpetokenizer_1k"
VOCAB_SIZE=1000
SPECIAL_TOKENS="<s>,<pad>,</s>,<unk>,<cls>,<sep>,<mask>"


# Run process_csv.py
echo "Generating text files from CSV data..."
python tokenization/process_csv.py --csv_data_path $CSV_DATA_PATH --text_data_path $TEXT_DATA_PATH


# Perform tokenization on the text data
echo "Tokenizing text data..."
python tokenization/run_tokenization.py --text_data_path $TEXT_DATA_PATH --tokenizer_path $TOKENIZER_PATH --vocab_size $VOCAB_SIZE --special_tokens $SPECIAL_TOKENS

echo "Tokenization complete."