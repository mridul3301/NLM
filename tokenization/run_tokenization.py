import os
import argparse
from pathlib import Path
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
from dotenv import load_dotenv



def ensure_directory_exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    print(f"Ensured directory: {directory_path}")

def train_tokenizer(text_files_path, tokenizer_path, vocab_size, special_tokens):
    paths = [str(x) for x in Path(text_files_path).glob("*.txt")]
    ensure_directory_exists(tokenizer_path)
    
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train(paths, vocab_size=vocab_size, special_tokens=special_tokens)

    # Wrap in a PreTrainedTokenizerFast
    model_length = 512
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=model_length)
    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = tokenizer.token_to_id("<s>")
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = tokenizer.token_to_id("</s>")
    tokenizer.unk_token = "<unk>"
    tokenizer.unk_token_id = tokenizer.token_to_id("<unk>")
    tokenizer.cls_token = "<cls>"
    tokenizer.cls_token_id = tokenizer.token_to_id("<cls>")
    tokenizer.sep_token = "<sep>"
    tokenizer.sep_token_id = tokenizer.token_to_id("<sep>")
    tokenizer.mask_token = "<mask>"
    tokenizer.mask_token_id = tokenizer.token_to_id("<mask>")
    fast_tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    load_dotenv()
    HF_API = os.getenv('HF_API')
    parser = argparse.ArgumentParser(description="Train and save a BPE tokenizer.")
    parser.add_argument("--text_data_path", type=str, required=True, help="Path to the text data directory.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to save the tokenizer.")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--special_tokens", type=str, required=True, help="Comma-separated list of special tokens.")
    
    args = parser.parse_args()
    
     # Convert comma-separated special tokens to a list
    special_tokens = args.special_tokens.split(',')

    train_tokenizer(args.text_data_path, args.tokenizer_path, args.vocab_size, special_tokens)
