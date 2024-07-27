# train_bert_model.py

import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from itertools import chain
import os

def load_and_prepare_datasets(data_files, tokenizer_path, max_length, truncate_longer_samples):
    """Load and prepare datasets."""
    # Load dataset
    dataset = load_dataset("csv", data_files=data_files)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({"pad_token" : "<pad>"})
    tokenizer.add_special_tokens({"bos_token" : "<s>"})
    tokenizer.add_special_tokens({"eos_token" : "</s>"})
    tokenizer.add_special_tokens({"unk_token" : "<unk>"})
    tokenizer.add_special_tokens({"cls_token" : "<cls>"})
    tokenizer.add_special_tokens({"sep_token" : "<sep>"})
    tokenizer.add_special_tokens({"mask_token" : "<mask>"})


    # Define encoding functions
    def encode_with_truncation(examples):
        return tokenizer(examples["Article"], truncation=True, padding="max_length",
                         max_length=max_length, return_special_tokens_mask=True)

    def encode_without_truncation(examples):
        return tokenizer(examples["Article"], return_special_tokens_mask=True)

    # Choose the appropriate encode function
    encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

    # Tokenize datasets
    train_dataset = dataset["train"].map(encode, batched=True)
    test_dataset = dataset["test"].map(encode, batched=True)

    if truncate_longer_samples:
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
        test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

    return train_dataset, test_dataset, tokenizer

def group_texts(examples, max_length):
    """Group text data into chunks of max_length."""
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

def train_model(data_files, tokenizer_path, model_output_dir, max_length, truncate_longer_samples, num_train_epochs, 
                per_device_train_batch_size, per_device_eval_batch_size, logging_steps, save_steps):
    """Train the BERT model."""
    # Data preparation
    train_dataset, test_dataset, tokenizer = load_and_prepare_datasets(
        data_files, tokenizer_path, max_length, truncate_longer_samples
    )

    if not truncate_longer_samples:
        train_dataset = train_dataset.map(lambda x: group_texts(x, max_length=max_length), batched=True)
        test_dataset = test_dataset.map(lambda x: group_texts(x, max_length=max_length), batched=True)
        train_dataset.set_format("torch")
        test_dataset.set_format("torch")
    

    
    # Initialize model configuration and model
    model_config = BertConfig(vocab_size=1000, max_position_embeddings=max_length)
    model = BertForMaskedLM(config=model_config)
    
    # Setup data collator for MLM task
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,  # Output directory
        eval_strategy="steps",        # Evaluation strategy
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,              # Number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,   # Training batch size
        gradient_accumulation_steps=8,  # Accumulate gradients before updating weights
        per_device_eval_batch_size=per_device_eval_batch_size,   # Evaluation batch size
        logging_steps=logging_steps,    # Log and save model checkpoints every logging_steps steps
        save_steps=save_steps,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and train BERT model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the testing CSV file")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer directory")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Output directory for the trained model and processed data")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum length of tokenized sequences")
    parser.add_argument("--truncate", type=bool, required=True, help="Whether to truncate longer samples")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, required=True, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True, help="Evaluation batch size per device")
    parser.add_argument("--logging_steps", type=int, required=True, help="Steps between logging metrics")
    parser.add_argument("--save_steps", type=int, required=True, help="Steps between saving model checkpoints")

    args = parser.parse_args()

    data_files = {"train": args.train_file, "test": args.test_file}
    train_model(
        data_files=data_files,
        tokenizer_path=args.tokenizer_path,
        model_output_dir=args.model_output_dir,
        max_length=args.max_length,
        truncate_longer_samples=args.truncate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps
    )
