import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import random
import numpy as np
from scipy.stats import poisson


# Define a simple tokenizer
class BinaryTokenizer:
    def __init__(self, vocab_size=2, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.model_max_length = 32  # Example max length, adjust as needed

    def encode(self, text):
        return [int(s) for s in text.split(',')]  # Split the string and convert to integers

    def decode(self, token_ids):
        return ','.join([str(token) for token in token_ids])

    def pad(self, encoded_inputs, max_length=None):
        if max_length is None:
            max_length = self.model_max_length
        padded_inputs = {
            "input_ids": [],
            "labels": [],  # Same as input_ids, remove if not used
            "attention_mask": [],
        }
        for input_ids in encoded_inputs["input_ids"]:
            padding_length = max_length - len(input_ids)
            padded_inputs["input_ids"].append(input_ids + [self.pad_token_id] * padding_length)
            padded_inputs["labels"].append(input_ids + [self.pad_token_id] * padding_length)
            padded_inputs["attention_mask"].append([1] * len(input_ids) + [0] * padding_length)
        return padded_inputs


# Data collator for language modeling
class BinaryDataCollator:
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, features):
        batch = {
            "input_ids": [f["input_ids"] for f in features],
            "labels": [f["labels"] for f in features],
        }
        batch = self.tokenizer.pad(batch, max_length=32)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
        }


import random


def generate_sequence(length, lower_bound=1, upper_bound=10):
    if length <= 0:
        return []

    sequence = [random.randint(lower_bound, upper_bound)]  # Start with a random integer between 1 and 10
    for _ in range(1, length):
        next_step = random.choice([1, 2, 3])
        next_value = sequence[-1] + next_step
        sequence.append(next_value)

    return sequence

class SequenceDataset(Dataset):
    '''
    This class generates always increasing sequences of integers (irreversible sequences)
    or partially increasing sequences (partially reversible sequences)
    '''
    def __init__(self, num_samples, seq_length, lower_bound=0, upper_bound=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        all_data = []
        for _ in range(num_samples):
            data = generate_sequence(seq_length, lower_bound, upper_bound)
            if max(data) > 394:
                print(data)
            all_data.append(data)
        self.data = [','.join(map(str, d)) for d in all_data]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = tokenizer.encode(self.data[idx])
        return {"input_ids": input_ids, "labels": input_ids}





if __name__ == '__main__':
    # Instantiate the tokenizer
    tokenizer = BinaryTokenizer()

    seq_length = 128
    lower_bound = 1
    upper_bound = 10
    max_step_size = 3

    vocab_size = seq_length * max_step_size + upper_bound

    # Create dataset
    train_dataset = SequenceDataset(num_samples=1000, seq_length=seq_length, lower_bound=lower_bound, upper_bound=upper_bound)
    eval_dataset = SequenceDataset(num_samples=100, seq_length=seq_length, lower_bound=lower_bound, upper_bound=upper_bound)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Initialize GPT-2 configuration and model from scratch
    config = GPT2Config(vocab_size=vocab_size, n_positions=seq_length, n_embd=128, n_layer=4, n_head=4)
    model = GPT2LMHeadModel(config).to(device)

    data_collator = BinaryDataCollator(tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=10,
        evaluation_strategy="steps",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()



    


