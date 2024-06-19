import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import random
import numpy as np
import scipy.linalg


# Define a simple tokenizer
class BinaryTokenizer:
    def __init__(self, vocab_size=2, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.model_max_length = 32  # Example max length, adjust as needed

    def encode(self, text):
        return [int(bit) for bit in text]

    def decode(self, token_ids):
        return ''.join([str(token) for token in token_ids])

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


def generate_random_golden_mean_shift_sequence(length, start_symbol=None):
    prev = start_symbol  # Start with no previous symbol
    data = []
    for _ in range(length):
        if prev == 1:
            next_symbol = 0  # If the previous symbol is 1, the next must be 0
        else:
            next_symbol = random.choice([0, 1])  # Otherwise, choose randomly between 0 and 1

        data.append(next_symbol)
        prev = next_symbol  # Update the previous symbol

    return data


def generate_random_sequence(length, start_symbol=None):
    prev = start_symbol  # Start with no previous symbol
    data = []
    for _ in range(length):
        next_symbol = random.choice([0, 1])  # Otherwise, choose randomly between 0 and 1
        data.append(next_symbol)
    return data


def generate_regular_sequence(length, start_symbol=None):
    prev = start_symbol  # Start with no previous symbol
    data = []

    for _ in range(length):
        if prev == 1 or prev == None:
            next_symbol = 0
        else:
            next_symbol = 1
        data.append(next_symbol)
    return data


def generate_thue_morse_sequence(seq_length):
    """Thue–Morse sequence."""
    value = 1
    data = []
    start = random.randint(1, 100000)
    for n in range(start + seq_length):

        # Note: assumes that (-1).bit_length() gives 1
        x = (n ^ (n - 1)).bit_length() + 1
        if x & 1 == 0:
            # Bit index is even, so toggle value
            value = 1 - value
        if n >= start:
            data.append(value)

    return data

def generate_sequence(transition_matrix, sequence_length, initial_state=0):
    """
    Generate a random sequence based on a given transition matrix.

    Parameters:
    - transition_matrix (np.array): A 2D numpy array where transition_matrix[i][j]
                                    is the probability of transitioning from state i to state j.
    - initial_state (int): The starting state.
    - sequence_length (int): The length of the sequence to generate.

    Returns:
    - sequence (list): A list of states representing the generated sequence.
    """
    # convert the transition matrix to a stochastic probabilistic matrix
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    # Number of states
    num_states = transition_matrix.shape[0]

    # Initialize the sequence with the initial state
    sequence = [initial_state]

    # Generate the sequence
    current_state = initial_state
    for _ in range(sequence_length - 1):
        next_state = np.random.choice(num_states, p=transition_matrix[current_state])
        sequence.append(next_state)
        current_state = next_state

    return sequence


# Define custom dataset
class StateMachineDataset(Dataset):
    def __init__(self, transition_matrix, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length
        start_symbol = None
        all_data = []
        for _ in range(num_samples):
            # data = generate_random_golden_mean_shift_sequence(seq_length, start_symbol)
            # data = generate_random_sequence(seq_length, start_symbol)
            # data = generate_regular_sequence(seq_length, start_symbol)
            data = generate_sequence(transition_matrix, seq_length)
            # data = generate_thue_morse_sequence(seq_length)
            all_data.append(data)
        self.data = [''.join(map(str, d)) for d in all_data]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = tokenizer.encode(self.data[idx])
        return {"input_ids": input_ids, "labels": input_ids}


def calculate_entropy(transition_matrix):
    # Ensure the transition matrix is a numpy array
    P = np.array(transition_matrix)
    P = P / P.sum(axis=1, keepdims=True)
    # Calculate the stationary distribution
    eigvals, eigvecs = scipy.linalg.eig(P.T)
    stationary_distribution = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stationary_distribution = stationary_distribution / np.sum(stationary_distribution)
    stationary_distribution = stationary_distribution.flatten()

    # Calculate the entropy for each state
    H = -np.nansum(P * np.log(P), axis=1)

    # Calculate the overall entropy as a weighted sum
    total_entropy = np.sum(stationary_distribution * H)

    return total_entropy

if __name__ == '__main__':
    # Instantiate the tokenizer
    tokenizer = BinaryTokenizer()
    # transition_matrix = np.array([[1, 1], [1, 0]])
    # transition_matrix = np.array([[0.1, 0.9], [0.5, 0.5]])
    transition_matrix = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]])

    # eigvals, eigvecs = np.linalg.eig(transition_matrix)
    # print(f"Entropy is: {np.log(max(eigvals))}")
    print(f"Entropy is: {calculate_entropy(transition_matrix)}")

    seq_length = 128

    # Create dataset
    train_dataset = StateMachineDataset(transition_matrix=transition_matrix, num_samples=1000, seq_length=seq_length)
    eval_dataset = StateMachineDataset(transition_matrix=transition_matrix, num_samples=200, seq_length=seq_length)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Initialize GPT-2 configuration and model from scratch
    config = GPT2Config(vocab_size=len(transition_matrix), n_positions=seq_length, n_embd=128, n_layer=4, n_head=4)
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
