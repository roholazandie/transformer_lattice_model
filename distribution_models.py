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


def gaussian_distribution(mean, std_dev, sequence_length):
    """
    Create a Gaussian distribution for a given sequence length.

    Parameters:
    - mean (float): The mean of the Gaussian distribution.
    - std_dev (float): The standard deviation of the Gaussian distribution.
    - sequence_length (int): The length of the integer sequence (range is [0, sequence_length-1]).

    Returns:
    - distribution (np.array): An array of probabilities representing the Gaussian distribution.
    """

    x = np.arange(sequence_length)
    distribution = np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))
    distribution /= distribution.sum()  # Normalize to make it a valid probability distribution

    return distribution


def poisson_distribution(lam, max_k):
    """
        Compute the probabilities of a Poisson distribution.

        Parameters:
        - lam (float): The lambda (λ) parameter of the Poisson distribution.
        - max_k (int): The maximum value to consider for the distribution.

        Returns:
        - probabilities (np.array): An array of probabilities for the Poisson distribution.
    """

    k = np.arange(max_k)
    probabilities = poisson.pmf(k, lam)

    return probabilities

def power_law_distribution(alpha, sequence_length):
    """
    Create a Power Law distribution for a given sequence length.

    Parameters:
    - alpha (float): The exponent parameter of the Power Law distribution.
    - sequence_length (int): The length of the integer sequence (range is [0, sequence_length-1]).

    Returns:
    - distribution (np.array): An array of probabilities representing the Power Law distribution.
    """

    x = np.arange(1, sequence_length + 1)
    distribution = x ** (-alpha)
    distribution /= distribution.sum()  # Normalize to make it a valid probability distribution

    return distribution


from scipy.special import factorial, gammaln
def calculate_poisson_entropy(lambda_val, max_k=100):
    """
    Calculate the function: λ[1 - log(λ)] + e^(-λ) * sum_{k=0}^{∞} (λ^k log(k!)) / k!

    Parameters:
    - lambda_val (float): The λ parameter.
    - max_k (int): The maximum value of k to consider for the series approximation.

    Returns:
    - result (float): The calculated value of the function.
    """

    # First part of the function
    part1 = lambda_val * (1 - np.log(lambda_val))

    # Second part of the function (series)
    series_sum = 0
    for k in range(max_k + 1):
        term = (lambda_val ** k * gammaln(k + 1)) / factorial(k)
        series_sum += term

    part2 = np.exp(-lambda_val) * series_sum

    # Total result
    result = part1 + part2

    return result

# Define custom dataset
class DistributionDataset(Dataset):
    def __init__(self, distribution, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length
        start_symbol = None
        all_data = []
        for _ in range(num_samples):
            data = np.random.choice(seq_length, seq_length, p=distribution)
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


    # Create a Gaussian distribution
    mean = seq_length // 2
    std_dev = 10
    print(f"Entropy of the Gaussian distribution: {0.5*np.log(2*np.pi*np.e*(std_dev**2))}")
    gaussian_dist = gaussian_distribution(mean, std_dev, seq_length)


    # # Create power law distribution
    # alpha = 1.5
    # power_law_dist = power_law_distribution(alpha, seq_length)


    # # Create Poisson distribution
    # lam = 5 # The λ parameter of the Poisson distribution
    # # print(f"Entropy of the Poisson distribution: {0.5* np.log(2*np.pi*np.e*lam)}")# - 1/(12*lam) - 1/(24*lam**2) - 19/(360*lam**3)}")
    # print(f"Entropy of the Poisson distribution: {calculate_poisson_entropy(lam, 100)}")# - 1/(12*lam) - 1/(24*lam**2) - 19/(360*lam**3)}")
    # poisson_dist = poisson_distribution(lam, seq_length)

    # Create dataset
    train_dataset = DistributionDataset(distribution=gaussian_dist, num_samples=1000, seq_length=seq_length)
    eval_dataset = DistributionDataset(distribution=gaussian_dist, num_samples=200, seq_length=seq_length)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Initialize GPT-2 configuration and model from scratch
    config = GPT2Config(vocab_size=seq_length, n_positions=seq_length, n_embd=128, n_layer=4, n_head=4)
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

    # Inferencing
    input_ids = tokenizer.encode("64, 32")
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    outputs = []
    while True:
        output = model.generate(input_ids, max_length=128, num_return_sequences=1, do_sample=True)
        outputs.append(output)
        if len(outputs) == 50:
            break

    # flatten the list
    outputs = [tokenizer.decode(output[0].tolist()) for output in outputs]
    data1 = [[int(x) for x in output.split(',')] for output in outputs]
    # flatten the list
    data1 = [item for sublist in data1 for item in sublist]
    # plot the histogram
    import matplotlib.pyplot as plt

    data2 = np.random.choice(len(gaussian_dist), len(data1), p=gaussian_dist)

    # Plotting the histograms
    plt.hist(data1, bins=20, edgecolor='black', alpha=0.5, label='Generated')
    plt.hist(data2, bins=20, edgecolor='black', alpha=0.5, label='Gaussion')

    # Adding title and labels
    plt.title('Histogram of Sample Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Display the histogram
    plt.show()
