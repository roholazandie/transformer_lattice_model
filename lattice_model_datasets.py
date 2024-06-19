import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import product
import os
from scipy.io import loadmat
from boolean_tokenizer import BooleanTokenizer
import random


class IsingDataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, label2id, window_size=50):
        self.tokenizer = tokenizer
        self.data = []
        for file in os.listdir(dataset_dir):
            if file.endswith('.mat'):
                filepath = os.path.join(dataset_dir, file)
                mat_contents = loadmat(filepath)
                self.data.append({
                    'matrix': mat_contents['spin'],
                    'correlation_length': mat_contents['correlation_length'].item()
                })

        # normalize the correlation length
        correlation_lengths = [d['correlation_length'] for d in self.data]
        correlation_lengths = np.array(correlation_lengths)/np.max(correlation_lengths)
        self.data = [{'matrix': d['matrix'], 'correlation_length': c} for d, c in zip(self.data, correlation_lengths)]

        self.threshold = 0.6
        self.label2id = label2id
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # prepare the input for the model
        # choose a random window of size window_size from the lattice
        i, j = np.random.randint(0, self.data[idx]['matrix'].shape[0] - self.window_size, 2)
        bits = self.data[idx]['matrix'][i:i+self.window_size, j:j+self.window_size].flatten()
        # print(sum(bits))
        correlation_length = self.data[idx]['correlation_length']

        bits_tokens = torch.tensor(self.tokenizer.encode(bits), dtype=torch.long)
        attention_mask = torch.ones_like(bits_tokens)

        label = correlation_length
        # if correlation_length > self.threshold:
        #     label = 1
        # else:
        #     label = 0

        # label = self.label2id[str(label)]
        label = torch.tensor(label).float()
        return {
            'input_ids': bits_tokens,
            'attention_mask': attention_mask,
            'labels': label
        }


class GaussianRandomNumberDataset(Dataset):
    def __init__(self, n_samples, n_features, V, std=1, random_seed=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.V = V
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

        self.X = self.rng.normal(size=(self.n_samples, self.n_features))
        mean = V / 2
        self.X = self.rng.normal(loc=mean, scale=std, size=(self.n_samples, self.n_features))
        self.X = np.clip(self.X, 0, V-1).astype(int)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.X[idx], dtype=torch.int64),
            'attention_mask': torch.ones(self.n_features)  # all ones, since we don't have any padding tokens
        }
    

class GoldenMeanShiftDataset(Dataset):
    def __init__(self, tokenizer, label2id, length=10000, window_size=50):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.data = []
        start_symbol = None
        prev = start_symbol  # Start with no previous symbol

        for _ in range(length):
            if prev == 1:
                next_symbol = 0  # If the previous symbol is 1, the next must be 0
            else:
                next_symbol = random.choice([0, 1])  # Otherwise, choose randomly between 0 and 1

            self.data.append(next_symbol)
            prev = next_symbol  # Update the previous symbol


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # prepare the input for the model
        bits = self.data[idx:idx+self.window_size]
        bits_tokens = torch.tensor(self.tokenizer.encode(bits), dtype=torch.long)
        attention_mask = torch.ones_like(bits_tokens)

        return {
            'input_ids': bits_tokens,
            'attention_mask': attention_mask,
        }