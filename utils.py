import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class BooleanFunctionTrainConfig:
    model_name: str
    n_epochs: int
    train_batch_size: int
    eval_batch_size: int
    num_train_examples: int
    num_eval_examples: int
    output_dir: str
    save_steps: float
    device: str
    learning_rate: float = 1e-4
    max_length: int = 512
    window_size: int = 50

def load_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return BooleanFunctionTrainConfig(**config_dict)