from boolean_tokenizer import BooleanTokenizer
from lattice_model_datasets import IsingDataset, GaussianRandomNumberDataset, GoldenMeanShiftDataset, \
    IrreducibleRepresentationSymmetryGroupDataset
from utils import load_config
import os
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer, BertConfig, set_seed, BertForSequenceClassification, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer, BertForMaskedLM, DataCollatorForLanguageModeling
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions  # Assuming preds are already the final regression outputs, not logits

    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    return {
        'mse': mse,  # Mean Squared Error
        'mae': mae,  # Mean Absolute Error
        'r2': r2  # R-squared
    }


# Set up the Trainer
set_seed(42)
device = 'cuda:0'

os.environ["WANDB_PROJECT"] = "irreducible_representations"  # log to your project
os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

tokenizer = BooleanTokenizer()

id2label, label2id = {1: '1', 0: '0'}, {'1': 1, '0': 0}

dataset_name = "irreducible_representations"

if dataset_name == "ising_model":
    configs = load_config('configs/ising_config.json')
    train_dataset = IsingDataset('data/ising_matrices_500/train', tokenizer, label2id, configs.window_size)
    eval_dataset = IsingDataset('data/ising_matrices_500/eval', tokenizer, label2id, configs.window_size)

elif dataset_name == "gaussian_random_number":
    V = 20
    configs = load_config('configs/gaussian_config.json')
    train_dataset = GaussianRandomNumberDataset(configs.num_train_examples, n_features=256, V=V, std=5)
    eval_dataset = GaussianRandomNumberDataset(configs.num_eval_examples, n_features=256, V=V, std=5)

elif dataset_name == "golden_mean_shift":
    V = 2
    configs = load_config('configs/golden_mean_shift_config.json')
    train_dataset = GoldenMeanShiftDataset(tokenizer, label2id, configs.window_size)
    eval_dataset = GoldenMeanShiftDataset(tokenizer, label2id, configs.window_size)

elif dataset_name == "irreducible_representations":
    # partitions = [[0, 1, 2], [3, 4], [5]]
    # partitions = [[5,6,7], [8,9], [10]]
    partitions = [[i for i in range(5, 20)], [i for i in range(20, 24)], [i for i in range(24, 30)]]
    V = max(max(partition) for partition in partitions) + 1
    configs = load_config('configs/irreducible_representations_config.json')
    configs.max_length = sum([len(x) for x in partitions])
    train_dataset = IrreducibleRepresentationSymmetryGroupDataset(partitions)
    eval_dataset = IrreducibleRepresentationSymmetryGroupDataset(partitions)

# assert configs.max_length == configs.window_size ** 2, "max_length should be equal to window_size^2"

model_config = BertConfig(  #vocab_size=len(tokenizer.vocab),
    vocab_size=V,  # 10 for Gaussian random number
    # max_position_embeddings=configs.max_length + 2,
    max_position_embeddings=configs.max_length,
    num_hidden_layers=1,
    num_attention_heads=1
)
print(model_config)

# model_config.id2label = id2label
# model_config.label2id = label2id
# model_config.problem_type = "regression"
# model_config.num_labels = 1 # for regression

# model = AutoModelForSequenceClassification.from_config(config=model_config).to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

model = BertForMaskedLM(config=model_config).to(device)

working_dir = f"{configs.output_dir}_{configs.max_length}"

training_args = TrainingArguments(
    output_dir=working_dir,
    overwrite_output_dir=False,
    num_train_epochs=configs.n_epochs,
    per_device_train_batch_size=configs.train_batch_size,
    per_device_eval_batch_size=configs.eval_batch_size,
    learning_rate=configs.learning_rate,
    logging_dir=working_dir,
    dataloader_num_workers=1,
    logging_steps=3,
    save_strategy="steps",  # save a checkpoint every save_steps
    save_steps=int(configs.save_steps * len(train_dataset)),
    save_total_limit=5,
    evaluation_strategy="steps",  # evaluation is done every eval_steps
    eval_steps=int(configs.save_steps * len(train_dataset)),
    load_best_model_at_end=False,
    # metric_for_best_model="f1",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

wandb.finish()
