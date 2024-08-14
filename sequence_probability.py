import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def calculate_sequence_probability(sequence, model_name='gpt2'):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenize the input sequence
    tokens = tokenizer(sequence, return_tensors='pt')

    # Get input_ids and attention_mask from tokenized input
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits

    # Calculate probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Calculate the probability of the sequence
    sequence_prob = 1.0
    for i in range(len(input_ids[0]) - 1):
        token_id = input_ids[0][i + 1].item()
        token_prob = torch.exp(log_probs[0, i, token_id]).item()
        sequence_prob *= token_prob

    return sequence_prob


# Example usage
sequence1 = "The quick brown fox jumps over the lazy dog"
sequence2 = "The brown quick fox jumps over the lazy dog"
probability = calculate_sequence_probability(sequence1)
probability_reverse = calculate_sequence_probability(sequence2)
print(probability/ probability_reverse)
print(f"Probability of the sequence: {probability}")
