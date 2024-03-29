import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForMaskedLM
from error_correction import preprocess_text, correct_errors
from collections import Counter

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

def tokenize_and_mask(text):
    tokens = tokenizer.tokenize(text)
    masked_tokens = tokens.copy()
    for i, token in enumerate(tokens):
        masked_tokens[i] = '[MASK]'
        indexed_tokens = tokenizer.convert_tokens_to_ids(masked_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
    return tokens_tensor

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

def correct_text(input_text):
    tokens_tensor = tokenize_and_mask(input_text)
    with torch.no_grad():
        outputs = model(input_ids=tokens_tensor)
        predictions = torch.argmax(outputs.logits, dim=-1)
    corrected_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(predictions[0]))
    return corrected_text

input_text = "Это предложение с неправельным написанием 1."
corrected_text = correct_text(input_text)
print("Original Text:", input_text)
print("Corrected Text:", corrected_text)


def load_dataset_from_repo(repo_path):
    dataset = []
    for sentiment in ['positive', 'negative']:
        sentiment_dir = os.path.join(repo_path, sentiment)
        files = os.listdir(sentiment_dir)
        for file in files:
            with open(os.path.join(sentiment_dir, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                dataset.extend(lines)
    return dataset

repo_path = 'https://github.com/dkulagin/kartaslov/raw/master/dataset'
dataset = load_dataset_from_repo(repo_path)

train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

word_freq = Counter()
for text in train_dataset:
    tokens = preprocess_text(text)
    word_freq.update(tokens)

def train_model(train_dataset, model, optimizer, loss_fn, num_epochs=10):
    for epoch in range(num_epochs):
        for text in train_dataset:
            correct_text = preprocess_text(text)
            corrected_text = correct_errors(correct_text, word_freq)
            correct_tokens = tokenize_and_mask(correct_text)
            incorrect_tokens = tokenize_and_mask(corrected_text)
            optimizer.zero_grad()
            outputs = model(input_ids=correct_tokens, labels=correct_tokens)
            loss = loss_fn(outputs.logits.view(-1, tokenizer.vocab_size), correct_tokens.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

train_model(train_dataset, model, optimizer, loss_fn)

def correct_text(input_text):
    tokens_tensor = tokenize_and_mask(input_text)
    with torch.no_grad():
        outputs = model(input_ids=tokens_tensor)
        predictions = torch.argmax(outputs.logits, dim=-1)
    corrected_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(predictions[0]))
    return corrected_text

input_text = "Это предложение с неправельным написанием 1."
corrected_text = correct_text(input_text)
print("Original Text:", input_text)
print("Corrected Text:", corrected_text)