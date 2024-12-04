#!/usr/bin/env python
# coding: utf-8

# ### Data Reading

# #### I implemented my transformer algorithm on Kaggle, adapting the data reading for non-local computation.

# ### Data Reading

import pandas as pd

# Read the data sets
data1 = pd.read_csv("/kaggle/input/dts305/dataset1.csv", encoding="iso-8859-1")
data2 = pd.read_csv("/kaggle/input/dts305/dataset2.csv", encoding="iso-8859-1")
data3 = pd.read_csv("/kaggle/input/dts305/dataset3.csv", encoding="iso-8859-1")

# Number of printed data points
print("Number of data 1 points:", len(data1))
print("Number of data 2 points:", len(data2))
print("Number of data 3 points:", len(data3))

# Data Cleaning
data1 = data1.dropna()
data2 = data2.dropna()
data3 = data3.dropna()

# Explore unique values in the categorical features
print("Total categories in the feature spam:\n", data1["spam"].value_counts(), "\n")
print("Total categories in the feature Emotion:\n", data2["Emotion"].value_counts(), "\n")
print("Total categories in the feature Label:\n", data3["Label"].value_counts(), "\n")

# View dataset column names
print(data1.columns)
print(data2.columns)
print(data3.columns)

# ### Feature Processing

# Rename columns for consistency
data2 = data2.rename(columns={'Comment': 'text'})
data3 = data3.rename(columns={'Text': 'text'})

# Create label mapping for data1
label_map1 = {0: 0, 1: 1}

# Create label mapping for unique emotions in data2
label_map2 = {label: idx for idx, label in enumerate(data2['Emotion'].unique())}

# Create label mapping for unique labels in data3
label_map3 = {label: idx for idx, label in enumerate(data3['Label'].unique())}

# Map spam labels in data1 to new labels
data1['spam'] = data1['spam'].map(label_map1)

# Map emotion labels in data2 to new labels
data2['Emotion'] = data2['Emotion'].map(label_map2)

# Map label values in data3 to new labels
data3['Label'] = data3['Label'].map(label_map3)

# ### CNN Model Setup

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(texts):
    # Tokenize the input text with padding and truncation
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.texts[idx]['input_ids'].squeeze(),
            'attention_mask': self.texts[idx]['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Tokenize and create datasets
def create_dataset(data, label_column):
    texts = data['text'].tolist()
    labels = data[label_column].tolist()
    tokenized_texts = [tokenize_function(text) for text in texts]
    return TextDataset(tokenized_texts, labels)


# Create datasets for each of the three datasets
datasets = {
    'dataset1': create_dataset(data1, 'spam'),
    'dataset2': create_dataset(data2, 'Emotion'),
    'dataset3': create_dataset(data3, 'Label')
}

# Split the datasets into training and validation sets
train_datasets = {}
val_datasets = {}
for name, dataset in datasets.items():
    train_size = int(0.7 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_datasets[name] = train_dataset
    val_datasets[name] = val_dataset


# ### CNN Model Definition

class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNN_Text, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv2d(1, 100, (3, embed_dim))  # Conv layer with kernel size (3, embed_dim)
        self.conv2 = nn.Conv2d(1, 100, (4, embed_dim))  # Conv layer with kernel size (4, embed_dim)
        self.conv3 = nn.Conv2d(1, 100, (5, embed_dim))  # Conv layer with kernel size (5, embed_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, max_length, embed_dim)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, max_length, embed_dim)
        x1 = torch.relu(self.conv1(x)).squeeze(3)  # (batch_size, num_filters, max_length - 3 + 1)
        x2 = torch.relu(self.conv2(x)).squeeze(3)  # (batch_size, num_filters, max_length - 4 + 1)
        x3 = torch.relu(self.conv3(x)).squeeze(3)  # (batch_size, num_filters, max_length - 5 + 1)
        x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# ### Training and Evaluation
# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

        # Validation
        model.eval()
        val_acc = 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_acc += (preds == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {val_acc / len(val_loader.dataset)}")


# Prepare data for DataLoader
def prepare_dataloader(dataset, batch_size=128):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Train and evaluate for each dataset
vocab_size = len(tokenizer)  # Use tokenizer vocab size
embed_dim = 128  # Embedding dimension

for name, train_dataset in train_datasets.items():
    num_classes = len(set([sample['labels'].item() for sample in train_dataset]))
    model = CNN_Text(vocab_size, embed_dim, num_classes).to(device)
    train_loader = prepare_dataloader(train_dataset)
    val_loader = prepare_dataloader(val_datasets[name])

    print(f"Training model for {name}...")
    train_model(model, train_loader, val_loader)

# ### Confusion Matrix and Evaluation

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.show()
