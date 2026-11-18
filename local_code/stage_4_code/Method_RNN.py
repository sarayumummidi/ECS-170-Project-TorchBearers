import re
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from local_code.base_class.method import method


#text processing
def tokenize(text):
        #convert text-> word token

    return re.findall(r"\b\w+\b", text.lower())


def build_vocabulary(texts, max_vocab_size=30000):
    #build vocab only using train data

    word_counts = Counter()

    for text in texts:
        word_counts.update(tokenize(text))

    # 0 = padding, 1 = unknown word
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_length=300):
    tokens = tokenize(text)
    tokens = tokens[:max_length]

    sequence = [
        vocab.get(word, vocab["<UNK>"])
        for word in tokens
    ]
    if len(sequence) == 0:
        sequence = [vocab["<UNK>"]]

    return sequence


class TextDataset(Dataset):

    def __init__(self, texts, labels, vocab, max_length=300):

        self.sequences = [
            encode_text(text, vocab, max_length)
            for text in texts
        ]

        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long)
        )


def collate_batch(batch):

    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack(labels)

    return padded_sequences, lengths, labels


#Rnn model
class RNNClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_classes=2
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            hidden_dim * 2,
            num_classes
        )

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.rnn(packed)

        # Concatenate the forward and backward hidden states.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        combined = self.dropout(combined)

        output = self.fc(combined)

        return output


class Method_RNN(method):

    def __init__(self, mName=None, mDescription=None):

        super().__init__(mName, mDescription)

        self.model = None
        self.vocab = None

        self.max_length = 300
        self.batch_size = 64
        self.epochs = 10
        self.learning_rate = 0.001

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    def run(self, trainData, trainLabel, testData):

        print("Starting RNN text classification...")
        print("Device:", self.device)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Build vocabulary using training data only.
        self.vocab = build_vocabulary(trainData)

        print("Vocabulary size:", len(self.vocab))

        # Create PyTorch training dataset
        train_dataset = TextDataset(
            trainData,
            trainLabel,
            self.vocab,
            self.max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch
        )

        # Initialize RNN.
        self.model = RNNClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=100,
            hidden_dim=128,
            num_classes=2
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # Training
        for epoch in range(self.epochs):

            self.model.train()

            total_loss = 0
            correct = 0
            total = 0

            for sequences, lengths, labels in train_loader:

                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(sequences, lengths)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

            average_loss = total_loss / len(train_loader)
            accuracy = correct / total

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Loss: {average_loss:.4f} "
                f"Accuracy: {accuracy:.4f}"
            )

        # Convert test texts into sequences.
        test_sequences = [
            encode_text(text, self.vocab, self.max_length)
            for text in testData
        ]

        self.model.eval()

        predictions = []

        with torch.no_grad():

            for i in range(0, len(test_sequences), self.batch_size):

                batch_sequences = test_sequences[
                    i:i + self.batch_size
                ]

                tensors = [
                    torch.tensor(seq, dtype=torch.long)
                    for seq in batch_sequences
                ]

                lengths = torch.tensor(
                    [len(seq) for seq in batch_sequences],
                    dtype=torch.long
                )

                padded = pad_sequence(
                    tensors,
                    batch_first=True,
                    padding_value=0
                ).to(self.device)

                outputs = self.model(padded, lengths)

                batch_predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                predictions.extend(
                    batch_predictions.cpu().tolist()
                )

        print("Prediction completed.")

        return predictions
