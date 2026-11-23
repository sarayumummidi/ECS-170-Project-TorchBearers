import zipfile
from pathlib import Path

import re
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from local_code.base_class.method import method

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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



# Load pretrained GloVe word embeddings
def load_glove_embeddings(glove_path, vocab, embedding_dim=100):

    # Initialize embeddings for words not found in GloVe
    rng = np.random.default_rng(42)

    embedding_matrix = rng.normal(
        0, 0.1, (len(vocab), embedding_dim)
    ).astype(np.float32)

    # Padding vector should be zero
    embedding_matrix[vocab["<PAD>"]] = 0

    found = 0

    # Read GloVe vectors directly from ZIP, without extracting the full file.
    with zipfile.ZipFile(glove_path, "r") as archive:
        vector_files = [
            name for name in archive.namelist()
            if name.lower().endswith((".txt", ".vec"))
            and not name.startswith("__MACOSX/")
        ]

        if not vector_files:
            raise ValueError(
                "glove_dataset.zip has no .txt or .vec word-vector file. "
                "Make sure this is a GloVe embeddings ZIP, not the movie review dataset."
            )

        # Prefer a file for the requested embedding dimension if present.
        vector_files.sort(key=lambda name: (
            f"{embedding_dim}d" not in name.lower(),
            -archive.getinfo(name).file_size
        ))
        vector_file = vector_files[0]
        print(f"Reading vectors from ZIP member: {vector_file}", flush=True)

        wrong_dimension_count = 0
        with archive.open(vector_file) as file:
            for raw_line in file:
                line = raw_line.decode("utf-8", errors="replace").strip()
                word, separator, values = line.partition(" ")
                if not separator or word not in vocab:
                    continue

                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                if vector.size != embedding_dim:
                    wrong_dimension_count += 1
                    continue

                embedding_matrix[vocab[word]] = vector
                found += 1

    if found == 0:
        raise ValueError(
            f"No {embedding_dim}-dimensional vectors matched your vocabulary "
            f"(wrong-dimension matches: {wrong_dimension_count}). "
            "Check that glove_dataset.zip contains pretrained GloVe "
            f"{embedding_dim}d vectors."
        )

    print(f"GloVe words matched: {found}/{len(vocab)}")
    return torch.tensor(embedding_matrix)


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

    def run(self, trainData, trainLabel, testData, testLabel=None):

        print("Starting RNN text classification...")
        print("Device:", self.device)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Split labeled data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            list(trainData),
            list(trainLabel),
            test_size=0.2,
            random_state=42,
            stratify=list(trainLabel)
        )

        # Build vocabulary using only training data
        self.vocab = build_vocabulary(train_texts)

        print("Vocabulary size:", len(self.vocab))

        # Create PyTorch training dataset
        train_dataset = TextDataset(
            train_texts, train_labels,
            self.vocab, self.max_length
        )

        val_dataset = TextDataset(
            val_texts, val_labels,
            self.vocab, self.max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )

        # Initialize RNN.
        self.model = RNNClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=100,
            hidden_dim=128,
            num_classes=2
        ).to(self.device)

        # Locate glove_dataset.zip. The first location is recommended.
        project_root = Path(__file__).resolve().parents[2]
        possible_glove_paths = [
            project_root / "data" / "stage_4_data" / "glove_dataset.zip",
            project_root / "data" / "glove_dataset.zip",
            Path(__file__).resolve().parent / "glove_dataset.zip",
            project_root / "glove_dataset.zip",
        ]
        glove_path = next(
            (path for path in possible_glove_paths if path.is_file()),
            None
        )
        if glove_path is None:
            raise FileNotFoundError(
                "Could not find glove_dataset.zip. Put it in "
                f"{possible_glove_paths[0]} or update possible_glove_paths."
            )

        print(f"Loading pretrained GloVe from: {glove_path}", flush=True)
        glove_weights = load_glove_embeddings(
            glove_path,
            self.vocab,
            embedding_dim=100
        )

        # Initialize the existing embedding layer with pretrained vectors.
        # Embeddings remain trainable (the default for nn.Embedding).
        with torch.no_grad():
            self.model.embedding.weight.copy_(
                glove_weights.to(self.device)
            )
        print("GloVe embeddings initialized successfully!", flush=True)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

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

                total_loss += loss.item() * labels.size(0)

                predictions = torch.argmax(outputs, dim=1)

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

            average_loss = total_loss / total
            accuracy = correct / total

            # Save training metrics
            train_losses.append(average_loss)
            train_accuracies.append(accuracy)

            # Evaluate on validation dataset
            self.model.eval()

            val_total_loss = 0
            val_correct = 0
            val_total = 0
            val_true = []
            val_pred = []

            with torch.no_grad():
                for sequences, lengths, labels in val_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(sequences, lengths)
                    loss = criterion(outputs, labels)

                    predictions = torch.argmax(outputs, dim=1)

                    val_total_loss += loss.item() * labels.size(0)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

                    val_true.extend(labels.cpu().tolist())
                    val_pred.extend(predictions.cpu().tolist())

            val_loss = val_total_loss / val_total
            val_accuracy = val_correct / val_total

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Loss: {average_loss:.4f} "
                f"Accuracy: {accuracy:.4f}"
            )

        # Generate learning curves
        epochs_range = range(1, self.epochs + 1)

        output_dir = Path(__file__).resolve().parent

        # Loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_losses, label="Training Loss")
        plt.plot(epochs_range, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("RNN with GloVe - Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "rnn_glove_loss.png", dpi=300)
        plt.close()

        # Accuracy curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
        plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("RNN with GloVe - Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "rnn_glove_accuracy.png", dpi=300)
        plt.close()

        print("Learning curves saved!")

        # Evaluation on validation set after final epoch
        print("\nFinal Validation Evaluation:")
        print(classification_report(
            val_true,
            val_pred,
            digits=4,
            zero_division=0
        ))


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

        if testLabel is not None:
            print("\nTest Evaluation Results:")
            print(classification_report(
                testLabel,
                predictions,
                digits=4,
                zero_division=0
            ))
        else:
            print("Test labels unavailable.")
            print("Use validation metrics for evaluation.")

        return predictions

