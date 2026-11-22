import re
import random
import difflib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from local_code.base_class.method import method

# pad = padding, unk = unknown, eos = end of sentence
PAD, UNK, EOS = "<PAD>", "<UNK>", "<EOS>"
IGNORE_INDEX = -100

# text processing
def tokenize(text):
    return re.findall(r"[a-z0-9']+|\.{2,}|[.,!?;:\"]", text.lower())

def detokenize(tokens):
    text = " ".join(tokens)
    return re.sub(r" ([.,!?;:])", r"\1", text)

# loops thru the list of jokes and adds it to the vocab
def build_vocab(jokes):
    counter = Counter()
    for joke in jokes:
        counter.update(tokenize(joke))

    vocab = {PAD: 0, UNK: 1, EOS: 2}

    for word, _ in counter.most_common():
        vocab[word] = len(vocab)

    return vocab

def encode_text(tokens, vocab):
    return [vocab.get(token, vocab[UNK]) for token in tokens]



class JokeDataset(Dataset):
    def __init__(self, jokes, vocab):
        self.samples = []
        for joke in jokes:
            sequence = encode_text(tokenize(joke), vocab) + [vocab[EOS]]
            if len(sequence) >= 2:
                self.samples.append(sequence)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence = self.samples[index]
        return (torch.tensor(sequence[:-1], dtype=torch.long),
                torch.tensor(sequence[1:], dtype=torch.long))


def collate_batch(batch):
    inputs, targets = zip(*batch)

    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=0)

    # -100 ignored by CrossEntropyLoss so padding is not taken into account
    padded_targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=IGNORE_INDEX)

    return padded_inputs, padded_targets


class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 128, hidden_dim = 256, num_layers = 2, cell_type = "rnn", dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        rnn_classes = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }

        self.rnn = rnn_classes[cell_type](
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

        #one score for every word in vocab
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))

        output, hidden = self.rnn(embedded, hidden)

        output = self.fc(self.dropout(output))
        return output, hidden

class Method_RNN_Generation(method):
    def __init__(self, mName=None, mDescription=None, cell_type = "rnn"):
        super().__init__(mName, mDescription)

        self.model = None
        self.vocab = None

        self.cell_type = cell_type
        self.batch_size = 32
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.num_layers = 2
        self.dropout = 0.3
        self.learning_rate = 0.002
        self.max_generation_length = 40

        self.epochs = 50

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def run(self, trainData, startingWords):
        print("Starting RNN text generation ...")
        print("Device: ", self.device)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)


        #build vocab from all the jokes
        self.vocab = build_vocab(trainData)
        id_to_word = {idx: word for word, idx in self.vocab.items()}
        print("Vocab size: ", len(self.vocab))

        # creating training set
        train_dataset = JokeDataset(trainData, self.vocab)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch)

        self.model = RNNGenerator(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            cell_type=self.cell_type).to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses = []
        train_accuracies = []

        #training starts here
        for epoch in range(self.epochs):
            self.model.train()

            total_loss = 0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                outputs, _ = self.model(inputs)

                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()

                # don't count padding, only real words
                mask = targets != IGNORE_INDEX
                num_words = mask.sum().item()

                total_loss += loss.item() * num_words
                predictions = outputs.argmax(dim=-1)

                correct += ((predictions == targets) & mask).sum().item()
                total += num_words

            average_loss = total_loss / total
            accuracy = correct / total

            train_losses.append(average_loss)
            train_accuracies.append(accuracy)

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Loss: {average_loss:.4f} "
                f"Next-word Accuracy: {accuracy:.4f}"
            )


        # generate learning curve
        epochs_range = range(1, self.epochs + 1)
        output_dir = Path(__file__).resolve().parent

        name = self.cell_type.upper()

        # Loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{name} Generation - Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{self.cell_type}_generation_loss.png", dpi=300)
        plt.close()

        # Accuracy curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_accuracies, label="Training Next-word Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{name} Generation - Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{self.cell_type}_generation_accuracy.png", dpi=300)
        plt.close()


        # generate a joke for each set of starting words (3 starting words)

        self.model.eval()

        train_tokens = [tokenize(joke) for joke in trainData]
        results = []

        with torch.no_grad():
            for start in startingWords:
                generated = tokenize(start)

                # read the starting starts to build up the hidden state
                inputs = torch.tensor([encode_text(generated, self.vocab)], device=self.device)
                outputs, hidden = self.model(inputs)

                for _ in range(self.max_generation_length):
                    # most likely the next word
                    next_idx = torch.argmax(outputs[0,-1]).item()

                    if next_idx == self.vocab[EOS]:
                        break

                    generated.append(id_to_word[next_idx])

                    # feed the layers the new word
                    outputs, hidden = self.model(torch.tensor([[next_idx]], device = self.device), hidden)

                # compare with training data
                potential_jokes = [
                    joke for joke in train_tokens
                    if joke[:3] == generated[:3]
                ] or train_tokens

                closest_joke = max(potential_jokes, key=lambda joke: difflib.SequenceMatcher(None, generated, joke).ratio())

                similarity = difflib.SequenceMatcher(None, generated, closest_joke).ratio()

                results.append({
                    "start": start,
                    "generated": detokenize(generated),
                    "closest_training_joke": detokenize(closest_joke),
                    "similarity": similarity,
                    "exact_copy_check": generated == closest_joke
                })

        print("Generation complete.")

        return results









