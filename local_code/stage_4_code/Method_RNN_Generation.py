import re
import random
import difflib
import zipfile
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


# Text processing
def tokenize(text):
    return re.findall(r"[a-z0-9']+|\.{2,}|[.,!?;:\"]", text.lower())


def detokenize(tokens):
    text = " ".join(tokens)
    return re.sub(r" ([.,!?;:])", r"\1", text)


# Loop through jokes to build the vocabulary
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


# NEW: Load pretrained GloVe 100d vectors directly from glove_dataset.zip
# without unzipping the large vector file.
def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    rng = np.random.default_rng(42)
    embedding_matrix = rng.normal(
        0, 0.1, (len(vocab), embedding_dim)
    ).astype(np.float32)

    # Special padding token should have a zero vector.
    embedding_matrix[vocab[PAD]] = 0
    matched_words = set()

    with zipfile.ZipFile(glove_path, "r") as archive:
        vector_files = [
            name for name in archive.namelist()
            if name.lower().endswith((".txt", ".vec"))
            and not name.startswith("__MACOSX/")
        ]
        if not vector_files:
            raise ValueError(
                "glove_dataset.zip does not contain a .txt or .vec file."
            )

        # Prefer dimension-matching file names; otherwise use the largest.
        vector_files.sort(key=lambda name: (
            f"{embedding_dim}d" not in name.lower(),
            -archive.getinfo(name).file_size,
        ))
        vector_file = vector_files[0]
        print(f"Reading vectors from ZIP member: {vector_file}", flush=True)

        with archive.open(vector_file) as file:
            for raw_line in file:
                line = raw_line.decode("utf-8", errors="replace").strip()
                word, separator, values = line.partition(" ")
                if not separator or word not in vocab:
                    continue

                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                if vector.size == embedding_dim:
                    embedding_matrix[vocab[word]] = vector
                    matched_words.add(word)

    if not matched_words:
        raise ValueError(
            f"No {embedding_dim}-dimensional GloVe vectors matched the joke "
            "vocabulary. Check the ZIP contents and vector dimensions."
        )

    print(f"GloVe words matched: {len(matched_words)}/{len(vocab)}", flush=True)
    return torch.tensor(embedding_matrix)


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
        return (
            torch.tensor(sequence[:-1], dtype=torch.long),
            torch.tensor(sequence[1:], dtype=torch.long),
        )


def collate_batch(batch):
    inputs, targets = zip(*batch)

    padded_inputs = pad_sequence(
        inputs, batch_first=True, padding_value=0
    )

    # -100 is ignored by CrossEntropyLoss so padding is not counted.
    padded_targets = pad_sequence(
        targets, batch_first=True, padding_value=IGNORE_INDEX
    )

    return padded_inputs, padded_targets


class RNNGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,  # CHANGED: GloVe is 100-dimensional
        hidden_dim=256,
        num_layers=2,
        cell_type="rnn",
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
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
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(self.dropout(output))
        return output, hidden


class Method_RNN_Generation(method):
    def __init__(self, mName=None, mDescription=None, cell_type="rnn"):
        super().__init__(mName, mDescription)

        self.model = None
        self.vocab = None
        self.cell_type = cell_type
        self.batch_size = 32
        self.embedding_dim = 100  # CHANGED from 128 to match GloVe 100d
        self.hidden_dim = 256
        self.num_layers = 2
        self.dropout = 0.3
        self.learning_rate = 0.002
        self.max_generation_length = 40
        self.epochs = 50
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self, trainData, startingWords):
        print("Starting RNN text generation ...")
        print("Device:", self.device)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.vocab = build_vocab(trainData)
        id_to_word = {idx: word for word, idx in self.vocab.items()}
        print("Vocab size:", len(self.vocab))

        train_dataset = JokeDataset(trainData, self.vocab)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )

        self.model = RNNGenerator(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            cell_type=self.cell_type,
        ).to(self.device)

        # NEW: Find the same GloVe ZIP used by your text classifier.
        project_root = Path(__file__).resolve().parents[2]
        possible_glove_paths = [
            project_root / "data" / "stage_4_data" / "glove_dataset.zip",
            project_root / "data" / "glove_dataset.zip",
            Path(__file__).resolve().parent / "glove_dataset.zip",
            project_root / "glove_dataset.zip",
        ]
        glove_path = next(
            (path for path in possible_glove_paths if path.is_file()),
            None,
        )
        if glove_path is None:
            raise FileNotFoundError(
                "Could not find glove_dataset.zip. Expected it at "
                f"{possible_glove_paths[0]}"
            )

        print(f"Loading pretrained GloVe from: {glove_path}", flush=True)
        glove_weights = load_glove_embeddings(
            glove_path, self.vocab, embedding_dim=self.embedding_dim
        )

        # NEW: Copy GloVe vectors into the existing embedding layer.
        # We leave embeddings trainable (fine-tuning is enabled).
        with torch.no_grad():
            self.model.embedding.weight.copy_(
                glove_weights.to(self.device)
            )
        print("GloVe embeddings initialized successfully!", flush=True)

        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        train_losses = []
        train_accuracies = []

        # Training starts here.
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
                    targets.reshape(-1),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()

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

        # Generate learning curves.
        epochs_range = range(1, self.epochs + 1)
        output_dir = Path(__file__).resolve().parent
        name = self.cell_type.upper()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{name} Generation with GloVe - Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{self.cell_type}_generation_loss.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_accuracies, label="Training Next-word Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{name} Generation with GloVe - Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            output_dir / f"{self.cell_type}_generation_accuracy.png", dpi=300
        )
        plt.close()

        # Generate a joke for each set of three starting words.
        self.model.eval()
        train_tokens = [tokenize(joke) for joke in trainData]
        results = []

        with torch.no_grad():
            for start in startingWords:
                generated = tokenize(start)
                if not generated:
                    raise ValueError("startingWords must contain nonempty text prompts")

                inputs = torch.tensor(
                    [encode_text(generated, self.vocab)],
                    device=self.device,
                )
                outputs, hidden = self.model(inputs)

                for _ in range(self.max_generation_length):
                    next_idx = torch.argmax(outputs[0, -1]).item()
                    if next_idx == self.vocab[EOS]:
                        break
                    generated.append(id_to_word[next_idx])
                    outputs, hidden = self.model(
                        torch.tensor([[next_idx]], device=self.device), hidden
                    )

                potential_jokes = [
                    joke for joke in train_tokens
                    if joke[:3] == generated[:3]
                ] or train_tokens

                closest_joke = max(
                    potential_jokes,
                    key=lambda joke: difflib.SequenceMatcher(
                        None, generated, joke
                    ).ratio(),
                )
                similarity = difflib.SequenceMatcher(
                    None, generated, closest_joke
                ).ratio()

                results.append({
                    "start": start,
                    "generated": detokenize(generated),
                    "closest_training_joke": detokenize(closest_joke),
                    "similarity": similarity,
                    "exact_copy_check": generated == closest_joke,
                })

        print("Generation complete.")
        return results










