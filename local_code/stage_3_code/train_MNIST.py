from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_loader import ImageDataset
from Method_CNN_MNIST import Method_CNN_MNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "stage_3_data" / "MNIST"
result_path = project_root / "result"

result_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

#load dataset

train_set = ImageDataset(
    str(data_path),
    "train",
    "mnist"
)

test_set = ImageDataset(
    str(data_path),
    "test",
    "mnist"
)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)

print("Training samples:", len(train_set))
print("Testing samples:", len(test_set))

#initialize cnn + training settings

model = Method_CNN_MNIST().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 10

train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

#evaluation function

def evaluate(model, loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy


# train cnn

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Track training results
        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_train_loss = running_loss / total
    epoch_train_accuracy = 100 * correct / total

    # Evaluate on testing dataset
    epoch_test_loss, epoch_test_accuracy = evaluate(
        model, test_loader
    )

    train_losses.append(epoch_train_loss)
    test_losses.append(epoch_test_loss)

    train_accuracies.append(epoch_train_accuracy)
    test_accuracies.append(epoch_test_accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {epoch_train_loss:.4f} "
        f"Train Accuracy: {epoch_train_accuracy:.2f}% "
        f"Test Loss: {epoch_test_loss:.4f} "
        f"Test Accuracy: {epoch_test_accuracy:.2f}%"
    )

#final eval
final_loss, final_accuracy = evaluate(model, test_loader)

print("\n========== Final MNIST Results ==========")

print(f"Test Loss: {final_loss:.4f}")

print(f"Test Accuracy: {final_accuracy:.2f}%")

torch.save(
    model.state_dict(),
    result_path / "MNIST_CNN.pth"
)

plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)

plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, epochs + 1), test_losses, label="Testing Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MNIST Training and Testing Loss")
plt.legend()
plt.grid(True)

# accuracy curve
plt.subplot(1, 2, 2)

plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
plt.plot(range(1, epochs + 1), test_accuracies, label="Testing Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST Training and Testing Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    result_path / "MNIST_Learning_Curve.png"
)

plt.show()

#for 3.5
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nMNIST Test Evaluation")
print("Accuracy:", accuracy_score(all_labels, all_predictions))
print("Precision:", precision_score(all_labels, all_predictions, average="macro"))
print("Recall:", recall_score(all_labels, all_predictions, average="macro"))
print("F1-score:", f1_score(all_labels, all_predictions, average="macro"))