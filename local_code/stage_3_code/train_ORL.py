from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset_loader import ImageDataset
from Method_CNN_ORL import Method_CNN_ORL
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.transform = transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "stage_3_data" / "ORL"
result_path = project_root / "result"

result_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

#load dataset

train_set = ImageDataset(
    str(data_path),
    "train",
    "orl"
)

test_set = ImageDataset(
    str(data_path),
    "test",
    "orl"
)

# Keep original training data for evaluation
train_eval_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=False
)

# Apply augmentation only during training
augmented_train_set = AugmentedDataset(train_set)

train_loader = DataLoader(
    augmented_train_set,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)

print("Training samples:", len(train_set))
print("Testing samples:", len(test_set))


#initialize cnn + training settings
model = Method_CNN_ORL().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

epochs = 50

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

        # Track training loss
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    # Calculate average training loss
    epoch_train_loss = running_loss / total

    # Evaluate after completing the entire epoch
    _, epoch_train_accuracy = evaluate(
        model, train_eval_loader
    )

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

# final eval
final_loss, final_accuracy = evaluate(model, test_loader)

all_predictions = []
all_labels = []

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())


print("\n========== Final ORL Results ==========")

print(f"Test Loss: {final_loss:.4f}")

print(f"Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")

print(
    f"Precision: {precision_score(all_labels, all_predictions, average='macro', zero_division=0):.4f}"
)

print(
    f"Recall: {recall_score(all_labels, all_predictions, average='macro', zero_division=0):.4f}"
)

print(
    f"F1-score: {f1_score(all_labels, all_predictions, average='macro', zero_division=0):.4f}"
)


# save trained model
torch.save(
    model.state_dict(),
    result_path / "ORL_CNN.pth"
)


plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)

plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, epochs + 1), test_losses, label="Testing Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ORL Training and Testing Loss")
plt.legend()
plt.grid(True)

# accuracy curve
plt.subplot(1, 2, 2)

plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
plt.plot(range(1, epochs + 1), test_accuracies, label="Testing Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ORL Training and Testing Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    result_path / "ORL_Learning_Curve.png"
)

plt.show()
