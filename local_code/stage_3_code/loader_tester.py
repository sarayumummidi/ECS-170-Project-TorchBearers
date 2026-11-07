
from pathlib import Path
from torch.utils.data import DataLoader
from dataset_loader import ImageDataset

# Find the folder containing this Python file
folder = Path(__file__).resolve().parent

#Load MNIST
train_set = ImageDataset(
    str(folder / "MNIST"),
    "train",
    "mnist"
)

#ORL train_set = ImageDataset(str(folder / "ORL"), "train", "orl")

#CIFAR train_set = ImageDataset(str(folder / "CIFAR"), "train", "cifar")

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

images, labels = next(iter(train_loader))

print("Image shape:", images.shape)
print("Label shape:", labels.shape)
