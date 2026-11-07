#To run this, run loader_tester.py

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, filename, split, dataset_name):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.samples = data[split]
        self.name = dataset_name.lower()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = np.asarray(sample["image"])
        label = int(sample["label"])

        if self.name == "orl":
            image = image[:, :, 0]  #keep only R
            label -= 1             #1-40 -> 0-39

        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        else:
            image = image.transpose(2, 0, 1)

        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float()
        image = image / 255.0  #for 0-255 pixels

        return image, torch.tensor(label).long()
