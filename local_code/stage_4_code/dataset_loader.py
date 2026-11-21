

import os
from local_code.base_class.dataset import dataset
from pathlib import Path


class Dataset_Loader(dataset):

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

        self.dataset_source_folder_path = str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "stage_4_data"
            / "text_classification"
        )
    def load(self):

        print("Loading text classification data...")

        trainData = []
        trainLabel = []

        testData = []
        testLabel = []

        if not os.path.isdir(self.dataset_source_folder_path):
            raise FileNotFoundError(
                "Dataset folder not found: "
                + self.dataset_source_folder_path
            )

        # Read training and testing data.
        for split in ["train", "test"]:

            for sentiment in ["pos", "neg"]:

                folder = os.path.join(
                    self.dataset_source_folder_path,
                    split,
                    sentiment
                )

                if not os.path.isdir(folder):
                    raise FileNotFoundError(
                        "Folder not found: " + folder
                    )

                # Positive = 1, Negative = 0
                label = 1 if sentiment == "pos" else 0

                for filename in sorted(os.listdir(folder)):

                    filepath = os.path.join(folder, filename)

                    if not os.path.isfile(filepath):
                        continue

                    with open(
                        filepath,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        text = f.read().strip()

                    if split == "train":
                        trainData.append(text)
                        trainLabel.append(label)

                    else:
                        testData.append(text)
                        testLabel.append(label)

        print("Training samples:", len(trainData))
        print("Testing samples:", len(testData))

        return {
            "trainData": trainData,
            "trainLabel": trainLabel,
            "testData": testData,
            "testLabel": testLabel
        }
