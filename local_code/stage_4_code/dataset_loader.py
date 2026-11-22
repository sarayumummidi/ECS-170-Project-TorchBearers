import csv
import os
from local_code.base_class.dataset import dataset
from pathlib import Path


class Dataset_Loader(dataset):

    def __init__(self, dName=None, dDescription=None, task=None):
        super().__init__(dName, dDescription)

        if task is None or task not in ["generation", "classification"]:
            raise ValueError("Task must be 'classification' or 'generation'")

        self.task = task

        self.dataset_source_folder_path = str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "stage_4_data"
            / ("text_" + task)
        )

    def load(self):
        if self.task =="classification":
            return self.load_classification()
        return self.load_generation()

    def load_classification(self):

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

    def load_generation(self):
        print("Loading text generation data...")

        file = os.path.join(self.dataset_source_folder_path, "data")

        if not os.path.isfile(file):
            raise FileNotFoundError(
                "Dataset file not found: "
                + file
            )

        trainData = []

        with open(file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                joke = row["Joke"].strip()
                if joke:
                    trainData.append(joke)

        print("Training samples/Number of Jokes:", len(trainData))

        return {
            "trainData": trainData,
        }
