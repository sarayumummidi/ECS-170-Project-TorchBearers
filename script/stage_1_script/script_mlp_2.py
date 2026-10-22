
import numpy as np
import torch

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP

if __name__ == "__main__":

    np.random.seed(2)
    torch.manual_seed(2)

    # Load training data
    train_loader = Dataset_Loader("training data", "")
    train_loader.dataset_source_folder_path = "/Users/kavirajesh/ECS-170-Project-TorchBearers/data/stage_2_data/"
    train_loader.dataset_source_file_name = "train.csv"

    # Load testing data
    test_loader = Dataset_Loader("testing data", "")
    #change the path and make sure you add train and test csv files to stage_2_data!
    test_loader.dataset_source_folder_path = "/Users/kavirajesh/ECS-170-Project-TorchBearers/data/stage_2_data/"
    test_loader.dataset_source_file_name = "test.csv"

    train_data = train_loader.load()
    test_data = test_loader.load()

    # Check data dimensions
    print("Input features:", len(train_data["X"][0]))
    print("Training classes:", sorted(set(train_data["y"])))
    print("Testing classes:", sorted(set(test_data["y"])))

    # Initialize model
    method_obj = Method_MLP("multi-layer perceptron", "")

    # Give the model the training and testing data
    method_obj.data = {
        "train": train_data,
        "test": test_data
    }

    # Train and test
    method_obj.run()

    print("************ Finished ************")
