
from local_code.stage_4_code.dataset_loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN

from sklearn.metrics import accuracy_score


# Step 1: Load the text classification dataset
loader = Dataset_Loader()
data = loader.load()

trainData = data["trainData"]
trainLabel = data["trainLabel"]

testData = data["testData"]
testLabel = data["testLabel"]


# Step 2: Initialize RNN model
# GloVe embeddings are loaded inside Method_RNN.run()
rnn = Method_RNN()


# Step 3: Train RNN with GloVe and classify test data
predictions = rnn.run(
    trainData,
    trainLabel,
    testData,
    testLabel
)


# Step 4: Calculate final testing accuracy
accuracy = accuracy_score(testLabel, predictions)

print("\n===== STAGE 4.3 RESULTS =====")
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

print("RNN with GloVe training and evaluation completed!")
