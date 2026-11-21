
from local_code.stage_4_code.dataset_loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN

from sklearn.metrics import accuracy_score, classification_report


# Step 1: Load the dataset
loader = Dataset_Loader()
data = loader.load()

trainData = data["trainData"]
trainLabel = data["trainLabel"]

testData = data["testData"]
testLabel = data["testLabel"]


# Step 2: Initialize RNN model
rnn = Method_RNN()


# Step 3: Train RNN and classify test data
predictions = rnn.run(
    trainData,
    trainLabel,
    testData
)


# Step 4: Calculate testing accuracy
accuracy = accuracy_score(testLabel, predictions)

print("\n===== STAGE 4.3 RESULTS =====")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    testLabel,
    predictions,
    target_names=["Negative", "Positive"],
    digits=4,
    zero_division=0
))
