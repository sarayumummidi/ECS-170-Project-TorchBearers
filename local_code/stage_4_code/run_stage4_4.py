from local_code.stage_4_code.dataset_loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation

loader = Dataset_Loader(task="generation")
data = loader.load()

train_data = data["trainData"]

# first 5 are from the dataset, last 2 are our own
startingWords = [
    "what did the",
    "why did the",
    "two artists had",
    "what do you",
    "how do you",
    "do you like",
    "my favorite thing"
]

rnn = Method_RNN_Generation(cell_type="rnn")

results = rnn.run(train_data, startingWords)

print("\n===== STAGE 4.4 RESULTS =====")
for result in results:
    print(f"\nStart:            {result['start']}")
    print(f"Generated:        {result['generated']}")
    print(f"Closest training: {result['closest_training_joke']}")
    print(f"Similarity: {result['similarity']:.2f}   ")
    print(f"Exact copy: {result['exact_copy_check']}")
