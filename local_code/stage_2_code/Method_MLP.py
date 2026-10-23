'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # 2.4
        # run 3: changed the architecture from 784 -> 256 -> 128 -> 10 to 784 -> 512 -> 256 -> 10
        # run 4: deepened network- 784 -> 512 -> 256 -> 128 -> 10 (added fourth layer)
        self.fc_layer_1 = nn.Linear(784, 512)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(512, 256)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.fc_layer_3 = nn.Linear(256, 128)
        self.fc_layer_4 = nn.Linear(128, 10)
        # 2.4
        # run 1: added 0.3 dropout rate for regularization
        # run 2: 0.3 changed to 0.2
        self.dropout = nn.Dropout(0.2)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h1 = self.dropout(self.activation_func_1(self.fc_layer_1(x)))
        h2 = self.dropout(self.activation_func_2(self.fc_layer_2(h1)))
        # output layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.fc_layer_3(h2)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        loss_history = []
        accuracy_history = []

        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            loss_history.append(train_loss.item())

            accuracy = (y_pred.argmax(dim=1) == y_true).float().mean().item()
            accuracy_history.append(accuracy)

            if epoch%10 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("MLP Learning Convergence")
        plt.show()

        plt.plot(accuracy_history)
        plt.xlabel("Epoch")
        plt.ylabel("Training Accuracy")
        plt.title("MLP Training Accuracy")
        plt.show()

    def test(self, X):
        # do the testing, and result the result
        self.dropout.eval()
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')

        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        true_y = self.data['test']['y']

        # Calculate evaluation metrics
        accuracy = accuracy_score(true_y, pred_y)
        precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        recall = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        f1 = f1_score(true_y, pred_y, average='weighted', zero_division=0)

        print("Testing Accuracy:", accuracy)
        print("Testing Precision:", precision)
        print("Testing Recall:", recall)
        print("Testing F1 Score:", f1)

        return {'pred_y': pred_y, 'true_y': true_y}
