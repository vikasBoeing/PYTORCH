'''
this module will help to learning logistic regression with help of pytorch
'''

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time

def logisticReg():
    '''
    this is the logistic regression problem
    '''

    trainDataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    print(len(trainDataset))
    showImg = trainDataset[0][0].numpy().reshape(28, 28)
    plt.imshow(showImg, cmap='gray')
    plt.show()
    print("Okay")
    testDataset = datasets.MNIST(root='data/', train=False,
                                 transform=transforms.ToTensor())
    print(len(testDataset))

    BATCHSIZE = 100
    N_ITERATIONS = 3000
    N_EPOCHS = 5
    LR = 0.001

    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                              batch_size=BATCHSIZE,
                                              shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset,
                                             batch_size=BATCHSIZE,
                                             shuffle=False)
    inputDim = 28*28
    outputDim = 10
    model = LogisticRe(inputDim, outputDim)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    ## Training Model
    iter = 0
    for epoch in range(N_EPOCHS):
        for i, (images, labels) in enumerate(trainLoader):
            # Load images as Variable
            images = images.view(-1, 28 * 28).requires_grad_()
            labels = labels

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in testLoader:
                    # Load images to a Torch Variable
                    images = images.view(-1, 28 * 28).requires_grad_()

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / total

                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

## Building Model
class LogisticRe(nn.Module):
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.linear = nn.Linear(inputDim, outputDim)
    def forward(self, x):
        out = self.linear(x)
        return out


def run():
    startTime = time.time()
    logisticReg()
    endTime = time.time()
    print("Total Time : {}".format(endTime-startTime))
    pass