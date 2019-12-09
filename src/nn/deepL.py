'''
in this module ill learn about the deep lerning though pytorch
'''
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
import time

def feedForward():
    trainDataset = datasets.MNIST(root='data', train=True,
                                  transform=transforms.ToTensor(),
                                  download=True)
    testDataset = datasets.MNIST(root='data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
    BATCH_SIZE = 100
    N_ITERS = 3000
    N_EPOCHS = N_ITERS / (len(trainDataset)/ BATCH_SIZE)
    N_EPOCHS = int(N_EPOCHS)
    INPUT_DIM = 28*28
    HIDDEN_DIM = 100
    OUTPUT_DIM = 10
    device = 'cpu'
    LR_RATE = 0.1

    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                              batch_size = BATCH_SIZE,
                                              shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset,
                                              batch_size = BATCH_SIZE,
                                              shuffle=False)
    ## NN class
    model = FeedForwardNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

    ## Loss Class
    criterion = nn.CrossEntropyLoss()

    ## optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_RATE)

    #training Process
    iter = 0
    for epoch in range(N_EPOCHS):
        for i, (images, labels) in enumerate(trainLoader):
            # load images with gradient accumulation capability
            images = images.view(-1, 28*28).requires_grad_().to(device)
            labels = labels.to(device)
            ## Clear Gradients
            optimizer.zero_grad()

            ## forward pass
            output = model(images)

            ## calculate loss
            loss = criterion(output, labels)

            ##getting gradient with respect to params
            loss.backward()

            ## update params
            optimizer.step()

            iter += 1
            if iter % 500 == 0:
                # calculate accuracy
                correct = 0
                total = 0
                # iterate through test dataset
                for images, labels in testLoader:
                    images = images.view(-1, 28*28).requires_grad_().to(device)
                    output = model(images)

                    # Get Prediction for max
                    _, predicted = torch.max(output.data, 1)
                    predicted = predicted.cpu()

                    total += labels.size(0)

                    correct += (predicted== labels).sum()
                accuracy = 100 * correct / total
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


## create Model Class
class FeedForwardNN(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super().__init__()

        ##linear function
        self.fc1 = nn.Linear(inputDim, hiddenDim)

        ## Non Linearity
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        self.ReLU1 = nn.ReLU()
        ##Linear function (readOut)
        self.fc2 = nn.Linear(hiddenDim, hiddenDim)
        self.ReLU2 = nn.ReLU()

        self.fc3 =  nn.Linear(hiddenDim, hiddenDim)
        self.ReLU3 = nn.ReLU()

        self.fc4 =  nn.Linear(hiddenDim, hiddenDim)
        self.ReLU4 = nn.ReLU()

        self.fc5 = nn.Linear(hiddenDim, outputDim)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.sigmoid(out)
        # out = self.tanh(out)
        out = self.ReLU1(out)
        out = self.fc2(out)
        out = self.ReLU2(out)
        out = self.fc3(out)
        out = self.ReLU3(out)
        out = self.fc4(out)
        out = self.ReLU4(out)
        out = self.fc5(out)

        return out

def run():
    '''
    starting point
    '''
    startTime = time.time()
    feedForward()
    endTime = time.time()
    print("Required Time : {}".format(endTime - startTime))
    pass