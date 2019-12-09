import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
import time


def cnn():
    trainDataset = datasets.MNIST(root='data', train=True,
                                  transform=transforms.ToTensor(),
                                  download=True)
    print(trainDataset.train_data.size())
    print(trainDataset.train_labels.size())
    testDataset = datasets.MNIST(root='data', train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)
    print(testDataset.test_data.size())
    print(testDataset.test_labels.size())

    BATCH_SIZE = 100
    N_ITERS = 3000
    N_EPOCHS = N_ITERS / (len(trainDataset) / BATCH_SIZE)
    N_EPOCHS = int(N_EPOCHS)
    device = 'cpu'
    LR_RATE = 0.01
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)
    ## NN class
    model = CNN().to(device)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR_RATE)
    optimizer = torch.optim.Adam(model.parameters())

    ## Training
    iter = 0
    for epoch in range(N_EPOCHS):
        for i, (image, label) in enumerate(trainLoader):
            image = image.requires_grad_().to(device)
            # label.to(device)
            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            iter += 1
            if iter % 500 == 0:
                correct = 0
                total = 0

                for image, label in testLoader:
                    image = image.requires_grad_().to(device)
                    outputs = model(image)
                    _, predict = torch.max(outputs.data, 1)
                    predict = predict.cpu()
                    total += label.size(0)

                    correct += (predict == label).sum()
                accuracy = 100 * correct / total
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

    pass


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        # max pooling1
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2)

        # convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # max pooling 2
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2)

        ##fully connected layer
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # conv 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # max pooling 1
        out = self.maxpooling1(out)

        ##conv 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpooling2(out)
        ## resize the image
        # orignal = (100, 32, 7, 7)
        # new out sieze = (100, 32*7*7)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out


def run():
    startTime = time.time()
    cnn()
    endTime = time.time()
    print("Required Time : {}".format(endTime - startTime))
