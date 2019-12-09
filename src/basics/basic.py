import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
def resiZing():
    a = torch.ones(2, 2)
    print(a)
    print(a.view(4).size())
    pass


def gradient():
    '''
    this will help to calculate gradient
    :return: none
    '''
    a = torch.ones((2, 2), requires_grad=True)
    print(a)
    print(a.requires_grad)

    b = torch.ones((2, 2))
    b.requires_grad_()
    print(b.requires_grad)

    ## addition with grad tensor
    print(a + b)
    print(torch.add(a, b))

    print(a * b)
    print(torch.mul(a, b))
    pass

def gradEq():
    '''
    this will ellustrate solving equiation
    y = 5 (xi + 1) ^2
    :return:
    '''
    x = torch.ones(2, requires_grad=True)
    print(x)

    y = 5 * (x + 1)**2
    print(y)
    o = (1/2) * torch.sum(y)
    print(o)

    ## calculating gradient
    o.backward(create_graph=True)
    print(x.grad)

    o.backward()
    print(x.grad)

def linearregression():
    '''
    simple linear regression
    :return: none
    '''
    # np.random.seed(1)
    n = 50
    x = np.random.randn(n)
    y = x * np.random.randn(n)

    colors = np.random.rand(n)

    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.show()

def linearPytorch():
    '''
    linear regression with pytorch
    :return: none
    '''
    xValues = [i for i in range(11)]
    print(xValues)

    ## converting list to np arrays
    xTrain = np.array(xValues, dtype=np.float32)
    print(xTrain.shape)

    ## reshape
    xTrain = xTrain.reshape(-1, 1)
    print(xTrain.shape)

    ## y = 2x + 1
    yValues = [2 * i + 1 for i in xValues]
    print(yValues)
    yTrain = np.array(yValues, dtype=np.float32)
    print(yTrain.shape)
    yTrain = yTrain.reshape(-1, 1)
    print(yTrain.shape)
    inputDim = 1
    outputDim = 1
    model = LinearRegression(inputDim, outputDim)
    criterion = nn.MSELoss()
    learningRate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

    ## Training
    epochs = 1000
    for epoch in range(epochs):
        epoch += 1
        inputs = torch.from_numpy(xTrain).requires_grad_()
        label = torch.from_numpy(yTrain)

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()
        print("epoch {}, loss {}".format(epoch, loss.item()))
    predicted = model(torch.from_numpy(xTrain).requires_grad_()).data.numpy()
    print("Predicted {}".format(predicted))
    print("Real {}".format(yTrain))

    ## plotting the result
    plt.clf()

    plt.plot(xTrain, yTrain, 'go', label='True data', alpha=0.5)
    plt.plot(xTrain, predicted, '--', label='Prediction', alpha=0.5)
    plt.legend(loc='best')
    plt.show()

    ## Saving the model
    torch.save(model.state_dict(), 'models/linearReg.pkl')

class LinearRegression(nn.Module):
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.linear = nn.Linear(inputDim, outputDim)

    def forward(self, x):
        out = self.linear(x)
        return out

def run():
    '''
    Entry point of the module
    :return:
    '''
    # resiZing()
    # gradient()
    # gradEq()
    linearPytorch()
    print(torch.cuda.is_available())
    pass


