import pickle
import gzip
import numpy as np
from matplotlib import pyplot
import torch
import math
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

with gzip.open("/Users/chentao/Desktop/code/data/mnist/mnist.pkl.gz", 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
pyplot.show()


x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return log_softmax(xb @ weights + bias)

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


loss_func = nll

bs = 64
xb = x_train[0:bs]
preds = model(xb)
yb = y_train[0:bs]
print(loss_func(preds, yb))
print(accuracy(preds, yb))



lr = 0.5
epochs = 2
for epoch in range(epochs):
    for i in range(1, 100):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]

        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()

        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(10), requires_grad=True)

    def forward(self, xb):
        return xb @ self.weights + self.bias


def get_model(lr=1e-3):
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)
loss_func = F.cross_entropy

model, opt = get_model()

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
for epoch in range(2000):
    for xb, yb in train_dl:
        prd = model(xb)

        loss = loss_func(prd, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f'[{epoch}], accuracy {accuracy(model(x_valid), y_valid)}')

print(loss_func(model(xb), yb))


