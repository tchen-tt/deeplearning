# -*- coding: utf-8 -*-

import math
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from matplotlib import pyplot as plt



# tc = w * tu + b
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] 
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] 
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

# create linear model
def model(t_u, w, b):
    return w * t_u + b

# mean square error(MSE)
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2 
    return squared_diffs.mean()

# partial derivative with respect to w
def dmodelw(t_u, w, b):
    return t_u
# partial derivative with respect to b
def dmodelb(t_u, w, b):
    return 1.0
def dloss_fn(t_p, t_c):
    return 2 * (t_p - t_c)
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c) * dmodelw(t_u, w, b)
    dloss_db = dloss_fn(t_p, t_c) * dmodelb(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)

        params = params - learning_rate * grad
        print(params)
        # print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params



def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        params = (params - learning_rate * params.grad).detach().requires_grad_()
        if epoch % 10 == 0:
            print("Epoch %d, Loss %f" % (epoch, float(loss)))
    return params

training_loop(n_epochs=5000, learning_rate=1e-2, params=torch.tensor([1.0, 0], requires_grad=True),
            t_u = t_u, t_c = t_c)


def tarining_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print("Epoch %d, Loss %f" % (epoch, float(loss)))

    print(params)
    return params

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
# optimizer = optim.SGD([params], lr=learning_rate)
optimizer = optim.Adam([params], lr=learning_rate)
tarining_loop(n_epochs=5000,
              optimizer=optimizer,
              params=params,
              t_u=t_u,
              t_c=t_c)


def train_loop2(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        val_t_p = model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print("Epoch {}, Train loss {}, Validation loss {}".format(epoch, float(train_loss), float(val_loss)))
    print(params)
    return params


# split sample with train and valid
# train 0.8, valid 0.2
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

# random index
shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.Adam([params], lr=learning_rate)

train_loop2(
    n_epochs=5000,
    optimizer=optimizer,
    params=params,
    train_t_u=train_t_u,
    val_t_u=val_t_u,
    train_t_c=train_t_c,
    val_t_c=val_t_c
)


def train_loop3(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epochs in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    print(params)
    return params


train_loop3(
    n_epochs=50000,
    optimizer=optimizer,
    params=params,
    train_t_u=train_t_u,
    val_t_u=val_t_u,
    train_t_c=train_t_c,
    val_t_c=val_t_c
)


def training_loop3(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(epoch, float(loss_train), float(loss_val)))




linear_model = nn.Linear(1, 1)
optimizer = optim.Adam(seq_model3.parameters(), lr=1e-2)

training_loop3(
    n_epochs=3000,
    optimizer=optimizer,
    model=linear_model,
    loss_fn=nn.MSELoss(),
    t_u_train=train_t_u,
    t_u_val=val_t_u,
    t_c_train=train_t_c,
    t_c_val=val_t_c
)


seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.ReLU(),
    nn.Linear(13, 1)
)

seq_model2 = nn.Sequential(
    OrderedDict([
        ('hidden_linear', nn.Linear(1, 8)),
        ('hidden_activation', nn.Tanh()),
        ('output_linear', nn.Linear(8, 1))
    ])
)

seq_model3 = nn.Linear(1, 1)

optimizer = optim.Adam(seq_model.parameters(), lr=1e-2)
training_loop3(
    n_epochs=50000,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(),
    t_u_train=train_t_u.unsqueeze(1),
    t_u_val=val_t_u.unsqueeze(1),
    t_c_train=train_t_c.unsqueeze(1),
    t_c_val=val_t_c.unsqueeze(1)
)

training_loop3(
    n_epochs=50000,
    optimizer=optimizer,
    model=seq_model3,
    loss_fn=nn.MSELoss(),
    t_u_train=train_t_u.unsqueeze(1),
    t_u_val=val_t_u.unsqueeze(1),
    t_c_train=train_t_c.unsqueeze(1),
    t_c_val=val_t_c.unsqueeze(1)
)

print()
print(linear_model.weight)
print(linear_model.bias)



class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)
    def forward(self):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t
subclass_model = SubclassModel()
subclass_model


from matplotlib import pyplot as plt
t_range = torch.arange(20.0, 90.0).unsqueeze(1)
fig = plt.figure(dpi=600)
plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(t_range).detach().numpy(), "c-")
plt.plot(t_u, seq_model(t_u.unsqueeze(1)).detach().numpy(), 'kx')
fig.show()


