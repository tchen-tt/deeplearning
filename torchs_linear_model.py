import torch
import numpy as np 
import math



print("Numpy")
print("---------------------------------------------------------")
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(3000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss = np.square(y_pred - y).sum()

    if t % 100 == 99:
        print(f"[{t:4d}], {loss:6.2f}")
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
print(f"Result: y = {a} + {b}x + {c}x^2 + {d}x^3")
print("---------------------------------------------------------")
print("")
print("Torch")
print("---------------------------------------------------------")

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device = device, dtype = dtype)
y = torch.sin(x)


a = torch.randn((), device = device, dtype = dtype)
b = torch.randn((), device = device, dtype = dtype)
c = torch.randn((), device = device, dtype = dtype)
d = torch.randn((), device = device, dtype = dtype)

learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loess = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(f"[{t:4d}], {loess:7.2f}")
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
print(f"Result: y = {a} + {b}x + {c}x^2 + {d}x^3")
print("---------------------------------------------------------")
print()
print("Autograd")
print("---------------------------------------------------------")
dtype = torch.float
device = torch.device("cpu")

a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loess = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(f"[{t:4d}], {loess:7.2f}")
    
    loess.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
print("---------------------------------------------------------")
print()
print("PyTorch: Define new autograd fucntions")
print("---------------------------------------------------------")
class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 -1)
dtype = torch.float
device = torch.device("cpu")
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    P3 = LegendrePolynomial3.apply
    y_pred = a + b * P3(c + d*x)

    loss = (y_pred -y).pow(2).sum()
    if t % 100 == 99:
        print(f"[{t:4d}], {loess.item():7.2f}")
    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
	c -= learning_rate * c.grad
	d -= learning_rate * d.grad



	a.grad = None 
	b.grad = None
        c.grad = None
	d.grad = None
print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')

