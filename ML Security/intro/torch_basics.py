# %%

import numpy as np
import torch

# %%
# (1) Numpy -> Torch, Torch -> Numpy


X_np = np.array([1, 2, 3, 4])
X_torch = torch.tensor(X_np)

print("Torch from list: ", torch.tensor([1, 2, 3, 4]))
print(torch.tensor([1.0, 2.0, 3.0, 4.0]).dtype)

print(X_np.shape)
print(X_torch.shape)
print(X_np)
print(X_torch)

print(X_torch.numpy())

# %%
# (2) Function evaluation


def f(x):
    return x**2


x = torch.tensor([5])
print(f(x))

# %%
# (3) Matplotlib

import matplotlib.pyplot as plt


def f(x):
    return torch.sin(x)


x = torch.linspace(-10, 10, 1000)

plt.plot(x, f(x))
plt.show()

# %%
# (4) Automatic Differentiation

x = torch.tensor([4.0], requires_grad=True)


def f(x):
    return x**3


def f_dev(x):
    return 3 * x**2


y = f(x)
y.backward()

print(y)
print("f'(x)", x.grad, f_dev(x))

# %%
