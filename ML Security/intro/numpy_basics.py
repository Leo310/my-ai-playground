# %%

import numpy as np

# %%
# (1) Vector, Matrix

x0 = np.array([1, 2, 3, 4])
x1 = np.array([5, 6, 7, 8])

# %%
# (2) Operations (sum, mul, ...)

print(x0 + x1)

print(x0.mean())
print(x0.sum())

print(x0 + 2 * x1)

# %%
# (3) Axis, Indexing, Slicing

x = np.array([1, 2, 3, 4])
print(x.shape)
print(x[0:3])


X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]])
print(X.shape)


# %%
# (4) Broadcasting

# For a full explanation see: https://numpy.org/doc/stable/user/basics.broadcasting.html
# When operating on two arrays, NumPy compares their shapes element-wise. It starts
# with the trailing (i.e. rightmost) dimension and works its way left. Two dimensions
# are compatible when
# - they are equal, or
# - one of them is 1.

# This works as [1] gets extended to [1, 1, 1] and then stretched to [1, 2, 3]
X = np.array([1, 2, 3])
Y = np.array([1])
print(X + Y)

# This works because [1, 3] gets extended to [1, 1, 3] and stretched to [6, 2, 3]
X = np.random.random((6, 2, 3))
Y = np.random.random((1, 3))
print(X + Y)

# This does not work, because once we extend the dimensions we get [1, 6, 2] which
# can not be stretched to fill [6, 2, 3] (mismatch on second dimension)
X = np.random.random((6, 2, 3))
Y = np.random.random((6, 2))
print(X + Y)


# %%
# (5) Masking

X = np.random.random((10, 5))
print(X)

mask = X < 0.5
print(mask)
print(X[mask])

# %%
# (6) "Vectorization"


def f(x):
    return x**2 + 3


print(f(np.array([3, 4, 5])))

# %%
# (7) Misc

# (7.1) Functions
# np.random.normal
# np.linspace
# np.meshgrid
# np.exp, np.log, np.max, np.min, np.mean, np.std

print(np.random.normal(6, 1))
print(np.linspace(-10, 10, 10))


# %%
# (8) Matplotlib

import matplotlib.pyplot as plt


def f(x):
    return x**2


X = np.linspace(-10, 10, 1000)
Y = f(X)

plt.plot(X, Y)
plt.show()


X = np.linspace(-10, 10, 1000)
Y = np.linspace(-10, 10, 1000)

XX, YY = np.meshgrid(X, Y)

Z = XX**2 + YY * 2

plt.contourf(X, Y, Z, cmap="hot")

# %%
