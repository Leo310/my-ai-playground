# %%
# (1)
# What does np.meshgrid() do?

import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x**2 + y**2


resolution = 5
x = np.linspace(-10, 10, resolution)
y = np.linspace(-10, 10, resolution)

# Let's first define, what we actually want to plot. We have a function "f" that
# takes in the parameters x and y and we want to plot each combination in the
# grid.
#
# z[i, j] = f(x[i], z[j]) where -10 < i,j < 10
#

# What would happen if we did not use meshgrid?
z = f(x, y)
print("z.shape", z.shape)

# With this formulation, we only receive a shape of z.shape == (resolution, )
# So what is happening here and why is that wrong?

# We are passing in two vectors to "f" which get evaluated element-wise. So our
# output from f(x, y) is equivalent to:
#
#   z[i] = f(x[i], y[i])
#
# i.e.,
#   z[0] = f(x[0], y[0])
#   z[1] = f(x[1], y[1])
#   ...
#
# What is wrong with this? We only evaluated the diagonal line (i==j) of our 2D
# grid, but not the entire grid!

z_diagonal = np.array([f(x_, y_) for x_, y_ in zip(x, y)])

print("z: ", z)
print("z[i] = f(x[i], y[i]):", z_diagonal)
assert (z == z_diagonal).all()

# Now how does np.meshgrid help us here?
X, Y = np.meshgrid(x, y)

# As you can see, meshgrid "duplicates" the single dimensional x for all
# dimensions in y (vice versa), thus creating a grid for x values and y values.
print("x: ", x)
print("X: ", X)
print("X.shape", X.shape)

# If we now evaluate "f" on X and Y, we no longer evaluate on the diagonal, but
# element-wise for each entry in the two grids, i.e., for each combination
# of x and y value.
#
#   z[i, j] = f(x[i], y[j])
#
Z = f(X, Y)

# We can also recreate this without numpy with a nested list comprehension in
# python. However, numpy uses optimized C functions to process the data which
# is much faster for large arrays.
Z_grid = np.array([[f(x_, y_) for y_ in y] for x_ in x])
print("Z: ", Z)
print("Z[i,j] = f(x[i], y[j]):", Z_grid)
assert (Z == Z_grid).all()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z)
plt.show()

# %%
