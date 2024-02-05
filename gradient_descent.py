from sympy import symbols
from sympy import diff
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# x, y = symbols('x y')
# a, b, c = symbols('a b c')
# alpha, beta, gamma = symbols('alpha beta gamma')
# func = x**2 - 4*x + 1
# print("func: ", func)
# derivative_of_func = diff(func, x)
# print("derivative_of_func: ", derivative_of_func)


# x = torch.tensor(0.0, requires_grad = True)
# print("x:", x)
# y = x**2 + 1
# y.backward()
# dx = x.grad
# print("dy/dx:", dx)


# # create tensor without requires_grad = true
# x = torch.tensor(3)
# # create tensors with requires_grad = true
# w = torch.tensor(2.0, requires_grad = True)
# b = torch.tensor(5.0, requires_grad = True)

# # print the tensors
# print("x:", x)
# print("w:", w)
# print("b:", b)

# # define a function y for the above tensors
# y = w*x + b
# print("y:", y)

# # Compute gradients by calling backward function for y
# y.backward()

# # Access and print the gradients w.r.t x, w, and b
# dx = x.grad
# dw = w.grad
# db = b.grad
# print("x.grad :", dx)
# print("w.grad :", dw)
# print("b.grad :", db)



# Estimates the gradient of f(x)=x^2 at points [-2, -1, 1, 4]
coordinates = (torch.tensor([-2., -1., 1., 4.]),)
values = torch.tensor([4., 1., 4., 16.], ) # f(-2) = 4, f(-1) = 1, f(1) = 1, f(4) = 16
torch.gradient(values, spacing = coordinates)
print("torch.gradient(values, spacing = coordinates): ", torch.gradient(values, spacing = coordinates))
# Estimates the gradient of the R^2 -> R function whose samples are
# described by the tensor t. Implicit coordinates are [0, 1] for the outermost
# dimension and [0, 1, 2, 3] for the innermost dimension, and function estimates
# partial derivative for both dimensions.
t = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]])
gradient = torch.gradient(t)

print("torch.gradient(t): ", torch.gradient(t))

gradient = torch.gradient(values, spacing = coordinates)

# Function and gradient definitions
def f(x):
    return x**2

def df_dx(x):
    return 2*x

# Generate more points for a smooth curve
x_smooth = torch.linspace(-2, 4, 100)
y_smooth = f(x_smooth)

# Cubic spline interpolation
spline = CubicSpline(x_smooth, y_smooth)

# Sample points
x_values = torch.tensor([-2., -1., 1., 4.])
y_values = f(x_values)

# Gradient at sample points
gradient_values = 2 * x_values

# Scaling factor for line length
line_length_scale = 1.5  # Adjust this scale factor as needed

# Plotting the smooth function
plt.figure(figsize=(8, 6))
plt.plot(x_smooth.numpy(), spline(x_smooth), 'b-', label='Smooth f(x) = x^2')  # Removed .numpy() here

# Plotting the gradient
plt.quiver(x_values.numpy(), f(x_values).numpy(), np.ones_like(x_values.numpy()), gradient_values.numpy(),
           angles='xy', scale_units='xy', scale=10, color='r', label='Gradient')

# Adding linear lines corresponding to the gradient
for x, y, gradient in zip(x_values, y_values, gradient_values):
    x_end = x + 1
    x_start = x - 1
    y_end = y + 1 * gradient * line_length_scale
    y_start = y - 1 * gradient * line_length_scale
    plt.plot([x_start.numpy(), x_end.numpy()], [y_start.numpy(), y_end.numpy()], 'g--')

# Adding labels and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Smooth Function, its Gradient, and Longer Linear Lines')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()