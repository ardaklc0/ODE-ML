"""
1.1: There five types of DE: ODE(Ordinary), PDE(Partial), SDE(Stochastic), DDE(Delay), and DAE(Differential-Algebraic).
    1.1.1: ODE: ODE is a differential equation that contains one independent variable and one or more dependent variables.
    1.1.2: PDE: PDE is a differential equation that contains two or more independent variables and one or more dependent variables.
    1.1.3: SDE: SDE is a differential equation that contains one independent variable and one or more dependent variables, 
    and the dependent variables are subjected to random noise.
    1.1.4: DDE: DDE is a special type of ODE that contains a time delay.
    1.1.5: DAE: DAE is a generalization of ODE that contains algebraic equations as well as differential equations.

1.2: Typed of Differential Equation Problems: Initial Value Problem(IVP) and Boundary Value Problem(BVP), Dirichlet, Neumann, and Robin boundary conditions.

1.3: Differential Equations Associated with Physical Problems Arising in Engineering
    Coupled L-R electric circuit
    Motion of a pendulum
    Motion of a spring-mass-damper system
    Heat conduction in a rod
    Wave equation
    Laplace's equation

 1.4: General Introduction of Numerical Methods for Solving Differential Equations
 In field of mathematics the existence and uniqueness of the solution of a differential equation is guaranteed by various theorems, but no numerical method for
 obtaining those solutions in explicit and closed form is known. In view of this the limitations of nalytic methods in practical applications have led the 
 evolution of numerical methods for solving differential equations. 
    1.4.1: Shooting Method: The shooting method is a numerical method for solving boundary value problems. It is one of the most popular methods for solving 
    two-point boundary value problems. The shooting method is based on the idea of reducing a boundary value problem to an initial value problem. The solution
    of the initial value problems are then used to approximate (by adding two solutions) the solution of the boundary value problem. And calculated using 
    Runge-Kutta method(4th, 5th, 6th).
    1.4.2: Finite Difference Method: The finite difference method is a numerical method for solving differential equations. Functions are represented by their
    values at a finite number of points. It is an iterative method.
    1.4.3: Finite Element Method: The finite element method is a numerical method for solving differential equations. It is more general than the finite difference 
    method and more useful for real world problems. It is based on the idea of approximating the solution of a differential equation by a piecewise polynomial
    1.4.4: Finite Volume Method
    1.4.5: Spline Based Method
    1.4.6: Neural Network Method: NN can solve both ODE and PDE that relies on the approximation capabilities of feed forward neural networks. It minimizes the
    error between the actual and predicted values of the dependent variable. It requires the computation of the derivatives of the dependent variable with respect,
    which is also called gradient descent method.
"""
import numpy as np
"""
The oldest problem in the neural network is the XOR problem. The XOR problem is a two-class classification problem.
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def forward_propagation(inputs, weights, bias):
    return np.dot(inputs, weights) + bias
def backward_propagation(error, output):
    return error * sigmoid_derivative(output)
np.random.seed(5)
epochs = 1 # Number of iterations
lr = 0.1 # Learning rate
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])
expected_output = np.array([[0],
                            [1],
                            [1],
                            [0]])
hidden_weights = np.random.uniform(size=(2,2))
hidden_bias =np.random.uniform(size=(1,2))
output_weights = np.random.uniform(size=(2,1))
output_bias = np.random.uniform(size=(1,1))

print("Initial hidden weights: ",end='')
print(*hidden_weights)
print("Initial hidden biases: ",end='')
print(*hidden_bias)
print("Initial output weights: ",end='')
print(*output_weights)
print("Initial output biases: ",end='')
print(*output_bias)

for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    print("hidden_layer_activation_ohne_bias: ", hidden_layer_activation)
    hidden_layer_activation = hidden_layer_activation + hidden_bias
    print("hidden_layer_activation: ", hidden_layer_activation)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    print("hidden_layer_output: ", hidden_layer_output)
    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T) 
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights = output_weights + lr * hidden_layer_output.T.dot(d_predicted_output)
    output_bias = output_bias + lr * np.sum(d_predicted_output)
    hidden_weights = hidden_weights + lr * inputs.T.dot(d_hidden_layer)
    hidden_bias = hidden_bias + lr * np.sum(d_hidden_layer)

# print("Final hidden weights: ",end='')
# print(*hidden_weights)
# print("Final hidden bias: ",end='')
# print(*hidden_bias)
# print("Final output weights: ",end='')
# print(*output_weights)
# print("Final output bias: ",end='')
# print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*[1 if value > 0.5 else 0 for value in predicted_output])


# """
# For the AND gate we can just change the expected_output to [0, 0, 0, 1] 
# """
# inputs = np.array([[0,0],
#                    [0,1],
#                    [1,0],
#                    [1,1]])
# expected_output = np.array([[0],
#                             [0],
#                             [0],
#                             [1]])
# hidden_weights = np.random.uniform(size=(2,2))
# hidden_bias =np.random.uniform(size=(1,2))
# output_weights = np.random.uniform(size=(2,1))
# output_bias = np.random.uniform(size=(1,1))

# print("Initial hidden weights: ",end='')
# print(*hidden_weights)
# print("Initial hidden biases: ",end='')
# print(*hidden_bias)
# print("Initial output weights: ",end='')
# print(*output_weights)
# print("Initial output biases: ",end='')
# print(*output_bias)

# for _ in range(epochs):
#     # Forward Propagation
#     hidden_layer_activation = forward_propagation(inputs, hidden_weights, hidden_bias)
#     hidden_layer_output = sigmoid(hidden_layer_activation)
#     output_layer_activation = forward_propagation(hidden_layer_output, output_weights, output_bias)
#     predicted_output = sigmoid(output_layer_activation)

#     # Backpropagation
#     error = expected_output - predicted_output
#     d_predicted_output = backward_propagation(error, predicted_output)
#     error_hidden_layer = d_predicted_output.dot(output_weights.T) # .T means transpose
#     d_hidden_layer = backward_propagation(error_hidden_layer, hidden_layer_output)

#     # Updating Weights and Biases
#     output_weights = output_weights + lr * hidden_layer_output.T.dot(d_predicted_output)
#     output_bias = output_bias + lr * np.sum(d_predicted_output)
#     hidden_weights = hidden_weights + lr * inputs.T.dot(d_hidden_layer)
#     hidden_bias = hidden_bias + lr * np.sum(d_hidden_layer)

# print("Final hidden weights: ",end='')
# print(*hidden_weights)
# print("Final hidden bias: ",end='')
# print(*hidden_bias)
# print("Final output weights: ",end='')
# print(*output_weights)
# print("Final output bias: ",end='')
# print(*output_bias)

# print("\nOutput from neural network after 10,000 epochs: ",end='')
# print(*[1 if value > 0.5 else 0 for value in predicted_output])