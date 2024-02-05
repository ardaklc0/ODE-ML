import numpy as np 
import torch

def sigmoid(z):
    return 1/(1 + torch.exp(-z))
def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

torch.set_printoptions(precision=15)

## True values. These are the actual values. These are observed values. Using for
## example let in input matrix X second row represent "weight of person" and third
## row represent "height of person". Using these values we want to predict if the
## person is obese or not. So, these are the actual values.
## Observe that [1., 30., 35. ] are one input.
y = torch.tensor([[1.0],   # --> Obese
                  [0.0],   # --> Not Obese
                  [0.0],   # --> Not Obese
                  [1.0]]) # --> Obese   
print("y: ", y)

## Input Matrix X. 
# x_11, x_21, x_31, x_41 stand for Bias terms. These Bias terms are just placeholder.
# There are 4 samples and 2 features (one of the columns is bias).
X = torch.tensor([[1., 30., 35.],
                  [1., 25., 18.],
                  [1., 45., 30.],
                  [1., 40., 32.]])
# Initialize Weight matrices with random inputs.
w_1 = torch.tensor([[0.44, 0.47],
                    [0.64, 0.67],
                    [0.67, 0.09]])
print("w_1: ", w_1)
w_2 = torch.tensor([[0.83],
                    [0.21],
                    [0.36]])
print("w_2: ", w_2)


## Hidden Layer
# Forward Propagation
z_2 = torch.mm(X, w_1)
print("z_2: ", z_2)
# Using activation function
a_2 = sigmoid(z_2)
print("a_2: ", a_2)
# Adding bias term to a_2
a_2 = torch.cat((torch.ones((X.shape[0], 1)), a_2), 1)
print("a_2: ", a_2)


## Output Layer
z_3 = torch.mm(a_2, w_2)
print("z_3: ", z_3)
y_hat = sigmoid(z_3)
print("a_3: ", y_hat) # These are y_hat values. These are the predictions.


## Loss Function (Binary Cross Entropy) we calculate L(w_1) and L(w_2)
## -1/n * SUM(y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i))
## Loss Function (Binary Cross Entropy)
bce = 0
for i in range(4):
    bce += y[i] * torch.log(y_hat[i]) + (1 - y[i]) * torch.log(1 - y_hat[i])
bce = (-1/4 * bce).item()
print("bce: ", bce)


# Our objective is to minimize the loss function. Using the optimization algorithm called
# Gradient Descent we will update the weights and biases 
# w_1 = w_1 - alpha * dL/dw_1
# w_2 = w_2 - alpha * dL/dw_2


## Backward Propagation: the partial derivative or the gradient is calculated using a process
## called backpropagation. First, error contributed by each node called delta is calculated
## by traversing backward from the output layer towards the input layer. The error term is then
## multiplied with the activation value of the node to determine partial derivative.
## delta_j^l error of node j in layer l


## Output Layer
# delta_3 = (y_hat - y)
delta_3 = y_hat - y
print("delta_3: ", delta_3)


## Hidden Layer
# delta_2 = delta_3 * w_2 * a_2 * (1 - a_2)
delta_2 = torch.mm(delta_3, torch.t(w_2)) * a_2 * (1 - a_2)
print("delta_2: ", delta_2)


## There is no error term for input layer


## Partial Derivatives
D_2 = torch.mm(torch.t(a_2), delta_3)
print("D_2: ", D_2)
delta_2_without_bias = delta_2[:, 1:]
print("delta_2_without_bias: ", delta_2_without_bias)
D_1 = torch.mm(torch.t(X), delta_2_without_bias)
print("D_1: ", D_1)


## Update Weights
alpha = 0.005
w_1 = w_1 - alpha * D_1
print("w_1: ", w_1)
w_2 = w_2 - alpha * D_2
print("w_2: ", w_2)
print("bce: ", bce)





