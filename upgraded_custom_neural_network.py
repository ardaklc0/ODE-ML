import numpy as np 
import torch

def sigmoid(z):
    return 1/(1 + torch.exp(-z))
def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

## Observed Values
torch.manual_seed(10)
y = torch.randint(0, 2, (40, 1)).type(torch.FloatTensor)
print("y: ", y)

## Input Matrix X. Let input be 40 samples and 5 features. Represent:
## 1. Bias term (1)
## 2. Weight of person (2)
## 3. Height of person (3)
## 4. Age of person (4)
## 5. Marriage Status (5)

biases = torch.ones(40, 1)
weights = torch.randint(60, 120, (40, 1)).type(torch.FloatTensor)
heights = torch.randint(150, 190, (40, 1)).type(torch.FloatTensor)
ages = torch.randint(20, 40, (40, 1)).type(torch.FloatTensor)
marriage_status = torch.randint(0, 2, (40, 1)).type(torch.FloatTensor)
X = torch.cat((biases, weights, heights, ages, marriage_status), 1)
print("X: ", X) # 40x5


w_1 = torch.randn(5, 4) # 5x4
print("w_1: ", w_1)
w_2 = torch.randn(5, 1) # 5x1
print("w_2: ", w_2)

z_2 = torch.mm(X, w_1) # 40x4
print("z_2: ", z_2)
a_2 = sigmoid(z_2)
print("a_2: ", a_2)
a_2 = torch.cat((torch.ones((X.shape[0], 1)), a_2), 1)
print("a_2: ", a_2) # 40x5

z_3 = torch.mm(a_2, w_2)
print("z_3: ", z_3)
y_hat = sigmoid(z_3)
print("y_hat: ", y_hat)

bce = 0
for i in range(40):
    bce += y[i] * torch.log(y_hat[i]) + (1 - y[i]) * torch.log(1 - y_hat[i])
bce = (-1/4 * bce).item()
delta_3 = y_hat - y
delta_2 = torch.mm(delta_3, torch.t(w_2)) * a_2 * (1 - a_2)
D_2 = torch.mm(torch.t(a_2), delta_3)
delta_2_without_bias = delta_2[:, 1:]
D_1 = torch.mm(torch.t(X), delta_2_without_bias)
alpha = 0.005
w_1 = w_1 - alpha * D_1
w_2 = w_2 - alpha * D_2

for i in range(500):
    z_2 = torch.mm(X, w_1)
    a_2 = sigmoid(z_2)
    a_2 = torch.cat((torch.ones((X.shape[0], 1)), a_2), 1)
    z_3 = torch.mm(a_2, w_2)
    y_hat = sigmoid(z_3)
    bce = 0
    for i in range(40):
        bce += y[i] * torch.log(y_hat[i]) + (1 - y[i]) * torch.log(1 - y_hat[i])
    bce = (-1/4 * bce).item()
    delta_3 = y_hat - y
    delta_2 = torch.mm(delta_3, torch.t(w_2)) * a_2 * (1 - a_2)
    D_2 = torch.mm(torch.t(a_2), delta_3)
    delta_2_without_bias = delta_2[:, 1:]
    D_1 = torch.mm(torch.t(X), delta_2_without_bias)
    w_1 = w_1 - alpha * D_1
    w_2 = w_2 - alpha * D_2
    print("bce: ", bce)
