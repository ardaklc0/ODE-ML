import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# check if GPU is available and use it; otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Building Neural Network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(1, 50)
        self.hidden_layer2 = nn.Linear(50, 50)
        self.hidden_layer3 = nn.Linear(50, 50)
        self.hidden_layer4 = nn.Linear(50, 50)
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        layer_out = torch.tanh(self.hidden_layer(x))
        output = self.output_layer(layer_out)
        return output
    
N = Network()
N = N.to(device)


# In order to test the code uncomment with Ctrl + K + U
# ----------------------------------------------------------------------------------- 

# # Solving ODE => dy/dx = e^x, x in [0, 1] and y(0) = 1
# def f(x):
#     return torch.exp(x)


# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     return torch.mean((dy_dx - f(x))**2 + (y[0] - 1)**2)


# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(0, 1, 100)[:, None]


# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val


# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)

    
# xx = torch.linspace(0, 1, 100)[:, None]
# with torch.no_grad():
#     yy = N(xx)

# plt.figure(figsize=(10, 6))
# plt.plot(xx, yy, label="Predicted")
# plt.plot(xx, torch.exp(xx), '--', label="Exact")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show()

# ----------------------------------------------------------------------------------- 

# # Solving ODE => y' + y = e^x, y(0) = 1
# def f(x):
#     return torch.exp(x)

# # Exact solution function
# def exact_solution(x):
#     return torch.exp(-x) * (0.5 * torch.exp(2 * x) + 0.5)

# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     return torch.mean((dy_dx + y - torch.exp(x))**2 + (y[0] - 1)**2)

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(0, 1, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val 

# # Plotting the initial state
# plt.figure(figsize=(10, 6))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)
    
#     xx = torch.linspace(0, 1, 100)[:, None]
#     with torch.no_grad():
#         yy = N(xx)
#         exact_yy = exact_solution(xx)
    
#     plt.clf()  # Clear the current figure
#     plt.plot(xx, yy, label="Predicted (Epoch {})".format(i+1))
#     plt.plot(xx, exact_yy, '--', label="Exact")
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid()
#     plt.pause(1)  # Pause to show the graph for a short time before proceeding to the next epoch
# plt.show()

# ----------------------------------------------------------------------------------- 

# # Solving ODE => y'' + y = 0, y(0) = 1, y'(0) = 1
# def f(x):
#     return 0

# # Exact solution function
# def exact_solution(x):
#     return torch.sin(x) + torch.cos(x)

# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     y_prime = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     y_double_prime = torch.autograd.grad(y_prime.sum(), x, create_graph=True)[0]
#     loss = torch.mean(
#         0.5*(y_double_prime + y - f(x))**2 +
#         0.5*(y[0] - 1)**2 + 
#         0.5*(y[0] - 1)**2
#     )
#     print(loss)
#     return loss

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(0, 1, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val

# # Plotting the initial state
# plt.figure(figsize=(10, 6))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)
    
#     xx = torch.linspace(0, 1, 100)[:, None]
#     with torch.no_grad():
#         yy = N(xx)
#         exact_yy = exact_solution(xx)
    
#     plt.clf()  # Clear the current figure
#     plt.plot(xx, yy, label="Predicted (Epoch {})".format(i+1))
#     plt.plot(xx, exact_yy, '--', label="Exact")
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid()
#     plt.pause(1)  # Pause to show the graph for a short time before proceeding to the next epoch
# plt.show()

# ----------------------------------------------------------------------------------- 

# # Solving ODE => y'' = -1, y(0) = 0, y'(0) = 0
# def f(x):
#     return -1

# # Exact solution function
# def exact_solution(x):
#     return -0.5 * x**2

# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     y_prime = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     y_double_prime = torch.autograd.grad(y_prime.sum(), x, create_graph=True)[0]
#     print("y[0]: ", y[0][0], "y_prime[0]: ", y_prime[0][0])
#     loss = torch.mean(
#         0.5*(y_double_prime + 0 - f(x))**2 +
#         0.5*(y[0] - 0.)**2 + 
#         0.5*(y_prime[0] - 0.)**2
#     )
#     return loss

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(0, 1, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val

# # Plotting the initial state
# plt.figure(figsize=(10, 6))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)
    
#     xx = torch.linspace(0, 1, 100)[:, None]
#     with torch.no_grad():
#         yy = N(xx)
#         exact_yy = exact_solution(xx)
    
#     plt.clf()  # Clear the current figure
#     plt.plot(xx, yy, label="Predicted (Epoch {})".format(i+1))
#     plt.plot(xx, exact_yy, '--', label="Exact")
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid()
#     plt.pause(1)  # Pause to show the graph for a short time before proceeding to the next epoch
# plt.show()

# ----------------------------------------------------------------------------------- 

# # Solving ODE => y'' = -1, y(0) = 0, y'(1) = 0
# def f(x):
#     return -1

# # Exact solution function
# def exact_solution(x):
#     return -0.5*(x-2)*x

# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     y_prime = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     y_double_prime = torch.autograd.grad(y_prime.sum(), x, create_graph=True)[0]
#     print("y[0]: ", y[0][0] - 0, "y_prime[-1]: ", y_prime[-1][0] - 0)
#     loss = torch.mean(
#         0.5*(y_double_prime + 0 - f(x))**2 +
#         0.5*(y[0] - 0.)**2 + 
#         0.5*(y_prime[-1] - 0.)**2
#     )
#     return loss

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(0, 1, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val

# # Plotting the initial state
# plt.figure(figsize=(10, 6))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)
    
#     xx = torch.linspace(0, 1, 100)[:, None]
#     with torch.no_grad():
#         yy = N(xx)
#         exact_yy = exact_solution(xx)
    
#     plt.clf()  # Clear the current figure
#     plt.plot(xx, yy, label="Predicted (Epoch {})".format(i+1))
#     plt.plot(xx, exact_yy, '--', label="Exact")
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid()
#     plt.pause(1)  # Pause to show the graph for a short time before proceeding to the next epoch
# plt.show()

# ----------------------------------------------------------------------------------- 

# # Solving ODE => y'' = -1, y(0) = 0, y(1) = 0
# def f(x):
#     return -1

# # Exact solution function
# def exact_solution(x):
#     return -0.5*(x-1)*x

# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     y_prime = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     y_double_prime = torch.autograd.grad(y_prime.sum(), x, create_graph=True)[0]
#     print("y[0]: ", y[0][0] - 0, "y_prime[-1]: ", y_prime[-1][0] - 0)
#     loss = torch.mean(
#         0.5*(y_double_prime + 0 - f(x))**2 +
#         0.5*(y[0] - 0.)**2 + 
#         0.5*(y[-1] - 0.)**2
#     )
#     return loss

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(0, 1, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val

# # Plotting the initial state
# plt.figure(figsize=(10, 6))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)
    
#     xx = torch.linspace(0, 1, 100)[:, None]
#     with torch.no_grad():
#         yy = N(xx)
#         exact_yy = exact_solution(xx)
    
#     plt.clf()  # Clear the current figure
#     plt.plot(xx, yy, label="Predicted (Epoch {})".format(i+1))
#     plt.plot(xx, exact_yy, '--', label="Exact")
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid()
#     plt.pause(1)  # Pause to show the graph for a short time before proceeding to the next epoch
# plt.show()


# # Solving ODE => y''' = -1, y(0) = 1, y'(1) = 1, y''(0) = 1
# def f(x):
#     return -1

# # Exact solution function
# def exact_solution(x):
#     return 1/6 * (-x**3 + 3*x**2 + 3*x + 6)

# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     y_prime = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     y_double_prime = torch.autograd.grad(y_prime.sum(), x, create_graph=True)[0]
#     y_triple_prime = torch.autograd.grad(y_double_prime.sum(), x, create_graph=True)[0]
#     loss = torch.mean(
#         0.5*(y_triple_prime - f(x))**2 +
#         0.5*(y[0] - 1.)**2 + 
#         0.5*(y_prime[-1] - 1.)**2 +
#         0.5*(y_double_prime[0] - 1.)**2
#     )
#     return loss

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(0, 1, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val

# # Plotting the initial state
# plt.figure(figsize=(10, 6))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)
    
#     xx = torch.linspace(0, 1, 100)[:, None]
#     with torch.no_grad():
#         yy = N(xx)
#         exact_yy = exact_solution(xx)
    
#     plt.clf()  # Clear the current figure
#     plt.plot(xx, yy, label="Predicted (Epoch {})".format(i+1))
#     plt.plot(xx, exact_yy, '--', label="Exact")
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid()
#     plt.pause(1)  # Pause to show the graph for a short time before proceeding to the next epoch
# plt.show()

# -----------------------------------------------------------------------------------
# # Solving ODE => dy/dx = e^x, x in [-5, 5] and y(-5) = 0
# def f(x):
#     return torch.exp(x)


# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     return torch.mean((dy_dx - f(x))**2 + (y[0] - 0)**2)

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(-5, 5, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)

# xx = torch.linspace(-5, 5, 100)[:, None]
# with torch.no_grad():
#     yy = N(xx)

# plt.figure(figsize=(10, 6))
# plt.plot(xx, yy, label="Predicted")
# plt.plot(xx, torch.exp(xx), '--', label="Exact")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show()

# -----------------------------------------------------------------------------------
# # Solving ODE => d^2y/dx^2 = -1, x in [-5, 5] and y(-5) = 0, y(5) = 0
# def f(x):
#     return -1

# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     y_prime = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     y_double_prime = torch.autograd.grad(y_prime.sum(), x, create_graph=True)[0]
#     loss = torch.mean(
#         0.5*(y_double_prime + 0 - f(x))**2 +
#         0.5*(y[-1] - 0.)**2 + 
#         0.5*(y[0] - 0.)**2
#     )
#     return loss

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(-5, 5, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)

# xx = torch.linspace(-5, 5, 100)[:, None]
# with torch.no_grad():
#     yy = N(xx)

# plt.figure(figsize=(10, 6))
# plt.plot(xx, yy, label="Predicted")
# plt.plot(xx, 0.5*(25-xx**2), '--', label="Exact")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show() 

# -----------------------------------------------------------------------------------
# # Solving ODE => dy/dx = e^x, x in [-5, 5] and y(-5) = 0
# def f(x):
#     return torch.exp(x)


# def loss(x):
#     x.requires_grad = True
#     y = N(x)
#     dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
#     return torch.mean((dy_dx - f(x))**2 + (y[0] - 0)**2)

# optimizer = torch.optim.LBFGS(N.parameters())
# x = torch.linspace(-5, 5, 100)[:, None]

# def closure():
#     optimizer.zero_grad()
#     loss_val = loss(x)
#     loss_val.backward()
#     return loss_val

# epochs = 10
# for i in range(epochs):
#     optimizer.step(closure)

# xx = torch.linspace(-5, 5, 100)[:, None]
# with torch.no_grad():
#     yy = N(xx)

# plt.figure(figsize=(10, 6))
# plt.plot(xx, yy, label="Predicted")
# plt.plot(xx, torch.exp(xx), '--', label="Exact")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show()

# -----------------------------------------------------------------------------------
# Solving ODE => y''' = -1, x in [-4, 0] and y(-4) = 0, y'(-3) = 0, y''(-2)=0
def f(x):
    return -1
def loss(x):
    x.requires_grad = True
    y = N(x)
    
    y_prime = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    y_double_prime = torch.autograd.grad(y_prime.sum(), x, create_graph=True)[0]
    y_triple_prime = torch.autograd.grad(y_double_prime.sum(), x, create_graph=True)[0]
    loss = torch.mean(
        0.5*(y_triple_prime - abs(f(x)))**2 +
        0.5*(y[0] - 0.)**2 + 
        0.5*(y_prime[249] - 0.)**2 +
        0.5*(y_double_prime[499] - 0.)**2
    )
    return loss

optimizer = torch.optim.LBFGS(N.parameters())
x = torch.linspace(-4, 0, 1000)[:, None]

def closure():
    optimizer.zero_grad()
    loss_val = loss(x)
    loss_val.backward()
    return loss_val

epochs = 10
for i in range(epochs):
    optimizer.step(closure)

xx = torch.linspace(-4, 0, 1000)[:, None]
with torch.no_grad():
    yy = N(xx)

plt.figure(figsize=(10, 6))
plt.plot(xx, yy, label="Predicted")
plt.plot(xx, (1/6)*((xx+1)**2)*(xx+4), '--', label="Exact")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show() 