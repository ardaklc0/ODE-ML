import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib . pyplot as plt
import seaborn as sns
import argparse

#Initialization and Some Presettings:
sns . color_palette ('bright')
parser = argparse.ArgumentParser()
parser.add_argument ('--method', type =str , choices =[ 'dopri5 ', 'adams '], default ='dopri5 ')
#dopri5: https://www.rdocumentation.org/packages/deTestSet/versions/1.1.7.4/topics/dopri5
#adams: https://en.wikipedia.org/wiki/Linear_multistep_method#Two-step_Adams%E2%80%93Bashforth
parser.add_argument ('--data_size', type =int , default = 1200)
parser.add_argument ('--batch_time', type =int , default =10)
parser.add_argument ('--batch_size', type =int , default =20)
parser.add_argument ('--niters', type =int , default =400)
parser.add_argument ('--test_freq', type =int , default =20)
parser.add_argument ('--viz', action ='store_true')
parser.add_argument ('--gpu', type =int , default =0)
parser.add_argument ('--adjoint', action ='store_true')
args = parser.parse_args()
if args.adjoint :
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
device = torch.device ('cuda:' + str ( args.gpu ) if torch.cuda . is_available () else 'cpu') #Enables usage of gpu for faster computation

class Nabla(nn.Module):
    def __init__(self, true_dy):
        super(Nabla, self).__init__()
        self.true_dy = true_dy

    def forward(self, t, y):
        return torch.mm(y, self.true_dy)


true_y0 = torch.tensor([[6., 0.]]).to(device) # Initial Condition
t = torch.linspace(0., 25., args.data_size).to(device) # Time step t_0 to t_N for --data_size
true_dy = torch.tensor([[-0.1, -1.],
                        [1., -0.1]]).to(device) # Observe that this is same equation as in Experiment.png
                                                # This is also Spiral ODE
true_A = torch.tensor([[-0.1, 2.0],
                       [-2.0, -0.1]]).to(device)

class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    pred_y = odeint(Lambda(), true_y0, t, method='dopri5')

print("pred_y: ", pred_y)
