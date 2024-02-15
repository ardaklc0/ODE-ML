import torch.nn as nn
import torch

class Nabla(nn.Module):
    def __init__(self):
        super(Nabla, self).__init__()

    def forward(self, y, true_dy):
        return torch.mm(y, true_dy)
