import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()

        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        self.net = nn.ModuleList(layer_list)

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = self.activation(self.net[i](x))
        return self.net[-1](x)