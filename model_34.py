import torch
import torch.nn as nn
import sympy as sp
import random
import torch.optim as optim
# from data import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# seed = 42 + 0
# torch.manual_seed(seed)
# random.seed(seed)


def Data_PreProcess(input):

    delta = input[:, 0]
    tau = input[:, 1]

    neurons = []
    
    d = torch.zeros([34,1])
    t = torch.zeros([34,1])
    c = torch.zeros([34,1])


    d[0:7,0] = torch.tensor([1, 1, 1, 1, 2, 2, 3], dtype=torch.float32)
    d[7:34,0] = torch.tensor([1, 2, 4, 5, 5, 5, 6, 6, 6, 1, 1, 4, 4, 4, 7, 8, 2, 3, 3, 5, 5, 6, 7, 8, 10, 4, 8], dtype=torch.float32)

    t[0:7,0] = torch.tensor([0.00, 0.75, 1.00, 2.00, 0.75, 2.00, 0.75], dtype=torch.float32)
    t[7:34,0] = torch.tensor([1.50, 1.50, 2.50, 0.00, 1.50, 2.00, 0.00, 1.00, 2.00, 3.00, 6.00, 3.00, 6.00, 8.00, 6.00, 0.00, 7.00, 12.00, 16.00, 22.00, 24.00, 16.00, 24.00, 8.00, 2.00, 28.00, 14.00], dtype=torch.float32)

    c[7:34,0] = torch.tensor([1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,4,4,4,4,4,4,5,6], dtype=torch.float32)

 
    for i in range(0,7):
        neuron = (delta**d[i,0])*(tau**t[i,0])
        neurons.append(neuron)

    for i in range(7, 34):
        neuron = (delta**d[i,0])*(tau**t[i,0])*torch.exp(-1*(delta**c[i,0]))
        neurons.append(neuron)


    x = torch.stack(neurons, dim=1)
    variable_size = x.size(0)
    x = x.view(variable_size, 34)

    return x

class MyNetwork(nn.Module):
    def __init__(self, seed=None):
        super(MyNetwork, self).__init__()
        self.input_layer = nn.Linear(34, 1, bias=False)
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.input_layer.weight)
        
    def forward(self, input):
        # x = self.batch_norm_input(input)
        x = Data_PreProcess(input)
        x = self.input_layer(x.float())
        return x
