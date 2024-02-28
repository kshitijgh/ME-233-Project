import torch
import torch.nn as nn
import sympy as sp
from hyperparameters import *
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
    
    d = torch.zeros([42,1])
    t = torch.zeros([42,1])
    c = torch.zeros([42,1])
    alpha = torch.zeros([42,1])
    beta = torch.zeros([42,1])
    gamma = torch.zeros([42,1])
    epsilon = torch.zeros([42,1])
    a = torch.zeros([42,1])
    b = torch.zeros([42,1])
    A = torch.zeros([42,1])
    B = torch.zeros([42,1])
    C = torch.zeros([42,1])
    D = torch.zeros([42,1])

    d[0:7,0] = torch.tensor([1, 1, 1, 1, 2, 2, 3], dtype=torch.float32)
    d[7:34,0] = torch.tensor([1, 2, 4, 5, 5, 5, 6, 6, 6, 1, 1, 4, 4, 4, 7, 8, 2, 3, 3, 5, 5, 6, 7, 8, 10, 4, 8], dtype=torch.float32)
    d[34:39,0] = torch.tensor([2, 2, 2, 3, 3], dtype=torch.float32)

    t[0:7,0] = torch.tensor([0.00, 0.75, 1.00, 2.00, 0.75, 2.00, 0.75], dtype=torch.float32)
    t[7:34,0] = torch.tensor([1.50, 1.50, 2.50, 0.00, 1.50, 2.00, 0.00, 1.00, 2.00, 3.00, 6.00, 3.00, 6.00, 8.00, 6.00, 0.00, 7.00, 12.00, 16.00, 22.00, 24.00, 16.00, 24.00, 8.00, 2.00, 28.00, 14.00], dtype=torch.float32)
    t[34:39,0] = torch.tensor([1, 0, 1, 3, 3], dtype=torch.float32)

    c[7:34,0] = torch.tensor([1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,4,4,4,4,4,4,5,6], dtype=torch.float32)

    alpha[34:39,0] = torch.tensor([25, 25, 25, 15, 20], dtype=torch.float32)

    beta[34:42, 0] = torch.tensor([325, 300, 300, 275, 275, 0.3, 0.3, 0.3], dtype=torch.float32)

    gamma[34:39,0] = torch.tensor([1.16, 1.19, 1.19, 1.25, 1.22], dtype=torch.float32)

    epsilon[34:39,0] = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)

    a[39:42, 0] = torch.tensor([3.5, 3.5, 3], dtype=torch.float32)
    b[39:42, 0] = torch.tensor([0.875, 0.925, 0.875], dtype=torch.float32)

    A[39:42, 0] = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32)
    B[39:42, 0] = torch.tensor([0.3, 0.3, 1], dtype=torch.float32)
    C[39:42, 0] = torch.tensor([10, 10, 12.5], dtype=torch.float32)
    D[39:42, 0] = torch.tensor([275, 275, 275], dtype=torch.float32)

    for i in range(0,7):
        neuron = (delta**d[i,0])*(tau**t[i,0])
        neurons.append(neuron)

    for i in range(7, 34):
        neuron = (delta**d[i,0])*(tau**t[i,0])*torch.exp(-1*(delta**c[i,0]))
        neurons.append(neuron)

    for i in range(34, 39):
        neuron = (delta**d[i,0])*(tau**t[i,0])*torch.exp(-alpha[i,0]*((delta - epsilon[i,0])**2))*torch.exp(-beta[i,0]*((tau - gamma[i,0])**2))
        neurons.append(neuron)

    for i in range(39, 42):
        tri = ((1-tau) + A[i,0]*((delta-1)**2)**(1/(2*beta[i,0])))**2 + B[i,0]*((delta-1)**2)**a[i,0]
        neuron = (tri**b[i,0])*delta*torch.exp(-C[i,0]*((delta - 1)**2))*torch.exp(-D[i,0]*((tau - 1)**2))
        neurons.append(neuron)

    x = torch.stack(neurons, dim=1)
    variable_size = x.size(0)
    x = x.view(variable_size, 42)
    # scaler1 = MinMaxScaler()
    # x_np = x.numpy()
    # x_scaled = torch.tensor(scaler1.fit_transform(x_np))
    # x = x_scaled + abs(x_scaled.min())

    return x

class MyNetwork(nn.Module):
    def __init__(self, seed=None):
        super(MyNetwork, self).__init__()
        self.batch_norm_input = nn.BatchNorm1d(42)
        self.input_layer = nn.Linear(42, 1, bias=False)
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.input_layer.weight)
        
    def forward(self, input):
        x = self.batch_norm_input(input)
        x = self.input_layer(x)
        return x
    


coefficients = torch.zeros([42,1])
coefficients[0:7,0] = torch.tensor([0.38856823203161, 0.29385475942740*10, -0.55867188534934*10, -0.76753199592477, 0.31729005580416, 0.54803315897767, 0.12279411220335], dtype=torch.float32)

coefficients[7:34,0] = torch.tensor([
    0.216589615432 * 10**1,
    0.158417351097 * 10**1,
    -0.231327054055 * 10**0,
    0.58116916431436 * 10**(-1),
    -0.55369137205382 * 10**0,
    0.48946615909422 * 10**0,
    -0.24275739843501 * 10**(-1),
    0.62494790501678 * 10**(-1),
    -0.121,
    -0.37055685270086 * 10**0,
    -0.16775879700426 * 10**(-1),
    -0.11960736637987 * 10**0,
    -0.45619362508778 * 10**(-1),
    0.35612789270346 * 10**(-1),
    -0.74427727132052 * 10**(-2),
    -0.17395704902432 * 10**(-2),
    -0.21810121289527 * 10**(-1),
    0.24332166559236 * 10**(-1),
    -0.37440133423463 * 10**(-1),
    0.14338715756878 * 10**0,
    -0.13491969083286 * 10**0,
    -0.23151225053480 * 10**(-1),
    0.12363125492901 * 10**(-1),
    0.21058321972940 * 10**(-2),
    -0.33958519026368 * 10**(-3),
    0.55993651771592 * 10**(-2),
    -0.30335118055646 * 10**(-3)
]
, dtype=torch.float32)

coefficients[34:39,0] = torch.tensor([
    -0.213654886883 * 10**3,
    0.266415691492 * 10**5,
    -0.24027212204557 * 10**5,
    -0.28341603423999 * 10**3,
    0.21247284400179 * 10**3
], dtype=torch.float32)

coefficients[39:42,0] = torch.tensor([
    -0.66642276540751 * 10**0,
    0.72608632349897 * 10**0,
    0.55068668612842 * 10**(-1)
], dtype=torch.float32)

