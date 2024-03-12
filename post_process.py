import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from model_34 import *
from data_hw import input, output
import torch


pth_files = ['Lp_1_reg_0.001.pth']  

multiplied_weights = []
function_matrix = Data_PreProcess(input)

print("Func Matrix Shape = ", function_matrix.shape)

# Iterate through each .pth file
for pth_file in pth_files:
    model = MyNetwork(seed=42)
    model.load_state_dict(torch.load(pth_file))
    model.eval()
    model_weights = model.state_dict()
    model_weights_tensors = {key: torch.tensor(value) for key, value in model_weights.items()}
    model_weights_tensor_list = list(model_weights_tensors.values())  # Convert dict values to list of tensors
    multiplied_weights.append(function_matrix * torch.cat(model_weights_tensor_list, dim=1))

    dot_products = torch.sum(multiplied_weights[0] * output, axis = 0).numpy()
    print(np.sort(np.abs(dot_products))[::-1])
    sorted_indices = np.argsort(np.abs(dot_products))[::-1]

    # print("Column indices in decreasing order of absolute dot product magnitude:")
    print(sorted_indices)
    # print(model_weights)




    

