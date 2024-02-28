from model import *
from data import *
# from utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import numpy as np
import wandb
from hyperparameters import *
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import OrthogonalMatchingPursuit
import torch.nn.functional as F
from scipy.optimize import minimize

model = MyNetwork(0)

# print()
# print(output)
# print()

# result = np.matmul(model(input), coefficients)
# print(result)
# print()

# print()

# # Calculate MSE loss
# mse_loss = F.mse_loss(output, result)

# print("MSE Loss:", mse_loss.item())
input_layer = model(input)
input_layer = input_layer.numpy()
output = output.numpy()
# print(input_layer.shape)

# ATA = input_layer.T @ input_layer
# ATb = input_layer.T @ output

# print(np.linalg.solve(ATA, ATb))
# print(output)

A = input_layer.copy()
b = output.copy()

def least_norm_solution(A, b, norm_constraint_value):
    # Define the objective function for minimization
    def objective_function(x):
        return np.linalg.norm(x)

    # Define the equality constraint: Ax - b = 0 and norm(x) - norm_constraint_value = 0
    def equality_constraint(x):
        return np.concatenate((np.dot(A, x) - b, np.array([np.linalg.norm(x) - norm_constraint_value])))

    # Initial guess for x
    x0 = np.ones(A.shape[1])

    # Constraint options
    constraint = {'type': 'eq', 'fun': equality_constraint}

    # Solve the constrained optimization problem
    result = minimize(objective_function, x0, constraints=constraint)

    # Extract the solution
    x_solution = result.x

    return x_solution

# Example usage:
# Generate random matrix A and vector b for demonstration
# A = np.random.rand(10000, 42)
# b = np.random.rand(10000, 1)
norm_constraint_value = 3

# Solve the least-norm problem with norm constraint
solution = least_norm_solution(A, b, norm_constraint_value)

# Display the solution
print("Solution x:", solution)
