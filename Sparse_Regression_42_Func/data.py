from CoolProp.CoolProp import PropsSI
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model import *


delta_values = np.linspace(0.5, 1.5, 100)
tau_values = np.linspace(304/500, 304/280, 100)

# Create DataFrames for P and T values
df_d = pd.DataFrame({'d': delta_values})
df_t = pd.DataFrame({'t': tau_values})

# Create an index column for both DataFrames to ensure a Cartesian product
df_d['key'] = 0
df_t['key'] = 0

# Perform an outer join to get all combinations of P and T values
df = pd.merge(df_d, df_t, on='key', how='outer').drop(columns=['key'])
df = df[['d', 't']]

# Convert the DataFrame to a NumPy array
input = df.to_numpy()
input = torch.tensor(input)

# print("input tensor = ", input.shape)

a, b = input.shape
output = torch.zeros([a,1])

for i in range(a):
    output[i,0] = PropsSI('alphar', 'T', 304/df.iloc[i, 1], 'D', df.iloc[i, 0]*467, 'CO2')


input = Data_PreProcess(input)

scaler2 = MinMaxScaler()
output_np = output.numpy()
output_scaled = torch.tensor(scaler2.fit_transform(output_np))
output = output_scaled + abs(output_scaled.min())

print("After normalization: input = ", input.shape)
print()
print("After normalization: output = ", output)







