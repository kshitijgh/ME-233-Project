from CoolProp.CoolProp import PropsSI
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# CO2 - Tc is 304 K, Pc is 73.8 bars, rhoc is 467 kg/m3

data_points = 50

delta_values = np.zeros(data_points)

# Pressure values: 100 - 300 bars
P_values = np.linspace(100*(10**5), 300*(10**5), data_points)

# Temperature values: 400 - 700 K
tau_values = np.linspace(304/700, 304/400, data_points)

# Calculating Density
for i in range(data_points):
    delta_values[i] = PropsSI('D', 'T', 304/tau_values[i], 'P', P_values[i], 'CO2')/467

# Now we will perform outer join to combine all possible tau and delta values together in a dataframe: So number of rows will be data_points*data_points

# Create DataFrames for delta and tau values
df_d = pd.DataFrame({'d': delta_values})
df_t = pd.DataFrame({'t': tau_values})

# Create an index column for both DataFrames to ensure a Cartesian product
df_d['key'] = 0
df_t['key'] = 0

# Perform an outer join to get all combinations of P and T values
df = pd.merge(df_d, df_t, on='key', how='outer').drop(columns=['key'])
df = df[['d', 't']]

################################# INPUT VALUES ##########################
print(df)
################################# INPUT VALUES ##########################



# Convert the DataFrame to a NumPy array
input = df.to_numpy()
input = torch.tensor(input)

a, b = input.shape
output = torch.zeros([a,1])


for i in range(a):
    output[i,0] = PropsSI('alphar', 'T', 304/df.iloc[i, 1], 'D', df.iloc[i, 0]*467, 'CO2')


################################# OUTPUT VALUES ##########################
print(output)
################################# OUTPUT VALUES ##########################
    





