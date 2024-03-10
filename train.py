import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from model_34 import *

# Define hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 100
regularization_term = 0.5

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, file_path):
        # Load data from file
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        self.inputs = data[:2000, 1:]  # Delta and Tau columns
        self.targets = np.loadtxt('output.txt')[:2000]

        # # Standardize inputs
        # self.scaler = StandardScaler()
        # self.inputs = self.scaler.fit_transform(self.inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# Custom loss function with L1 regularization
class CustomLoss(nn.Module):
    def __init__(self, model, regularization_term):
        super(CustomLoss, self).__init__()
        self.model = model
        self.regularization_term = regularization_term

    def forward(self, output, target):
        mse_loss = nn.functional.mse_loss(output, target)
        l1_loss = torch.tensor(0.0)
        for param in self.model.parameters():
            l1_loss += torch.norm(param, p=1)
        total_loss = mse_loss + self.regularization_term * l1_loss
        return total_loss

# Initialize dataset and dataloader
dataset = CustomDataset('data.csv')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = MyNetwork(seed=42)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = CustomLoss(model, regularization_term)

# Training loop
# for epoch in range(num_epochs):
#     epoch_loss = 0.0
#     for inputs, targets in dataloader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs.squeeze(), targets)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item() * inputs.size(0)
#     epoch_loss /= len(dataset)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# # Save trained model
# torch.save(model.state_dict(), '2000_reg_5e-1.pth')

# Load trained model
model.load_state_dict(torch.load('2000_reg_5e-1.pth'))
print("regularization = 5e-1")
# Display model weights
for name, param in model.named_parameters():
    print(f"Layer: {name}, Weights: {param.data}")