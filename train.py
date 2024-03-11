import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from model_34 import *
from data_hw import input, output
from sklearn.model_selection import train_test_split

# Split dataset into training, validation, and test sets (70-20-10)
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)

# Define hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 100
regularization_term = 0.0001

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        # # Load data from file
        # data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        # self.inputs = data[:2000, 1:]  # Delta and Tau columns
        # self.targets = np.loadtxt('output.txt')[:2000]

        # # # Standardize inputs
        # # self.scaler = StandardScaler()
        # # self.inputs = self.scaler.fit_transform(self.inputs)
        self.inputs = input_data
        self.targets = output_data

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
train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(x_train.shape)
print(x_test.shape)
# Initialize model
model = MyNetwork(seed=42)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = CustomLoss(model, regularization_term)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * inputs.size(0)
    epoch_train_loss /= len(train_dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}")

# Test loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        test_loss += loss.item() * inputs.size(0)
test_loss /= len(test_dataset)
print(f"Test Loss: {test_loss:.4f}")


# # Save trained model
# torch.save(model.state_dict(), '2000_reg_5e-1.pth')

# Load trained model
# model.load_state_dict(torch.load('2000_reg_5e-1.pth'))
# print("regularization = 5e-1")
# # Display model weights
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, Weights: {param.data}")