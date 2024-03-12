import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from model_34 import *
from data_hw import input, output
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib  import pyplot as plt

# Split dataset into training, validation, and test sets (70-20-10)
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)

# Define hyperparameters
learning_rate = 1e-4
batch_size = 64
num_epochs = 1500

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
        return self.inputs[idx].clone().detach(), self.targets[idx].clone().detach()


# Initialize dataset and dataloader
train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

p_Norms = [0.5, 0.75, 1]
regularization_terms = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

# Nested loop for hyperparameters search
for Lp in p_Norms:
    for regularization_term in regularization_terms:
        print(f"Training with Lp={Lp} and regularization_term={regularization_term}")

        # Define loss function with current hyperparameters
        class CustomLoss(nn.Module):
            def __init__(self, model):
                super(CustomLoss, self).__init__()
                self.model = model

            def forward(self, output, target):
                mse_loss = nn.functional.mse_loss(output, target.squeeze())
                l1_loss = torch.tensor(0.0)
                for param in self.model.parameters():
                    l1_loss += torch.norm(param, p=Lp)
                total_loss = mse_loss + regularization_term * l1_loss
                return total_loss

        # Initialize model
        model = MyNetwork(seed=42)

        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = CustomLoss(model)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0.0
            y_true_train = []
            y_pred_train = []
            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * inputs.size(0)
                y_true_train.extend(targets.cpu().numpy())
                y_pred_train.extend(outputs.detach().squeeze().cpu().numpy())
            epoch_train_loss /= len(train_dataset)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}")
            r2_train = r2_score(y_true_train, y_pred_train)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train R2 Score: {r2_train:.4f}")

        # Test loop
        model.eval()
        test_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                test_loss += loss.item() * inputs.size(0)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(outputs.squeeze().cpu().numpy())
        test_loss /= len(test_dataset)
        print(f"Test Loss MSE: {test_loss:.4f}")

        # Calculate R2 scores
        r2 = r2_score(y_true, y_pred)
        print(f"R2 Score: {r2:.4f}")

        torch.save(model.state_dict(), f'Lp_' + str(Lp) +  '_reg_' + str(regularization_term) + '.pth')

