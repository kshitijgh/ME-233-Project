import torch
import torch.optim as optim
from model import MyNetwork
from data import *
from torchviz import make_dot
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import math
import wandb
import random


np.random.seed(0)

combined_dataset = TensorDataset(input, output)
total_samples = len(combined_dataset)
print("Total samples:", total_samples)

# Set the split ratio
split_ratio = 0.8
split_index = int(total_samples * split_ratio)

# Create indices for shuffling
indices = np.arange(total_samples)
np.random.shuffle(indices)
print("indices = ", indices)

# Split the indices into training and testing indices
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Create training and testing datasets based on the shuffled indices
train_dataset = TensorDataset(input[train_indices], output[train_indices])
test_dataset = TensorDataset(input[test_indices], output[test_indices])

print("Shape of the training dataset:", len(train_dataset))
print("Shape of the testing dataset:", len(test_dataset))


def training_loop(config=None):

    # tell wandb to get started
    with wandb.init(project="Sparse Regression 1", config=config):
        config = wandb.config
        batch_size = config.batch_size

        run_seed = 42 + config.seed  # Calculate a unique seed for each run
        torch.manual_seed(run_seed)
        random.seed(run_seed)

        # Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = MyNetwork(run_seed)
        criterion = torch.nn.MSELoss()
        # wandb.watch(model, criterion, log="all", log_freq=10)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay = 0)

        # Function to calculate L1 regularization loss
        def l1_regularization(model):
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.abs(param).sum()
            return config.reg_factor * l1_loss
        
        print("Hi")

        num_epochs = config.epochs
        losses = []  # List to store losses
        mse_loss = []
        loss_reg = []

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_mse_loss = 0.0
            total_reg_loss = 0.0

            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(torch.float32)
                labels = labels.to(torch.float32)

                optimizer.zero_grad()
                # print("inputs = ", inputs)
                outputs = model(inputs)
                # print("outputs = ", outputs)
                loss_0 = criterion(outputs,labels)


                loss = loss_0 + 0*l1_regularization(model)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_mse_loss += loss_0.item()
                total_reg_loss += l1_regularization(model)

            # Calculate the average loss for the epoch
            average_loss = math.sqrt(total_loss / (i+1))
            average_mse_loss = math.sqrt(total_mse_loss / (i+1))
            average_reg_loss = math.sqrt(total_reg_loss / (i+1))

            if math.isnan(average_loss):
                print("Entered break loop")
                break

            if epoch == 10 and average_loss > 100:
                break
            
            wandb.log({"epoch": epoch, "loss": average_loss, "mse_loss": average_mse_loss, "reg_loss": average_reg_loss}, step=epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.8f} mse_loss: {average_mse_loss:.8f} reg_loss: {average_reg_loss:.8f}")
            losses.append(average_loss)
            mse_loss.append(average_mse_loss)
            loss_reg.append(average_reg_loss)


        print("Training finished")
        torch.save(model.state_dict(), 'model_weights.pth')
        testing_loop(model, test_dataloader, config)

        for key, value in model.state_dict().items():
            if 'weight' in key:
                print(key)
                print(value)

            if 'bias' in key:
                print(key)
                print(value)

        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training Losses vs. Epoch")

        # Plot losses vs. epoch
        axs[0, 0].plot(range(1, num_epochs + 1), losses)
        axs[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Total Loss")

        # Plot loss_rho vs. epoch
        axs[0, 1].plot(range(1, num_epochs + 1), loss_reg)
        axs[0, 1].set(xlabel="Epoch", ylabel="Loss", title="Loss_reg")

        # Plot loss_rho vs. epoch
        axs[0, 1].plot(range(1, num_epochs + 1), mse_loss)
        axs[0, 1].set(xlabel="Epoch", ylabel="Loss", title="mse_loss")

        # Adjust subplot layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Show the plots
        plt.show()





def testing_loop(model, dataloader, config):
    model.eval()  # Set the model to evaluation mode
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    num_samples = 0
    j = 0

    with torch.no_grad():  # Disable gradient computation during testing
        for data in dataloader:
            # print("Entering testing loop")
            inputs, labels = data

            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)

            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_samples += len(inputs)
            j=+1


    average_loss = math.sqrt(total_loss / (j+1))
    wandb.log({"Testing Loss": average_loss})
    print(f"Testing Loss: {average_loss:.8f}")

    model.train()  # Set the model back to training mode

    return average_loss


