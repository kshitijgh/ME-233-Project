from model import *
from data import *
from utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import numpy as np
import wandb
from hyperparameters import *

sweep_config = {
    'method': 'grid'
    }

sweep_config['parameters'] = hp_dict
sweep_id = wandb.sweep(sweep_config, project="Sparse Regression 7 - 42 functions - Normalized")

wandb.agent(sweep_id, training_loop)





#training_loop(model, train_dataloader, config)
# torch.save(model.state_dict(), 'model_weights.pth')
# wandb.save("model_weights.pth")










