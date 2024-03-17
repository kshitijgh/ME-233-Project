import torch
import numpy as np
from data_hw import input, output
from model_34 import *
import os, copy
from sklearn.metrics import r2_score
from pathlib import Path
import pandas as pd


def _terms_extract(model: MyNetwork, terms: int, preds: np.array, output: torch.Tensor) -> tuple[np.array, int]:
    """
    Helper function to actually find the dominant terms and their corresponding indices in the full model activations
    Args:
    model (MyNetwork): The full trained model
    terms (int): Number of terms in the full trained model
    preds (np.array): The array to store the outputs from only one nonzero weight models
    output (torch.Tensor): Ground truth

    Returns:
    indices (np.array): The indices of the dominant terms from the original full model activations
    n (int): Number of dominant terms
    """
    similarity_met = []
    model_dict = model.state_dict()
    model_weights = model_dict['input_layer.weight'] # Extract trained weights from full model

    for i in range(terms):
        
        test_model = MyNetwork(seed=42)
        test_model_dict = copy.deepcopy(model_dict)
        model_plot = torch.zeros_like(model_weights)
        model_plot[0,i] = model_weights[0,i] # Since there is only one layer in this model, use (34,) array to store weights

        # Construct model with only one non-zero weight and evaluate
        test_model_dict['input_layer.weight'] = model_plot
        test_model.load_state_dict(test_model_dict)
        test_model.eval()
        out = test_model(input).flatten()
        preds[:, i] = out.detach().numpy()

        # Calculate similarity through dot product between n_if_i and output
        similarity_met.append(output.dot(preds[:,i])/(np.linalg.norm(preds[:,i])*np.linalg.norm(output)))

    similarity_met = np.array(similarity_met)
    final_nterms = 15 # Expected number of dominant terms

    # Extract dominant terms by comparing similarity value against threshold
    sim_threshold = 0
    n = 0 
    for s in np.arange(0.3,0.95,0.01):
        if similarity_met[np.abs(similarity_met) >= float(s)].shape[0] <= final_nterms:
            sim_threshold = s
            n = similarity_met[np.abs(similarity_met) >= float(s)].shape[0]
            indices = np.argwhere(np.abs(similarity_met) >= float(s))
            indices = indices.reshape((indices.shape[0],))
            break
    
    return indices, n


def find_dominant_terms(dirpath: str, input: torch.Tensor, output: torch.Tensor) -> None:
    """
    Function to determine dominant terms for each saved model, through cosine similarity metric between y_true and y_pred. 
    Outputs the r2 score from the full trained and dominant-only models also.

    Args:
    dirpath (str): Path to the saved models directory
    input (torch.Tensor): Input training data
    output (torch.Tensor): Ground truth corresponding to input

    Returns:
    None
    """
    
    pth_files = [f for f in os.listdir(dirpath) if f[-3:]=='pth']
    output = output.detach().numpy()
    output = output.reshape((output.shape[0],))

    sim_dict = {}

    for pth_file in pth_files:
        
        # Load full model, find predicted output and r2 score
        model = MyNetwork(seed=42)
        if os.path.isfile(Path.joinpath(dirpath, pth_file)):
            model.load_state_dict(torch.load(Path.joinpath(dirpath, pth_file)))
        else:
            print(f"problem at {pth_file}")

        model.eval()
        pred_output = model(input).detach().numpy()
        print(f"r2 score with full trained model: {r2_score(output, pred_output)}")

        terms = 34
        model_dict = model.state_dict()
        model_weights = model_dict['input_layer.weight'] # Extract trained weights from full model
        
        preds = np.zeros((input.shape[0], terms)) # Store predictions from single weight models
        indices, n = _terms_extract(model, terms, preds, output) # Get dominant terms and resulting predictions from only one nonzero weight models

        # Construct a dominant-terms only model and evaluate
        dominant_model_dict = copy.deepcopy(model_dict)
        dominant_model_weights = torch.zeros_like(model_weights)
        dominant_model_weights[0, indices] = model_weights[0, indices]
        dominant_model_dict['input_layer.weight'] = dominant_model_weights

        dominant_model = MyNetwork(seed=42)
        dominant_model.load_state_dict(dominant_model_dict)
        dominant_out = dominant_model(input).detach().numpy()
        
        print(f"r2 score with dominant-only model: {r2_score(output, dominant_out)}")
        print("*"*50)

        sim_dict[f'{pth_file[:-4]}'] = [r2_score(output, pred_output), r2_score(output, dominant_out), n]
    
    df = pd.DataFrame.from_dict(sim_dict, orient='index')
    print(df.head())
    df.to_csv("./output.csv")

# path = path/to/saved_models
dirpath = Path(path)
find_dominant_terms(dirpath, input, output)
