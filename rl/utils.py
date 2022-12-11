import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state(obs):
    return np.transpose(obs, (2, 0, 1))
