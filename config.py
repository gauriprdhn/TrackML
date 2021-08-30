import torch
import numpy as np

root = "/home/gpradhan/trackML/train"
n_events = 100
batch_size: int = 4
validation_split: float = 0.1
pt_min: float = 0.0
shuffle = True
epochs = 10
input_dim: int = 3
output_dim: int = 8
learning_rate: float = 1e-4
decay_rate: float  = 1e-4

