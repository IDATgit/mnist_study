import torch
import torch.nn as nn

class LinearModel(nn.Module):
    """
    A simple linear model for MNIST classification.
    Architecture:
    1. Flatten 28x28 input to 784 features
    2. Single linear layer from 784 to 10 outputs
    """
    def __init__(self):
        super(LinearModel, self).__init__()
        
        # Input size: 28x28 = 784
        # Output size: 10 (digits 0-9)
        self.linear = nn.Linear(784, 10)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        # Initialize bias with zeros
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        # Flatten input: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(x.size(0), -1)
        # Apply linear layer
        return self.linear(x) 