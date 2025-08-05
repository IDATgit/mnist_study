import torch
import torch.nn as nn

class LinearModelOverParametrized(nn.Module):
    """
    A simple linear model for MNIST classification.
    Architecture:
    1. Flatten 28x28 input to 784 features
    2. Single linear layer from 784 to 10 outputs
    """
    def __init__(self):
        # Input size: 28x28 = 784
        super(LinearModelOverParametrized, self).__init__()
        
        # Output size: 10 (digits 0-9)
        self.linear1 = nn.Linear(784, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 1000)
        self.linear4 = nn.Linear(1000, 1000)
        self.linear5 = nn.Linear(1000, 10)
    
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear3.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear4.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear5.weight, gain=0.1)
        # Initialize bias with zeros
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        nn.init.zeros_(self.linear3.bias)
        nn.init.zeros_(self.linear4.bias)
        nn.init.zeros_(self.linear5.bias)
    
    def forward(self, x):
        # Flatten input: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(x.size(0), -1)
        # Apply linear layer
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x

    def get_name(self):
        return f"LinearModelOverParametrized"
        
    def get_num_parameters(self):
        """
        Calculate the total number of trainable parameters in the network.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 