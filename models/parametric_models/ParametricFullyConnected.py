import torch
import torch.nn as nn

class ParametricFullyConnected(nn.Module):
    """
    A flexible fully connected neural network that can handle variable depth and different activation functions.
    
    Args:
        dims (list): List of integers representing the dimensions of each layer.
                    First element is input dim, last element is output dim.
                    Example: [784, 128, 64, 10] creates a network with 2 hidden layers.
        activation (str or callable, optional): Activation function to use between layers.
                                             Can be a string ('relu', 'tanh', etc.) or a callable.
                                             Defaults to 'relu'.
    """
    def __init__(self, layer_sizes, activation='relu'):
        super(ParametricFullyConnected, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("dims must contain at least 2 dimensions (input and output)")
        
        # Dictionary of available activation functions
        self.available_activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        
        # Set activation function
        if isinstance(activation, str):
            activation = activation.lower()
            if activation not in self.available_activations:
                raise ValueError(f"Unsupported activation function: {activation}. "
                              f"Available options are: {list(self.available_activations.keys())}")
            self.activation = self.available_activations[activation]
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError("activation must be a string or callable")
        
        # Build layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Add linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add activation function after all but the last layer
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Flatten input if it's not already flat
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        return self.model(x)
    
    def get_num_parameters(self):
        """
        Calculate the total number of trainable parameters in the network.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Example usage:
if __name__ == "__main__":
    # Example 1: Simple network for MNIST (784 -> 128 -> 10)
    model1 = ParametricFullyConnected([784, 128, 10])
    print(f"Model 1 parameters: {model1.get_num_parameters()}")
    print(f"Model 1 architecture: {model1}")
    
    # Example 2: Deeper network with custom dimensions and tanh activation
    model2 = ParametricFullyConnected([784, 512, 256, 128, 64, 10], activation='tanh')
    print(f"Model 2 parameters: {model2.get_num_parameters()}")
    print(f"Model 2 architecture: {model2}")
    # Test with dummy input
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)  # MNIST-like input
    output = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 