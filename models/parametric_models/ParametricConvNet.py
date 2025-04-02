import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricConvNet(nn.Module):
    def __init__(self, 
                 conv_channels=[32, 64, 64],
                 kernel_sizes=[3, 3, 3],
                 padding=[1, 1, 1],
                 pool_sizes=[2, 2, 1],
                 fc_sizes=[512, 128],
                 dropout_rate=0.5):
        super(ParametricConvNet, self).__init__()
        
        # Store parameters
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.padding = padding
        self.pool_sizes = pool_sizes
        self.fc_sizes = fc_sizes
        self.dropout_rate = dropout_rate
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        in_channels = 1  # Start with 1 channel for MNIST
        
        for i, (out_channels, kernel_size, pad) in enumerate(zip(conv_channels, kernel_sizes, padding)):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad)
            )
            self.batch_norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        
        # Calculate size after convolutions and pooling
        final_size = 28  # Initial size
        for pool_size in pool_sizes:
            final_size = final_size // pool_size
        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = conv_channels[-1] * final_size * final_size
        
        for out_features in fc_sizes:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        
        # Final layer
        self.fc_layers.append(nn.Linear(in_features, 10))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Reshape input
        x = x.view(-1, 1, 28, 28)
        
        # Convolutional layers
        for i, (conv, bn, pool_size) in enumerate(zip(self.conv_layers, self.batch_norms, self.pool_sizes)):
            x = conv(x)
            x = F.relu(x)
            x = bn(x)
            if pool_size > 1:
                x = F.max_pool2d(x, pool_size)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for i, fc in enumerate(self.fc_layers[:-1]):
            x = fc(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.fc_layers[-1](x)
        
        return x
        
    def get_name(self):
        return f"ParametricConvNet_{self.conv_channels}_{self.fc_sizes}"
        
    def get_num_parameters(self):
        """
        Calculate the total number of trainable parameters in the network.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 