import torch
import torch.nn as nn
import torch.nn.functional as F

class DimensionReductionFC(nn.Module):
    def __init__(self):
        """
        Fully Connected Neural Network with systematic dimension reduction by factor of ~2.
        
        Architecture designed to test the hypothesis that dimensionality bottlenecks
        drive Fisher Information sensitivity patterns.
        
        Layer-by-layer breakdown:
        - Input: 784 (28x28 MNIST)
        - FC1: 784 → 392 (50% reduction)
        - FC2: 392 → 196 (50% reduction) 
        - FC3: 196 → 98 (50% reduction)
        - FC4: 98 → 49 (50% reduction)
        - FC5: 49 → 25 (49% reduction)
        - FC6: 25 → 12 (52% reduction)
        - FC7: 12 → 10 (17% reduction - final classification)
        
        Parameter count breakdown:
        - FC1: 784*392 + 392 = 307,520 + 392 = 307,912 parameters
        - FC2: 392*196 + 196 = 76,832 + 196 = 77,028 parameters
        - FC3: 196*98 + 98 = 19,208 + 98 = 19,306 parameters
        - FC4: 98*49 + 49 = 4,802 + 49 = 4,851 parameters
        - FC5: 49*25 + 25 = 1,225 + 25 = 1,250 parameters
        - FC6: 25*12 + 12 = 300 + 12 = 312 parameters
        - FC7: 12*10 + 10 = 120 + 10 = 130 parameters
        
        Total: ~410,789 parameters
        
        This design will allow us to test which dimension reduction steps
        contribute most to Fisher Information sensitivity.
        """
        super(DimensionReductionFC, self).__init__()
        
        # Define the layer dimensions
        self.layer_dims = [784, 392, 196, 98, 49, 25, 12, 10]
        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.layer_dims) - 1):
            self.fc_layers.append(
                nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            )
        
        # Optional: Add batch normalization for better training
        self.batch_norms = nn.ModuleList()
        for i in range(len(self.layer_dims) - 2):  # No BN on output layer
            self.batch_norms.append(nn.BatchNorm1d(self.layer_dims[i + 1]))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten input
        x = x.view(-1, 784)
        
        # Forward through all layers except the last
        for i, fc in enumerate(self.fc_layers[:-1]):
            x = fc(x)
            x = F.relu(x)
            
            # Apply batch normalization
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Apply dropout (except on last hidden layer to avoid affecting final logits too much)
            if i < len(self.fc_layers) - 2:
                x = self.dropout(x)
        
        # Final layer (no activation, no dropout, no batch norm)
        x = self.fc_layers[-1](x)
        
        return x
    
    def get_name(self):
        return "DimensionReductionFC"
    
    def get_num_parameters(self):
        """
        Calculate the total number of trainable parameters in the network.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self):
        """
        Get detailed information about each layer for analysis.
        
        Returns:
            dict: Layer information including dimensions and parameter counts
        """
        layer_info = {}
        
        for i, fc_layer in enumerate(self.fc_layers):
            in_features = fc_layer.in_features
            out_features = fc_layer.out_features
            num_params = fc_layer.weight.numel() + fc_layer.bias.numel()
            compression_ratio = in_features / out_features if out_features > 0 else 1.0
            
            layer_info[f'fc_{i+1}'] = {
                'input_dim': in_features,
                'output_dim': out_features,
                'num_params': num_params,
                'compression_ratio': compression_ratio,
                'reduction_percentage': (1 - 1/compression_ratio) * 100 if compression_ratio > 1 else 0
            }
        
        return layer_info
    
    def print_architecture(self):
        """
        Print detailed architecture information.
        """
        print(f"DimensionReductionFC Architecture:")
        print("=" * 50)
        
        layer_info = self.get_layer_info()
        total_params = 0
        
        for layer_name, info in layer_info.items():
            print(f"{layer_name}: {info['input_dim']} → {info['output_dim']} "
                  f"({info['reduction_percentage']:.1f}% reduction, {info['num_params']:,} params)")
            total_params += info['num_params']
        
        # Add batch norm and other parameter counts
        bn_params = sum(p.numel() for p in self.batch_norms.parameters())
        total_params += bn_params
        
        print(f"\nBatch Norm parameters: {bn_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Expected total: {self.get_num_parameters():,}")

if __name__ == "__main__":
    # Test the model
    model = DimensionReductionFC()
    model.print_architecture()
    
    # Test forward pass
    test_input = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
    output = model(test_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
