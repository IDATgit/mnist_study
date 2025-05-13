import torch
from torchviz import make_dot
import os
from pathlib import Path
from graphviz import Digraph
from collections import OrderedDict

def get_layer_info(layer):
    """Get the number of parameters in a layer."""
    params = sum(p.numel() for p in layer.parameters())
    return params

def plot_model_architecture(model, input_shape=(1, 1, 28, 28), save_path=None):
    """
    Create a cleaner visualization of a PyTorch model's architecture.
    
    Args:
        model (torch.nn.Module): The PyTorch model to visualize
        input_shape (tuple): Shape of the input tensor (default: MNIST-like input)
        save_path (str, optional): Path to save the visualization. If None, saves to 'outputs/model_architecture.svg'
    
    Returns:
        None: Saves the visualization as an SVG file
    """
    if save_path is None:
        outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        save_path = os.path.join(outputs_dir, 'model_architecture')
    elif save_path.endswith('.svg'):
        save_path = save_path[:-4]

    # Create a new directed graph
    dot = Digraph(comment='Model Architecture')
    dot.attr(rankdir='LR')  # Left to right layout
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Set graph attributes for better layout
    dot.attr('graph', nodesep='0.5', ranksep='0.5')
    
    # Add input node
    dot.node('input', f'Input\\nShape: {input_shape}', fillcolor='lightblue')
    
    # Dictionary to store tensor shapes after each layer
    tensor_shapes = {}
    
    # Manually track the model architecture for SmallConvNet
    nodes = []
    edges = []
    
    # Input shape
    current_shape = list(input_shape)
    
    # Add first node
    nodes.append(('input', f'Input\\nShape: {tuple(current_shape)}', 'lightblue'))
    
    # Process first Conv layer
    in_channels = current_shape[1]
    out_channels = model.conv_layers[0].out_channels
    kernel_size = model.conv_layers[0].kernel_size[0]
    padding = model.conv_layers[0].padding[0]
    params = sum(p.numel() for p in model.conv_layers[0].parameters())
    
    # Create Conv with ReLU
    nodes.append(('conv1', f'Conv2d\\nch_in: {in_channels}, ch_out: {out_channels}\\nkernel: ({kernel_size}, {kernel_size})\\nParams: {params}\\n+ReLU', 'lightblue'))
    edges.append(('input', 'conv1', f'{tuple(current_shape)}'))
    
    # Update shape: Conv changes channels
    current_shape[1] = out_channels
    
    # First BatchNorm
    bn_params = sum(p.numel() for p in model.batch_norms[0].parameters())
    nodes.append(('bn1', f'BatchNorm2d\\nchannels: {out_channels}\\nParams: {bn_params}', 'lightgray'))
    edges.append(('conv1', 'bn1', f'{tuple(current_shape)}'))
    
    # MaxPool after first block
    pool_size = model.pool_sizes[0]
    nodes.append(('pool1', f'MaxPool2d\\nkernel: ({pool_size}, {pool_size})', 'lightyellow'))
    edges.append(('bn1', 'pool1', f'{tuple(current_shape)}'))
    
    # Update shape: spatial dimensions reduce by half with pool_size=2
    current_shape[2] = current_shape[2] // pool_size
    current_shape[3] = current_shape[3] // pool_size
    
    # Process second Conv layer
    in_channels = current_shape[1]
    out_channels = model.conv_layers[1].out_channels
    kernel_size = model.conv_layers[1].kernel_size[0]
    padding = model.conv_layers[1].padding[0]
    params = sum(p.numel() for p in model.conv_layers[1].parameters())
    
    # Create Conv with ReLU
    nodes.append(('conv2', f'Conv2d\\nch_in: {in_channels}, ch_out: {out_channels}\\nkernel: ({kernel_size}, {kernel_size})\\nParams: {params}\\n+ReLU', 'lightblue'))
    edges.append(('pool1', 'conv2', f'{tuple(current_shape)}'))
    
    # Update shape: Conv changes channels
    current_shape[1] = out_channels
    
    # Second BatchNorm
    bn_params = sum(p.numel() for p in model.batch_norms[1].parameters())
    nodes.append(('bn2', f'BatchNorm2d\\nchannels: {out_channels}\\nParams: {bn_params}', 'lightgray'))
    edges.append(('conv2', 'bn2', f'{tuple(current_shape)}'))
    
    # MaxPool after second block
    pool_size = model.pool_sizes[1]
    nodes.append(('pool2', f'MaxPool2d\\nkernel: ({pool_size}, {pool_size})', 'lightyellow'))
    edges.append(('bn2', 'pool2', f'{tuple(current_shape)}'))
    
    # Update shape: spatial dimensions reduce by half with pool_size=2
    current_shape[2] = current_shape[2] // pool_size
    current_shape[3] = current_shape[3] // pool_size
    
    # Flatten operation
    nodes.append(('flatten', 'Flatten', 'white'))
    edges.append(('pool2', 'flatten', f'{tuple(current_shape)}'))
    
    flat_shape = [current_shape[0], current_shape[1] * current_shape[2] * current_shape[3]]
    
    # First FC layer with ReLU
    in_features = flat_shape[1]
    out_features = model.fc_layers[0].out_features
    params = sum(p.numel() for p in model.fc_layers[0].parameters())
    
    nodes.append(('fc1', f'Linear\\nin: {in_features}, out: {out_features}\\nParams: {params}\\n+ReLU', 'lightgreen'))
    edges.append(('flatten', 'fc1', f'{tuple(flat_shape)}'))
    
    # Update shape for Linear output
    fc_out_shape = [flat_shape[0], out_features]
    
    # Dropout (only if rate > 0)
    dropout_rate = model.dropout_rate
    prev_layer = 'fc1'
    
    if dropout_rate > 0:
        nodes.append(('dropout', f'Dropout\\nrate: {dropout_rate}', 'white'))
        edges.append((prev_layer, 'dropout', f'{tuple(fc_out_shape)}'))
        prev_layer = 'dropout'
    
    # Output FC layer
    in_features = out_features
    out_features = model.fc_layers[1].out_features
    params = sum(p.numel() for p in model.fc_layers[1].parameters())
    
    nodes.append(('fc2', f'Linear\\nin: {in_features}, out: {out_features}\\nParams: {params}', 'lightgreen'))
    edges.append((prev_layer, 'fc2', f'{tuple(fc_out_shape)}'))
    
    # Final output shape
    final_shape = [fc_out_shape[0], out_features]
    
    # Add output node
    nodes.append(('output', f'Output\\nShape: {tuple(final_shape)}', 'lightblue'))
    edges.append(('fc2', 'output', f'{tuple(final_shape)}'))
    
    # Add all nodes to the graph
    for name, label, color in nodes:
        dot.node(name, label, fillcolor=color)
    
    # Add all edges to the graph
    for src, dst, label in edges:
        dot.edge(src, dst, label=label)
        
    # Add constraints to keep nodes aligned
    dot.attr('edge', style='invis', weight='100')
    # These invisible edges help maintain alignment
    dot.edge('input', 'bn1', constraint='false')
    dot.edge('conv1', 'pool1', constraint='false')
    dot.edge('bn1', 'conv2', constraint='false')
    dot.edge('pool1', 'bn2', constraint='false')
    dot.edge('conv2', 'pool2', constraint='false')
    dot.edge('bn2', 'flatten', constraint='false')
    dot.edge('pool2', 'fc1', constraint='false')
    dot.attr('edge', style='solid', weight='1')
    
    print(f"Attempting to save architecture visualization to: {save_path}.svg")
    
    try:
        dot.render(save_path, format='svg', cleanup=True)
        print(f"Model architecture successfully saved to {save_path}.svg")
    except Exception as e:
        print(f"Error saving SVG visualization: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Target save directory exists: {os.path.exists(os.path.dirname(save_path))}")
        print(f"Target save directory is writable: {os.access(os.path.dirname(save_path), os.W_OK)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Graphviz is installed on your system")
        print("2. Verify Graphviz is in your system PATH")
        print("3. Try running 'dot -V' in terminal to check installation")

def plot_model(model, input_shape=(1, 1, 28, 28), save_path=None):
    """
    Create both detailed and architectural visualizations of a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model to visualize
        input_shape (tuple): Shape of the input tensor (default: MNIST-like input)
        save_path (str, optional): Path to save the visualization. If None, saves to 'outputs/model_*'
    """
    # Create the detailed computational graph
    if save_path is None:
        detailed_save_path = os.path.join(os.path.dirname(__file__), 'outputs', 'model_detailed')
    else:
        detailed_save_path = save_path.replace('.svg', '_detailed')
    
    # Create the architectural visualization
    if save_path is None:
        arch_save_path = os.path.join(os.path.dirname(__file__), 'outputs', 'model_architecture')
    else:
        arch_save_path = save_path.replace('.svg', '_architecture')
    
    plot_model_architecture(model, input_shape, arch_save_path)

if __name__ == "__main__":
    import os
    import sys
    # Add the project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(project_root)
    
    from models.specific_models.SmallConvNet import SmallConvNet
    plot_model(SmallConvNet())