import torch
import os
import sys
import importlib
from pathlib import Path
from typing import Tuple, Optional, Union

def load_model_from_trainer(
    trainer_module_path: str,
    checkpoint_name: str = 'model_latest.pt',
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, str, any]:
    """
    Load a model from a specific trainer module and its latest checkpoint.
    
    Args:
        trainer_module_path (str): Path to the trainer module in dot notation 
                                  (e.g., 'trainers.specific_trainers.train_linear_random_images')
        checkpoint_name (str): Name of the checkpoint file to load, defaults to 'model_latest.pt'
        device (str, optional): Device to load the model to ('cuda' or 'cpu'). 
                               If None, will use CUDA if available.
                               
    Returns:
        Tuple containing:
            - model (nn.Module): The loaded PyTorch model
            - model_name (str): Name of the model
            - data_loader: Data loader associated with the model
    
    Raises:
        ImportError: If the trainer module cannot be imported
        FileNotFoundError: If no checkpoints are found
    """
    # Add the project root to the path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Dynamically import the trainer module
    try:
        trainer_module = importlib.import_module(trainer_module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import trainer module '{trainer_module_path}': {e}")
    
    # Get model, model_name, and data_loader from the trainer module
    model = trainer_module.model
    model_name = trainer_module.model_name
    data_loader = trainer_module.data_loader
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Load latest checkpoint
    checkpoint_dir = Path('trainers') / 'outputs' / model_name / 'checkpoints'
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    if not checkpoint_path.exists():
        # Fall back to looking for any checkpoint files
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Model loaded on {device}")
    
    return model, model_name, data_loader

def load_best_model(trainer_module_path: str, device: Optional[str] = None) -> Tuple[torch.nn.Module, str, any]:
    """
    Load the best model from a specific trainer module.
    
    Args:
        trainer_module_path (str): Path to the trainer module in dot notation
        device (str, optional): Device to load the model to
        
    Returns:
        Tuple containing model, model_name, and data_loader
    """
    return load_model_from_trainer(trainer_module_path, 'model_best.pt', device)

def load_model_at_epoch(trainer_module_path: str, epoch: int, device: Optional[str] = None) -> Tuple[torch.nn.Module, str, any]:
    """
    Load a model from a specific epoch.
    
    Args:
        trainer_module_path (str): Path to the trainer module in dot notation
        epoch (int): Epoch number to load
        device (str, optional): Device to load the model to
        
    Returns:
        Tuple containing model, model_name, and data_loader
    """
    checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
    return load_model_from_trainer(trainer_module_path, checkpoint_name, device)

def load_model_interactive(device: Optional[str] = None) -> Tuple[torch.nn.Module, str, any]:
    """
    Interactively list available trained models, prompt user to select one, and load it.
    
    Args:
        device (str, optional): Device to load the model to ('cuda' or 'cpu')
        
    Returns:
        Tuple containing model, model_name, and data_loader
    """
    # Get the trainer modules directory
    outputs_dir = Path('trainers') / 'outputs'
    
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    
    # Find all model directories with checkpoints
    available_models = []
    for model_dir in outputs_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        checkpoint_dir = model_dir / 'checkpoints'
        if not checkpoint_dir.exists():
            continue
            
        # Check if there are any checkpoint files
        checkpoint_files = list(checkpoint_dir.glob('*.pt'))
        if checkpoint_files:
            available_models.append(model_dir.name)
    
    if not available_models:
        raise FileNotFoundError("No trained models with checkpoints found")
    
    # Sort models alphabetically
    available_models.sort()
    
    # Print available models
    print("\nAvailable trained models:")
    for i, model_name in enumerate(available_models):
        print(f"[{i}] {model_name}")
    
    # Prompt user to select a model
    while True:
        try:
            selection = input("\nEnter the number of the model to load: ")
            index = int(selection)
            if 0 <= index < len(available_models):
                selected_model = available_models[index]
                break
            else:
                print(f"Invalid selection. Please enter a number between 0 and {len(available_models)-1}.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"Selected model: {selected_model}")
    
    # Get the checkpoint dir for the selected model
    checkpoint_dir = outputs_dir / selected_model / 'checkpoints'
    
    # Find available checkpoints
    all_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    
    # Extract the available epoch numbers
    epoch_numbers = []
    for checkpoint in all_checkpoints:
        try:
            # Extract epoch number from filename
            filename = checkpoint.name
            epoch_str = filename.replace('checkpoint_epoch_', '').replace('.pt', '')
            epoch_numbers.append(int(epoch_str))
        except (ValueError, IndexError):
            continue
    
    # Determine available options
    has_best = (checkpoint_dir / 'model_best.pt').exists()
    has_latest = (checkpoint_dir / 'model_latest.pt').exists()
    has_epochs = len(epoch_numbers) > 0
    
    print("\nAvailable checkpoint options:")
    print("[1] Latest checkpoint" + (" (available)" if has_latest else " (not available)"))
    print("[2] Best checkpoint" + (" (available)" if has_best else " (not available)"))
    
    if has_epochs:
        min_epoch = min(epoch_numbers)
        max_epoch = max(epoch_numbers)
        print(f"[3] Specific epoch (available range: {min_epoch}-{max_epoch})")
    else:
        print("[3] Specific epoch (no epoch checkpoints available)")
    
    # Prompt user to select checkpoint type
    checkpoint_name = None
    while checkpoint_name is None:
        try:
            selection = input("\nEnter the number of the checkpoint type to load: ")
            checkpoint_type = int(selection)
            
            if checkpoint_type == 1:  # Latest
                if has_latest:
                    checkpoint_name = 'model_latest.pt'
                else:
                    print("Latest checkpoint not available, please select another option.")
            
            elif checkpoint_type == 2:  # Best
                if has_best:
                    checkpoint_name = 'model_best.pt'
                else:
                    print("Best checkpoint not available, please select another option.")
            
            elif checkpoint_type == 3:  # Specific epoch
                if has_epochs:
                    while True:
                        try:
                            epoch_input = input(f"\nEnter epoch number ({min_epoch}-{max_epoch}): ")
                            epoch = int(epoch_input)
                            
                            if min_epoch <= epoch <= max_epoch and epoch in epoch_numbers:
                                checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
                                break
                            else:
                                print(f"Invalid epoch. Please enter a number between {min_epoch} and {max_epoch} that exists.")
                        except ValueError:
                            print("Please enter a valid number.")
                else:
                    print("No epoch checkpoints available, please select another option.")
            
            else:
                print("Invalid selection. Please enter 1, 2, or 3.")
                
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"Selected checkpoint: {checkpoint_name}")
    
    # Create a mapping between model names and trainer modules (without importing them)
    # This is done statically based on our trainer file naming conventions
    model_to_trainer_mapping = {
        'LinearModel': 'trainers.specific_trainers.train_linear',
        'LinearModel_RandomLabels': 'trainers.specific_trainers.train_linear_random_labels',
        'LinearModel_RandomImages': 'trainers.specific_trainers.train_linear_random_images',
        'StandardFullyConnected': 'trainers.specific_trainers.fully_connected_trainer',
        'StandardFullyConnected_RandomLabels': 'trainers.specific_trainers.train_fully_connected_random_labels',
        'SmallFullyConnected': 'trainers.specific_trainers.train_small_fully_connected',
        'SmallFullyConnected_RandomLabels': 'trainers.specific_trainers.train_small_fully_connected_random_labels',
        'ResNet18': 'trainers.specific_trainers.resnet_trainer',
        'ResNet18_RandomLabels': 'trainers.specific_trainers.train_resnet_random_labels',
        'ResNet18_RandomImages': 'trainers.specific_trainers.train_resnet_random_images',
        'ShiftInvariantCNN': 'trainers.specific_trainers.train_shift_invariant',
        'ConvNet': 'trainers.specific_trainers.conv_trainer',
        'BranchingMergingNet': 'trainers.specific_trainers.branching_merging_trainer'
    }
    
    # Check if we have a mapping for this model
    if selected_model in model_to_trainer_mapping:
        trainer_module = model_to_trainer_mapping[selected_model]
        print(f"Found matching trainer module: {trainer_module}")
        return load_model_from_trainer(trainer_module, checkpoint_name, device)
    else:
        print(f"No specific trainer module found for {selected_model}.")
        print("Loading model directly from saved checkpoints...")
        
        # Load the model directly from checkpoints
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu' if device is None else device)
        
        # Check if model class info is saved in the checkpoint
        if 'model_class' in checkpoint:
            model_class = checkpoint['model_class']
            model = model_class()
        else:
            # Without model class info, we need user to specify the model type
            print("\nModel class information not found in checkpoint.")
            print("Please select model type:")
            print("[0] LinearModel")
            print("[1] StandardFullyConnected")
            print("[2] ShiftInvariantCNN")
            print("[3] ResNet")
            
            model_types = [
                ("models.specific_models.LinearModel", "LinearModel"),
                ("models.specific_models.StandardFullyConnected", "StandardFullyConnected"),
                ("models.specific_models.ShiftInvariantCNN", "ShiftInvariantCNN"),
                ("models.specific_models.ResNet", "ResNet")
            ]
            
            while True:
                try:
                    selection = input("\nEnter the number of the model type: ")
                    index = int(selection)
                    if 0 <= index < len(model_types):
                        model_module_path, model_class_name = model_types[index]
                        
                        # Import the model class and instantiate it
                        model_module = importlib.import_module(model_module_path)
                        model_class = getattr(model_module, model_class_name)
                        model = model_class()
                        break
                    else:
                        print(f"Invalid selection. Please enter a number between 0 and {len(model_types)-1}.")
                except (ValueError, ImportError, AttributeError) as e:
                    print(f"Error: {e}. Please try again.")
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Determine device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move model to correct device
        model = model.to(device)
        
        # Create a basic data loader
        from utils.data_loader import MNISTDataLoader
        data_loader = MNISTDataLoader(
            batch_size=64,
            preload_gpu=(device == 'cuda')
        )
        
        print(f"Model loaded successfully on {device}")
        return model, selected_model, data_loader

# Example usage in main section
if __name__ == "__main__":
    # Example: Load latest model
    model, model_name, data_loader = load_model_from_trainer('trainers.specific_trainers.train_linear_random_images')
    
    # Example: Load best model
    # model, model_name, data_loader = load_best_model('trainers.specific_trainers.train_linear_random_images')
    
    # Example: Load model at specific epoch
    # model, model_name, data_loader = load_model_at_epoch('trainers.specific_trainers.train_linear_random_images', epoch=10)
    
    # Example: Load model interactively
    # model, model_name, data_loader = load_model_interactive()
    
    # Print model architecture
    print(f"Loaded model: {model_name}")
    print(model) 