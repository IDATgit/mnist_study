import torch
import sys
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import MNISTDataLoader
from models.specific_models.StandardFullyConnected import StandardFullyConnected
from models.specific_models.StandardConvNet import StandardConvNet
from models.specific_models.BranchingMergingCNN import BranchingMergingCNN
from models.specific_models.ResNet import ResNet

class ClassificationStats:
    """Class to compute and store classification statistics."""
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.correct_count = 0
        self.total_count = 0
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    def update(self, predictions, targets):
        """Update statistics with batch results."""
        correct = predictions.eq(targets)
        
        # Update overall statistics
        self.correct_count += correct.sum().item()
        self.total_count += len(targets)
        
        # Update per-class statistics
        for pred, target, is_correct in zip(predictions, targets, correct):
            pred, target = pred.item(), target.item()
            self.class_total[target] += 1
            self.class_correct[target] += is_correct.item()
            self.confusion_matrix[target][pred] += 1
    
    def get_overall_stats(self):
        """Get overall classification statistics."""
        accuracy = 100. * self.correct_count / self.total_count
        error_rate = 100. - accuracy
        return {
            'total_samples': self.total_count,
            'correct': self.correct_count,
            'incorrect': self.total_count - self.correct_count,
            'accuracy': accuracy,
            'error_rate': error_rate
        }
    
    def get_class_stats(self):
        """Get per-class classification statistics."""
        stats = {}
        for class_idx in range(self.num_classes):
            total = self.class_total[class_idx]
            correct = self.class_correct[class_idx]
            if total > 0:
                accuracy = 100. * correct / total
                stats[class_idx] = {
                    'total': total,
                    'correct': correct,
                    'incorrect': total - correct,
                    'accuracy': accuracy,
                    'error_rate': 100. - accuracy
                }
        return stats

def plot_confusion_matrix(confusion_matrix, title, save_path=None):
    """
    Create and optionally save a confusion matrix plot.
    
    Args:
        confusion_matrix (np.ndarray): The confusion matrix to plot
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_model(model, model_name, batch_size=64, device=None, preload_gpu=False, save_plots=True):
    """
    Analyze model's classification performance on both training and test sets.
    
    Args:
        model (nn.Module): The trained model to analyze
        model_name (str): Name of the model for output organization
        batch_size (int): Batch size for data loading
        device (str): Device to run analysis on ('cuda' or 'cpu')
        preload_gpu (bool): Whether to preload data to GPU
        save_plots (bool): Whether to save the confusion matrix plots
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize data loader
    data_loader = MNISTDataLoader(batch_size=batch_size, preload_gpu=preload_gpu)
    
    # Initialize statistics trackers
    train_stats = ClassificationStats()
    test_stats = ClassificationStats()
    
    # Analyze training set
    print("\nAnalyzing training set...")
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader.get_train_loader(), desc='Training Set'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            train_stats.update(predicted, targets)
    
    # Analyze test set
    print("\nAnalyzing test set...")
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader.get_test_loader(), desc='Test Set'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_stats.update(predicted, targets)
    
    # Create output directory for plots if saving
    if save_plots:
        output_dir = os.path.join('model_interpretation', 'outputs', model_name)
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrices
    train_overall = train_stats.get_overall_stats()
    test_overall = test_stats.get_overall_stats()
    
    # Training set confusion matrix
    train_title = f"{model_name} - Training Set Confusion Matrix\nAccuracy: {train_overall['accuracy']:.2f}%"
    if save_plots:
        train_path = os.path.join(output_dir, 'train_confusion_matrix.png')
        plot_confusion_matrix(train_stats.confusion_matrix, train_title, train_path)
    else:
        plot_confusion_matrix(train_stats.confusion_matrix, train_title)
    
    # Test set confusion matrix
    test_title = f"{model_name} - Test Set Confusion Matrix\nAccuracy: {test_overall['accuracy']:.2f}%"
    if save_plots:
        test_path = os.path.join(output_dir, 'test_confusion_matrix.png')
        plot_confusion_matrix(test_stats.confusion_matrix, test_title, test_path)
    else:
        plot_confusion_matrix(test_stats.confusion_matrix, test_title)
    
    return {
        'train': {
            'overall': train_overall,
            'per_class': train_stats.get_class_stats(),
            'confusion_matrix': train_stats.confusion_matrix
        },
        'test': {
            'overall': test_overall,
            'per_class': test_stats.get_class_stats(),
            'confusion_matrix': test_stats.confusion_matrix
        }
    }

def find_available_models():
    """Find all available trained models in the trainers/outputs directory."""
    models = []
    outputs_dir = os.path.join('trainers', 'outputs')
    
    if not os.path.exists(outputs_dir):
        return models
        
    for model_dir in os.listdir(outputs_dir):
        model_path = os.path.join(outputs_dir, model_dir, 'checkpoints', 'model_best.pt')
        if os.path.exists(model_path):
            # Extract model type from directory name
            model_dir_lower = model_dir.lower()
            if 'fully_connected' in model_dir_lower or 'fc' in model_dir_lower:
                model_type = 'fc'
            elif 'branching' in model_dir_lower:
                model_type = 'branching'
            elif 'resnet' in model_dir_lower:
                model_type = 'resnet'
            else:
                model_type = 'conv'
            print(f"Found model: {model_dir} -> type: {model_type}")
            models.append({
                'name': model_dir,
                'path': model_path,
                'type': model_type
            })
    
    return models

def load_model(model_path, model_type):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}")
        
    print(f"Loading model from {model_path}")
    
    # Load checkpoint first to check its structure
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model_state_dict']
    
    # Create appropriate model based on type
    if model_type == 'fc':
        model = StandardFullyConnected()
    elif model_type == 'branching':
        model = BranchingMergingCNN()
    elif model_type == 'resnet':
        model = ResNet()
    else:
        model = StandardConvNet()
        
    # Clean state dict by removing 'module.' prefix if it exists
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
            
    # Load state dict with strict=False to ignore unexpected keys
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    return model

if __name__ == "__main__":
    # Find all available models
    available_models = find_available_models()
    if not available_models:
        print("No trained models found in trainers/outputs directory")
        sys.exit(1)
        
    # Print available models
    print("\nAvailable models:")
    print("0. Analyze all models")
    for i, model in enumerate(available_models):
        print(f"{i+1}. {model['name']} ({model['type']})")
        
    # Let user select a model
    while True:
        try:
            selection = int(input("\nSelect a model number (0-{}): ".format(len(available_models))))
            if 0 <= selection <= len(available_models):
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    if selection == 0:
        # Analyze all models
        print("\nAnalyzing all models...")
        all_results = {}
        
        for model_info in available_models:
            print(f"\n{'='*50}")
            print(f"Analyzing {model_info['name']}...")
            print(f"{'='*50}")
            
            model = load_model(model_info['path'], model_info['type'])
            stats = analyze_model(model, model_info['name'], batch_size=64, preload_gpu=True, save_plots=True)
            all_results[model_info['name']] = stats
            
            # Print summary statistics
            train_stats = stats['train']['overall']
            test_stats = stats['test']['overall']
            
            print(f"\n{model_info['name']} Results:")
            print("\nTraining Set Statistics:")
            print(f"Accuracy: {train_stats['accuracy']:.2f}%")
            print(f"Error Rate: {train_stats['error_rate']:.2f}%")
            
            print("\nTest Set Statistics:")
            print(f"Accuracy: {test_stats['accuracy']:.2f}%")
            print(f"Error Rate: {test_stats['error_rate']:.2f}%")
        
        # Print comparative summary
        print(f"\n{'='*50}")
        print("Comparative Summary:")
        print(f"{'='*50}")
        print(f"\n{'Model Name':<30} {'Train Acc':<15} {'Test Acc':<15}")
        print('-' * 60)
        for name, stats in all_results.items():
            train_acc = stats['train']['overall']['accuracy']
            test_acc = stats['test']['overall']['accuracy']
            print(f"{name:<30} {train_acc:>6.2f}%{' '*8} {test_acc:>6.2f}%")
            
    else:
        # Analyze single model
        selected_model = available_models[selection-1]
        model = load_model(selected_model['path'], selected_model['type'])
        
        # Analyze the model
        print(f"\nAnalyzing {selected_model['name']}...")
        stats = analyze_model(model, selected_model['name'], batch_size=64, preload_gpu=True, save_plots=True)
        
        # Print summary statistics
        train_stats = stats['train']['overall']
        test_stats = stats['test']['overall']
        
        print("\nTraining Set Statistics:")
        print(f"Accuracy: {train_stats['accuracy']:.2f}%")
        print(f"Error Rate: {train_stats['error_rate']:.2f}%")
        print(f"Total Samples: {train_stats['total_samples']}")
        print(f"Correct: {train_stats['correct']}")
        print(f"Incorrect: {train_stats['incorrect']}")
        
        print("\nTest Set Statistics:")
        print(f"Accuracy: {test_stats['accuracy']:.2f}%")
        print(f"Error Rate: {test_stats['error_rate']:.2f}%")
        print(f"Total Samples: {test_stats['total_samples']}")
        print(f"Correct: {test_stats['correct']}")
        print(f"Incorrect: {test_stats['incorrect']}")
    
    print("\nConfusion matrix plots have been saved to model_interpretation/outputs/[model_name]/") 