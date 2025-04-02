import torch
import sys
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import MNISTDataLoader

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

def analyze_model(model, batch_size=64, device=None, preload_gpu=False, save_plots=True):
    """
    Analyze model's classification performance on both training and test sets.
    
    Args:
        model (nn.Module): The trained model to analyze
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
        output_dir = os.path.join('model_interpretation', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrices
    train_overall = train_stats.get_overall_stats()
    test_overall = test_stats.get_overall_stats()
    
    # Training set confusion matrix
    train_title = f"Training Set Confusion Matrix\nAccuracy: {train_overall['accuracy']:.2f}%"
    if save_plots:
        train_path = os.path.join(output_dir, 'train_confusion_matrix.png')
        plot_confusion_matrix(train_stats.confusion_matrix, train_title, train_path)
    else:
        plot_confusion_matrix(train_stats.confusion_matrix, train_title)
    
    # Test set confusion matrix
    test_title = f"Test Set Confusion Matrix\nAccuracy: {test_overall['accuracy']:.2f}%"
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

if __name__ == "__main__":
    # Example usage
    from models.FullyConnected import FullyConnected
    
    # Create a model
    model = FullyConnected([784, 128, 64, 10])
    
    # Load a trained model if available
    model_path = "trainer/outputs/fc_mnist_test/checkpoints/model_best.pt"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Analyze the model
        stats = analyze_model(model, preload_gpu=True, save_plots=True)
    else:
        print(f"No trained model found at {model_path}")
        print("Please train a model first using basic_trainer.py") 