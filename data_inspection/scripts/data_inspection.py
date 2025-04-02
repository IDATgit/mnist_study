import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_loader import MNISTDataLoader

def calculate_dataset_statistics(data_loader):
    """
    Calculate mean and variance of the dataset.
    
    Args:
        data_loader: DataLoader containing the MNIST dataset
    
    Returns:
        tuple: (mean, variance) of the dataset
    """
    mean = 0.
    mean_sq = 0.
    total_samples = 0
    
    # First pass: calculate mean
    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_samples += batch_samples
    
    mean = mean / total_samples
    
    # Second pass: calculate variance
    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean_sq += ((images - mean) ** 2).mean(2).sum(0)
    
    variance = mean_sq / total_samples
    
    return mean, variance

def visualize_digits_by_label(data_loader, mnist_loader, num_examples=10, output_dir=None):
    """
    Create a 10x10 grid visualization where each row shows examples of the same digit.
    
    Args:
        data_loader: DataLoader containing the MNIST dataset
        mnist_loader: MNISTDataLoader instance for denormalization
        num_examples: Number of examples to show per digit
        output_dir: Directory to save the output visualization
    """
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If output_dir is not specified, use the output directory relative to the script
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(script_dir), 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create a figure with 10x10 subplots
    fig, axes = plt.subplots(10, num_examples, figsize=(1.5*num_examples, 15))
    
    # Dictionary to store images for each digit
    digit_images = {i: [] for i in range(10)}
    
    # Collect images for each digit
    for images, labels in data_loader:
        for img, label in zip(images, labels):
            if len(digit_images[label.item()]) < num_examples:
                digit_images[label.item()].append(img)
        
        # Check if we have enough examples for each digit
        if all(len(imgs) >= num_examples for imgs in digit_images.values()):
            break
    
    # Plot the images
    for digit in range(10):
        for example in range(num_examples):
            img = digit_images[digit][example]
            # Denormalize the image using the loader's method
            img = mnist_loader.denormalize(img)
            axes[digit, example].imshow(img.squeeze(), cmap='gray')
            axes[digit, example].axis('off')
            if example == 0:
                axes[digit, example].set_title(f'Digit {digit}')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'mnist_digits_grid_{num_examples}ex.png')
    print(f"Saving visualization to: {output_path}")
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_label_distribution(data_loader, output_dir=None):
    """
    Create a histogram showing the distribution of labels in the dataset.
    
    Args:
        data_loader: DataLoader containing the MNIST dataset
        output_dir: Directory to save the output visualization
    """
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If output_dir is not specified, use the output directory relative to the script
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(script_dir), 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all labels
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.tolist())
    
    # Count occurrences of each digit
    counts = np.bincount(all_labels)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), counts, width=0.8)
    plt.title('Distribution of MNIST Labels')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.xticks(range(10))
    
    # Add value labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'mnist_label_distribution.png')
    print(f"Saving label distribution plot to: {output_path}")
    plt.savefig(output_path)
    plt.close()
    return output_path

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize the MNIST data loader
    mnist_loader = MNISTDataLoader(batch_size=64)
    train_loader = mnist_loader.get_train_loader()
    
    # Calculate and print dataset statistics
    mean, variance = calculate_dataset_statistics(train_loader)
    print("\nDataset Statistics:")
    print(f"Mean: {mean.item():.4f}")
    print(f"Variance: {variance.item():.4f}")
    
    # Create the visualizations
    grid_path_10 = visualize_digits_by_label(train_loader, mnist_loader, num_examples=10)
    print(f"\nSmall grid visualization has been saved as '{grid_path_10}'")
    
    grid_path_100 = visualize_digits_by_label(train_loader, mnist_loader, num_examples=100)
    print(f"Large grid visualization has been saved as '{grid_path_100}'")
    
    dist_path = plot_label_distribution(train_loader)
    print(f"Label distribution plot has been saved as '{dist_path}'")

if __name__ == "__main__":
    main() 