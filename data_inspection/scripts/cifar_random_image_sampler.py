import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image

# Fix the import path correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from utils.data_loader import CIFAR10DataLoader

def sample_and_save_random_images(num_samples=16, output_dir='data_inspection/output/cifar_random_image_test/examples'):
    """
    Sample random CIFAR-10 images (both original and random noise) and save them to files.
    
    Args:
        num_samples (int): Number of images to sample and save
        output_dir (str): Directory to save the images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # CIFAR-10 class names
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("Sampling original CIFAR-10 images...")
    
    # Load original CIFAR-10 data
    original_loader = CIFAR10DataLoader(
        batch_size=num_samples,
        preload_gpu=False,
        random_labels=False,
        random_images=False,
        num_train_samples=50000
    )
    
    # Load random noise data  
    random_loader = CIFAR10DataLoader(
        batch_size=num_samples,
        preload_gpu=False,
        random_labels=False,
        random_images=True,  # Enable random images
        random_seed=42,
        num_train_samples=50000
    )
    
    # Get sample batches
    original_dataloader = original_loader.get_train_loader()
    random_dataloader = random_loader.get_train_loader()
    
    original_batch = next(iter(original_dataloader))
    random_batch = next(iter(random_dataloader))
    
    original_images, original_labels = original_batch
    random_images, random_labels = random_batch
    
    # Denormalize images for visualization
    # CIFAR-10 normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    def denormalize(tensor):
        return tensor * std + mean
    
    # Denormalize and clamp to [0, 1]
    original_images = torch.clamp(denormalize(original_images), 0, 1)
    random_images = torch.clamp(denormalize(random_images), 0, 1)
    
    # Create comparison grid
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('CIFAR-10 Images: Original vs Random Noise', fontsize=16)
    
    for i in range(min(num_samples, 16)):
        row = i // 4
        col_orig = (i % 4) * 2
        col_rand = col_orig + 1
        
        # Original image
        orig_img = original_images[i].permute(1, 2, 0).numpy()
        axes[row, col_orig].imshow(orig_img)
        axes[row, col_orig].set_title(f'Original\n{cifar10_classes[original_labels[i]]}', fontsize=8)
        axes[row, col_orig].axis('off')
        
        # Random noise image
        rand_img = random_images[i].permute(1, 2, 0).numpy()
        axes[row, col_rand].imshow(rand_img)
        axes[row, col_rand].set_title(f'Random\n{cifar10_classes[random_labels[i]]}', fontsize=8)
        axes[row, col_rand].axis('off')
    
    # Save comparison grid
    comparison_path = os.path.join(output_dir, 'cifar_original_vs_random_comparison.png')
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison grid to: {comparison_path}")
    
    # Save individual original images
    original_dir = os.path.join(output_dir, 'original')
    os.makedirs(original_dir, exist_ok=True)
    
    for i in range(min(num_samples, 16)):
        img = original_images[i].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        
        filename = f'original_{i:02d}_{cifar10_classes[original_labels[i]]}.png'
        filepath = os.path.join(original_dir, filename)
        img_pil.save(filepath)
    
    print(f"Saved {min(num_samples, 16)} original images to: {original_dir}")
    
    # Save individual random images
    random_dir = os.path.join(output_dir, 'random')
    os.makedirs(random_dir, exist_ok=True)
    
    for i in range(min(num_samples, 16)):
        img = random_images[i].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        
        filename = f'random_{i:02d}_{cifar10_classes[random_labels[i]]}.png'
        filepath = os.path.join(random_dir, filename)
        img_pil.save(filepath)
    
    print(f"Saved {min(num_samples, 16)} random noise images to: {random_dir}")
    
    # Create a summary plot showing statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image statistics
    orig_flat = original_images.flatten()
    ax1.hist(orig_flat.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Original Images - Pixel Value Distribution')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Random image statistics
    rand_flat = random_images.flatten()
    ax2.hist(rand_flat.numpy(), bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title('Random Noise Images - Pixel Value Distribution')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Mean pixel values per channel
    orig_means = original_images.mean(dim=(0, 2, 3))
    rand_means = random_images.mean(dim=(0, 2, 3))
    
    channels = ['Red', 'Green', 'Blue']
    x_pos = np.arange(len(channels))
    
    ax3.bar(x_pos - 0.2, orig_means.numpy(), 0.4, label='Original', alpha=0.7, color='blue')
    ax3.bar(x_pos + 0.2, rand_means.numpy(), 0.4, label='Random', alpha=0.7, color='red')
    ax3.set_title('Mean Pixel Values by Channel')
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Mean Value')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(channels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Standard deviation per channel
    orig_stds = original_images.std(dim=(0, 2, 3))
    rand_stds = random_images.std(dim=(0, 2, 3))
    
    ax4.bar(x_pos - 0.2, orig_stds.numpy(), 0.4, label='Original', alpha=0.7, color='blue')
    ax4.bar(x_pos + 0.2, rand_stds.numpy(), 0.4, label='Random', alpha=0.7, color='red')
    ax4.set_title('Standard Deviation by Channel')
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(channels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Save statistics plot
    stats_path = os.path.join(output_dir, 'cifar_statistics_comparison.png')
    plt.tight_layout()
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics comparison to: {stats_path}")
    
    print(f"\nSummary:")
    print(f"Original images - Mean: {orig_flat.mean():.4f}, Std: {orig_flat.std():.4f}")
    print(f"Random images - Mean: {rand_flat.mean():.4f}, Std: {rand_flat.std():.4f}")
    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    sample_and_save_random_images()

