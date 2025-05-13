import torch
import sys
import os

# Get the full path to the current file
current_file_path = __file__
current_filename = os.path.basename(__file__)
model_name = current_filename[:-3]

# Fix the import path correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from models.specific_models.SmallConvNet import SmallConvNet
from trainers.generic_trainers.basic_trainer import BasicTrainer
from utils.data_loader import MNISTDataLoader

model = SmallConvNet()

# Create a custom data loader with random images
data_loader = MNISTDataLoader(
    batch_size=64,
    preload_gpu=torch.cuda.is_available(),
    random_labels=False,
    random_images=True,  # Use random noise instead of real MNIST images
    random_seed=42,
    num_train_samples=10000  # Use fewer samples for random images training
)



def train_small_convnet_random_images():
    # Initialize the trainer with specific parameters
    trainer = BasicTrainer(
        model=model,
        model_name=model_name,
        learning_rate=1e-4,  # Lower learning rate for random images
        num_epochs=1000,      # More epochs for random data
        device='cuda' if torch.cuda.is_available() else 'cpu',
        data_loader=data_loader,
        visualization=True   # Enable visualization to see the random images
    )
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    train_small_convnet_random_images() 