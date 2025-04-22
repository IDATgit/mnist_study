import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.specific_models.ResNet import ResNet
from trainers.generic_trainers.basic_trainer import BasicTrainer
from utils.data_loader import MNISTDataLoader

def train_resnet_model_random_images():
    # Initialize the model
    model = ResNet()
    
    # Create a custom data loader
    data_loader = MNISTDataLoader(
        batch_size=64,
        preload_gpu=torch.cuda.is_available(),
        random_labels=False,
        random_images=True,
        random_seed=42,
        num_train_samples=7000
    )
    
    # Initialize the trainer with specific parameters
    trainer = BasicTrainer(
        model=model,
        model_name='ResNet18_RandomImages',
        learning_rate=0.001,
        num_epochs=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        data_loader=data_loader
    )
    
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    train_resnet_model_random_images() 