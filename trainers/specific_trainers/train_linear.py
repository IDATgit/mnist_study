import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.specific_models.LinearModel import LinearModel
from trainers.generic_trainers.basic_trainer import BasicTrainer
from utils.data_loader import MNISTDataLoader

def train_linear_model():
    # Initialize the model
    model = LinearModel()
    
    # Create a custom data loader
    data_loader = MNISTDataLoader(
        batch_size=64,
        preload_gpu=torch.cuda.is_available(),
        random_labels=False,
        random_images=False,
        random_seed=42,
        num_train_samples=60000
    )
    
    # Initialize the trainer with specific parameters
    trainer = BasicTrainer(
        model=model,
        model_name='LinearModel',
        learning_rate=1e-3,
        num_epochs=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        data_loader=data_loader
    )
    
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    train_linear_model() 