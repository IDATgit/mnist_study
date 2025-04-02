import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.specific_models.StandardFullyConnected import StandardFullyConnected
from trainers.generic_trainers.basic_trainer import BasicTrainer

def train_fully_connected_model():
    # Initialize the model
    model = StandardFullyConnected()
    
    # Initialize the trainer with specific parameters for FullyConnected
    trainer = BasicTrainer(
        model=model,
        model_name=model.get_name(),
        learning_rate=0.001,
        batch_size=64,
        num_epochs=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        preload_gpu=True
    )
    
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    train_fully_connected_model() 