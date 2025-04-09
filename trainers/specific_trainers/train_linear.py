import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.specific_models.LinearModel import LinearModel
from trainers.generic_trainers.basic_trainer import BasicTrainer

def train_linear_model():
    # Initialize the model
    model = LinearModel()
    
    # Initialize the trainer with specific parameters
    trainer = BasicTrainer(
        model=model,
        model_name='LinearModel',
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        preload_gpu=True
    )
    
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    train_linear_model() 