import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.specific_models.StandardConvNet import StandardConvNet
from trainers.generic_trainers.basic_trainer import BasicTrainer

def train_conv_model():
    # Initialize the model
    model = StandardConvNet()
    
    # Initialize the trainer with specific parameters for ConvNet
    trainer = BasicTrainer(
        model=model,
        model_name="conv_net",
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
    train_conv_model() 