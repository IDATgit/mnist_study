import torch
import sys
import os

# Fix the import path correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from models.specific_models.StandardFullyConnected import StandardFullyConnected
from trainers.generic_trainers.basic_trainer import BasicTrainer
from utils.data_loader import MNISTDataLoader

model_name = 'StandardFullyConnected'

model = StandardFullyConnected()

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
    model_name=model_name,
    learning_rate=0.001,
    num_epochs=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    data_loader=data_loader
)

def train_fully_connected_model():
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    train_fully_connected_model() 