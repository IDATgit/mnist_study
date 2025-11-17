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



from models.specific_models.ReGenInception import ReGenInception
from trainers.generic_trainers.basic_trainer import BasicTrainer
from utils.data_loader import CIFAR10DataLoader



model = ReGenInception()
print("Number of parameters: ", model.get_num_parameters())

# Create a custom data loader with random labels
data_loader = CIFAR10DataLoader(
    batch_size=64,
    preload_gpu=torch.cuda.is_available(),
    random_labels=True,  # Enable random labels
    random_images=False,
    random_seed=42,
)



def train_regen_inception_random_labels():
    # Initialize the trainer with specific parameters
    trainer = BasicTrainer(
        model=model,
        model_name=model_name,
        learning_rate=1e-3,
        num_epochs=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        data_loader=data_loader,
        visualization=False
    )
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    train_regen_inception_random_labels()
