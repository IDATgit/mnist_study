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



from models.specific_models.SmallConvNet10x import SmallConvNet10x
from trainers.generic_trainers.basic_trainer import BasicTrainer
from utils.data_loader import MNISTDataLoader



model = SmallConvNet10x()
print("Number of parameters: ", model.get_num_parameters())

# Create a custom data loader
data_loader = MNISTDataLoader(
    batch_size=64,
    preload_gpu=torch.cuda.is_available(),
    random_labels=False,
    random_images=False,
    random_seed=42,
    num_train_samples=10000
)



def train_small_convnet():
    # Initialize the trainer with specific parameters
    trainer = BasicTrainer(
        model=model,
        model_name=model_name,
        learning_rate=1e-3,
        num_epochs=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        data_loader=data_loader,
        visualization=True
    )
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    train_small_convnet() 