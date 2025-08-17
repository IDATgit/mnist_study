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

from models.specific_models.DimensionReductionFC import DimensionReductionFC
from trainers.generic_trainers.basic_trainer import BasicTrainer
from utils.data_loader import MNISTDataLoader

model = DimensionReductionFC()
print("Number of parameters: ", model.get_num_parameters())
print("\nModel Architecture:")
model.print_architecture()

# Create a custom data loader with 10k training samples (same as small_convnet_10k)
data_loader = MNISTDataLoader(
    batch_size=64,
    preload_gpu=torch.cuda.is_available(),
    random_labels=False,
    random_images=False,
    random_seed=42,
    num_train_samples=10000  # Reduced training set for faster computation
)

def train_dimension_reduction_fc():
    """
    Train the DimensionReductionFC model with 10k training samples.
    
    This model is designed to test the hypothesis that dimensionality bottlenecks
    drive Fisher Information sensitivity patterns. Each layer systematically 
    reduces dimensions by approximately 50%.
    """
    # Initialize the trainer with specific parameters
    trainer = BasicTrainer(
        model=model,
        model_name=model_name,
        learning_rate=1e-3,
        num_epochs=100,  # More epochs since FC networks might need more training
        device='cuda' if torch.cuda.is_available() else 'cpu',
        data_loader=data_loader,
        visualization=True
    )
    
    # Train the model
    trainer.train()
    
    return trainer.get_history()

if __name__ == "__main__":
    print(f"\n🔬 Training {model_name} for Fisher Information bottleneck analysis...")
    print(f"📊 Using {data_loader.num_train_samples} training samples")
    print(f"🏗️  Architecture: 784→392→196→98→49→25→12→10 (systematic 2x reduction)")
    print("=" * 70)
    
    train_dimension_reduction_fc()
