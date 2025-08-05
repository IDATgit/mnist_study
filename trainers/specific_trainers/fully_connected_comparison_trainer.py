import torch
import sys
import os
import matplotlib.pyplot as plt

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from models.specific_models.StandardFullyConnected import StandardFullyConnected
from trainers.generic_trainers.basic_trainer import BasicTrainer
from trainers.generic_trainers.line_search_sgd_trainer import LineSearchSGDTrainer
from utils.data_loader import MNISTDataLoader

# --- Configuration ---
MODEL_NAME_BASE = 'StandardFullyConnected_Comparison'
NUM_EPOCHS = 20  # Adjust as needed
BATCH_SIZE = 64
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basic Trainer specific config
BASIC_TRAINER_LR = 0.001

# Line Search Trainer specific config
LINE_SEARCH_INITIAL_LR = 1e-4
MAX_LINE_SEARCH_DOUBLINGS = 25

OUTPUT_DIR = os.path.join(project_root, 'trainers', 'outputs', MODEL_NAME_BASE)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End Configuration ---

def train_with_basic_trainer():
    print("\n--- Training with BasicTrainer ---")
    torch.manual_seed(RANDOM_SEED) # For model initialization reproducibility
    model = StandardFullyConnected().to(DEVICE)
    
    data_loader = MNISTDataLoader(
        batch_size=BATCH_SIZE,
        preload_gpu=(DEVICE == 'cuda'),
        random_seed=RANDOM_SEED
    )
    
    trainer = BasicTrainer(
        model=model,
        model_name=f"{MODEL_NAME_BASE}_Basic",
        learning_rate=BASIC_TRAINER_LR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        data_loader=data_loader,
        visualization=False # Disable visualization for comparison script
    )
    trainer.train()
    return trainer.get_history()

def train_with_line_search_trainer():
    print("\n--- Training with LineSearchSGDTrainer ---")
    torch.manual_seed(RANDOM_SEED) # For model initialization reproducibility
    model = StandardFullyConnected().to(DEVICE)
    
    data_loader = MNISTDataLoader(
        batch_size=BATCH_SIZE,
        preload_gpu=(DEVICE == 'cuda'),
        random_seed=RANDOM_SEED
    )
    
    trainer = LineSearchSGDTrainer(
        model=model,
        model_name=f"{MODEL_NAME_BASE}_LineSearch",
        learning_rate=LINE_SEARCH_INITIAL_LR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        data_loader=data_loader,
        max_line_search_doublings=MAX_LINE_SEARCH_DOUBLINGS,
        visualization=False # Disable visualization for comparison script
    )
    trainer.train()
    return trainer.get_history()

def plot_comparison(history_basic, history_line_search):
    print("\n--- Plotting Comparison ---")
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(15, 12))

    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history_basic['train_losses'], label='Basic Trainer Train Loss', marker='o', linestyle='-')
    plt.plot(epochs_range, history_line_search['train_losses'], label='Line Search SGD Train Loss', marker='x', linestyle='--')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Test Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history_basic['test_losses'], label='Basic Trainer Test Loss', marker='o', linestyle='-')
    plt.plot(epochs_range, history_line_search['test_losses'], label='Line Search SGD Test Loss', marker='x', linestyle='--')
    plt.title('Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 3: Training Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history_basic['train_accuracies'], label='Basic Trainer Train Accuracy', marker='o', linestyle='-')
    plt.plot(epochs_range, history_line_search['train_accuracies'], label='Line Search SGD Train Accuracy', marker='x', linestyle='--')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot 4: Test Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history_basic['test_accuracies'], label='Basic Trainer Test Accuracy', marker='o', linestyle='-')
    plt.plot(epochs_range, history_line_search['test_accuracies'], label='Line Search SGD Test Accuracy', marker='x', linestyle='--')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_BASE}_comparison_metrics.png')
    plt.savefig(plot_filename)
    print(f"Comparison plot saved to: {plot_filename}")
    plt.close()

if __name__ == "__main__":
    history_basic = train_with_basic_trainer()
    history_line_search = train_with_line_search_trainer()
    plot_comparison(history_basic, history_line_search)
    print("\nComparison finished.") 