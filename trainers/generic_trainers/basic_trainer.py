import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import shutil

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_loader import MNISTDataLoader

class BasicTrainer:
    """
    A basic trainer class for training neural networks on MNIST.
    """
    def __init__(
        self,
        model,
        model_name=None,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=10,
        device=None,
        preload_gpu=False
    ):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The neural network to train
            model_name (str): Name for the model run (for logging)
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for training
            num_epochs (int): Number of epochs to train
            device (str): Device to train on ('cuda' or 'cpu')
            preload_gpu (bool): If True, preload entire dataset to GPU
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model and move to device
        self.model = model.to(self.device)
        
        # Set model name
        if model_name is None:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"model_{current_time}"
        self.model_name = model_name
        
        # Create output directories
        self.output_dir = os.path.join('trainers', 'outputs', self.model_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize data loader
        self.data_loader = MNISTDataLoader(
            batch_size=batch_size,
            preload_gpu=preload_gpu and str(self.device) == 'cuda'
        )
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
        # Best model tracking
        self.best_accuracy = 0.0
        self.best_epoch = 0
    
    def save_checkpoint(self, epoch, test_acc, is_best=False):
        """
        Save a checkpoint of the model.
        
        Args:
            epoch (int): Current epoch number
            test_acc (float): Test accuracy for this epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'test_acc': test_acc,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best accuracy
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
            shutil.copyfile(checkpoint_path, best_path)
    
    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        train_loader = self.data_loader.get_train_loader()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def test(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        test_loader = self.data_loader.get_test_loader()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        
        return test_loss, test_acc
    
    def train(self):
        """Train the model for the specified number of epochs."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Outputs will be saved to: {self.output_dir}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Test
            test_loss, test_acc = self.test()
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc, epoch)
            
            # Check if this is the best model
            is_best = test_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = test_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, test_acc, is_best)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            if is_best:
                print(f"New best model! Best accuracy: {self.best_accuracy:.2f}%")
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\nTraining completed. Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch+1}")
    
    def get_history(self):
        """Get the training history."""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch
        } 