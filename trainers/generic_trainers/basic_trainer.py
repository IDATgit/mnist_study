import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import threading
import queue
import time

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
        preload_gpu=False,
        random_labels=False,
        random_seed=42,
        num_train_samples=60000,
        random_images=False,
        data_loader=None,
        visualization=False,
        visualization_mode='train'  # Options: 'train', 'test', or 'both'
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
            random_labels (bool): If True, use random labels for training
            random_seed (int): Random seed for reproducibility
            num_train_samples (int): Number of training samples to use
            random_images (bool): If True, use random noise images instead of real MNIST
            data_loader: Optional custom data loader. If None, a MNISTDataLoader will be created.
            visualization (bool): If True, visualize the first batch with predictions
            visualization_mode (str): What to visualize - 'train', 'test', or 'both'
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
        self.random_labels = random_labels
        self.random_seed = random_seed
        self.num_train_samples = num_train_samples
        self.random_images = random_images
        self.visualization = visualization
        self.visualization_mode = visualization_mode
        
        # Initialize criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize data loader
        if data_loader is not None:
            self.data_loader = data_loader
        else:
            self.data_loader = MNISTDataLoader(
                batch_size=batch_size,
                preload_gpu=preload_gpu and str(self.device) == 'cuda',
                random_labels=random_labels,
                random_seed=random_seed,
                num_train_samples=num_train_samples,
                random_images=random_images
            )
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.gradient_norms = []
        
        # Best model tracking
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
        # Setup visualization window if enabled
        if self.visualization:
            # Get screen resolution
            try:
                # Try with tkinter first
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
            except:
                try:
                    # Try with PyQt5
                    from PyQt5.QtWidgets import QApplication
                    app = QApplication([])
                    screen = app.primaryScreen()
                    geometry = screen.availableGeometry()
                    screen_width = geometry.width()
                    screen_height = geometry.height()
                except:
                    # Default values if detection fails
                    screen_width = 1920
                    screen_height = 1080
            
            # Calculate window size (1/4 of screen area means each dimension is 1/2)
            window_width = int(screen_width / 2)
            window_height = int(screen_height / 2)
            
            # Set up thread-safe queue for visualization data
            self.vis_queue = queue.Queue()
            self.vis_running = True
            self.current_epoch = 0
            
            # Start visualization thread
            self.vis_thread = threading.Thread(target=self._visualization_worker, 
                                              args=(window_width, window_height))
            self.vis_thread.daemon = True  # Thread will exit when main program exits
            self.vis_thread.start()
    
    def _visualization_worker(self, window_width, window_height):
        """Worker thread function for visualization."""
        plt.ion()  # Turn on interactive mode
        
        # Create figure with appropriate size
        vis_figure = plt.figure(figsize=(window_width/100, window_height/100))  # Convert pixels to inches
        vis_figure.suptitle(f"Training Visualization - Epoch: 0", fontsize=16)
        
        # Position window at top-left and size to ~1/4 of screen
        mngr = plt.get_current_fig_manager()
        
        # The exact implementation depends on the backend
        try:
            # For TkAgg backend
            mngr.window.wm_geometry(f"{window_width}x{window_height}+0+0")
        except:
            try:
                # For Qt backend
                mngr.window.setGeometry(0, 0, window_width, window_height)
            except:
                pass  # Fallback if positioning fails
        
        plt.pause(0.1)  # Give time for window to initialize
        
        # Visualization loop
        while self.vis_running:
            try:
                # Try to get visualization data with a timeout
                vis_data = self.vis_queue.get(timeout=0.1)
                
                inputs = vis_data['inputs']
                outputs = vis_data['outputs']
                targets = vis_data['targets']
                epoch = vis_data['epoch']
                is_training = vis_data['is_training']
                
                # Get predictions
                _, predicted = outputs.max(1)
                
                # Move tensors to CPU for visualization
                inputs = inputs.cpu()
                predicted = predicted.cpu()
                targets = targets.cpu()
                
                # Determine grid size
                batch_size = inputs.size(0)
                grid_size = min(4, int(batch_size**0.5))  # Show at most 4x4 images
                
                # Clear the current figure
                vis_figure.clf()
                
                # Update title with epoch and phase
                phase = "Training" if is_training else "Testing"
                vis_figure.suptitle(f"{phase} Visualization - Epoch: {epoch}", fontsize=16)
                
                for i in range(min(grid_size * grid_size, batch_size)):
                    # Get image
                    img = inputs[i].squeeze().numpy()
                    
                    # Get prediction and target
                    pred = predicted[i].item()
                    target = targets[i].item()
                    correct = pred == target
                    
                    # Plot image
                    ax = vis_figure.add_subplot(grid_size, grid_size, i + 1)
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f'Pred: {pred}, True: {target}', 
                               color='green' if correct else 'red')
                    ax.axis('off')
                
                vis_figure.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
                vis_figure.canvas.draw()
                plt.pause(0.001)
                
                self.vis_queue.task_done()
                
            except queue.Empty:
                # If queue is empty, just wait a bit
                plt.pause(0.1)
            except Exception as e:
                print(f"Visualization error: {e}")
                time.sleep(0.5)  # Sleep to avoid spinning too fast on errors
        
        # Clean up
        plt.close(vis_figure)
    
    def _visualize_batch(self, inputs, outputs, targets, epoch, is_training=True):
        """
        Queue a batch for visualization in the worker thread.
        
        Args:
            inputs: Input tensor
            outputs: Model output tensor
            targets: Target tensor
            epoch: Current epoch number
            is_training: Whether this is training or testing data
        """
        if not self.visualization:
            return
            
        # Put the data in the visualization queue
        self.vis_queue.put({
            'inputs': inputs.detach(),  # Detach tensors from computation graph
            'outputs': outputs.detach(),
            'targets': targets.detach(),
            'epoch': epoch,
            'is_training': is_training
        })
    
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
        
        # Always save the latest model
        latest_path = os.path.join(self.checkpoint_dir, 'model_latest.pt')
        shutil.copyfile(checkpoint_path, latest_path)
        
        # Save best model if this is the best accuracy
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
            shutil.copyfile(checkpoint_path, best_path)
    
    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        train_loader = self.data_loader.get_train_loader()
        
        running_loss = 0.0
        correct = 0
        total = 0
        total_grad_norm = 0.0
        
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
            
            # Calculate gradient norm
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.model.parameters() if p.grad is not None]))
            total_grad_norm += grad_norm.item()
            
            self.optimizer.step()
            
            # Visualize first batch if visualization is enabled for training
            if batch_idx == 0 and self.visualization and self.visualization_mode in ['train', 'both']:
                self._visualize_batch(inputs, outputs, targets, epoch, is_training=True)
            
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
        epoch_grad_norm = total_grad_norm / len(train_loader)
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.gradient_norms.append(epoch_grad_norm)
        
        return epoch_loss, epoch_acc
    
    def test(self, epoch):
        """Evaluate the model on the test set."""
        self.model.eval()
        test_loader = self.data_loader.get_test_loader()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Visualize first batch if visualization is enabled for testing
                if batch_idx == 0 and self.visualization and self.visualization_mode in ['test', 'both']:
                    self._visualize_batch(inputs, outputs, targets, epoch, is_training=False)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        
        return test_loss, test_acc
    
    def _plot_metrics(self):
        """Plot and save training metrics."""
        plt.figure(figsize=(10, 15))
        
        # Plot 1: Training and Test Loss
        plt.subplot(3, 1, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Training and Test Accuracy
        plt.subplot(3, 1, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Gradient L2 Norm
        plt.subplot(3, 1, 3)
        plt.plot(self.gradient_norms)
        plt.title('Gradient L2 Norm over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'))
        plt.close()

    def train(self):
        """Train the model for the specified number of epochs."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Outputs will be saved to: {self.output_dir}")
        print(f"Training samples: {self.num_train_samples}")
        if self.random_labels:
            print(f"Using random labels with seed: {self.random_seed}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train one epoch
            train_loss, train_acc = self.train_epoch(epoch + 1)
            
            # Test
            test_loss, test_acc = self.test(epoch + 1)
            
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
        
        # Plot and save metrics at the end of training
        self._plot_metrics()
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Clean up visualization thread
        if self.visualization:
            self.vis_running = False
            self.vis_thread.join(timeout=1.0)  # Wait for visualization thread to finish
        
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