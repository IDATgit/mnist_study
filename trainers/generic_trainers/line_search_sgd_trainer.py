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
import copy # For deepcopying model state

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_loader import MNISTDataLoader # Assuming this path is correct

class LineSearchSGDTrainer:
    """
    A trainer class for training neural networks on MNIST using SGD with a custom line search.
    """
    def __init__(
        self,
        model,
        model_name=None,
        learning_rate=0.0000001, # This will be the initial_lr for line search
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
        visualization_mode='train',  # Options: 'train', 'test', or 'both'
        max_line_search_doublings=20 # Max times to double LR in line search
    ):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The neural network to train
            model_name (str): Name for the model run (for logging)
            learning_rate (float): Initial learning rate for the line search
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
            max_line_search_doublings (int): Maximum number of times to double the learning rate during line search.
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        if model_name is None:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"model_linesearch_sgd_{current_time}"
        self.model_name = model_name
        
        self.output_dir = os.path.join('trainers', 'outputs', self.model_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
        self.initial_line_search_lr = learning_rate # Use learning_rate as initial for line search
        self.max_line_search_doublings = max_line_search_doublings
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_labels = random_labels
        self.random_seed = random_seed
        self.num_train_samples = num_train_samples
        self.random_images = random_images
        self.visualization = visualization
        self.visualization_mode = visualization_mode
        
        self.criterion = nn.CrossEntropyLoss()
        # Optimizer instance is mainly for zero_grad(). LR here is the initial_line_search_lr.
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_line_search_lr) 
        
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
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.gradient_norms = [] # We can still calculate this
        self.effective_lrs_used = [] # To store LRs used by line search

        self.best_accuracy = 0.0
        self.best_epoch = 0
        
        if self.visualization:
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
            except:
                try:
                    from PyQt5.QtWidgets import QApplication
                    app = QApplication([])
                    screen = app.primaryScreen()
                    geometry = screen.availableGeometry()
                    screen_width = geometry.width()
                    screen_height = geometry.height()
                except:
                    screen_width = 1920
                    screen_height = 1080
            
            window_width = int(screen_width / 2)
            window_height = int(screen_height / 2)
            
            self.vis_queue = queue.Queue()
            self.vis_running = True
            self.current_epoch = 0
            
            self.vis_thread = threading.Thread(target=self._visualization_worker, 
                                              args=(window_width, window_height))
            self.vis_thread.daemon = True
            self.vis_thread.start()

    def _visualization_worker(self, window_width, window_height):
        plt.ion()
        vis_figure = plt.figure(figsize=(window_width/100, window_height/100))
        vis_figure.suptitle(f"Training Visualization - Epoch: 0", fontsize=16)
        mngr = plt.get_current_fig_manager()
        try:
            mngr.window.wm_geometry(f"{window_width}x{window_height}+0+0")
        except:
            try:
                mngr.window.setGeometry(0, 0, window_width, window_height)
            except:
                pass
        plt.pause(0.1)
        
        while self.vis_running:
            try:
                vis_data = self.vis_queue.get(timeout=0.1)
                inputs, outputs, targets = vis_data['inputs'], vis_data['outputs'], vis_data['targets']
                epoch, is_training = vis_data['epoch'], vis_data['is_training']
                _, predicted = outputs.max(1)
                inputs, predicted, targets = inputs.cpu(), predicted.cpu(), targets.cpu()
                
                batch_size_vis = inputs.size(0)
                grid_size = min(4, int(batch_size_vis**0.5))
                vis_figure.clf()
                phase = "Training" if is_training else "Testing"
                vis_figure.suptitle(f"{phase} Visualization - Epoch: {epoch}", fontsize=16)
                plt.subplots_adjust(wspace=0.3, hspace=0.3)
                
                for i in range(min(grid_size * grid_size, batch_size_vis)):
                    img = inputs[i].squeeze().numpy()
                    pred, target_val = predicted[i].item(), targets[i].item()
                    correct = pred == target_val
                    ax = vis_figure.add_subplot(grid_size, grid_size, i + 1)
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f'Pred: {pred}, True: {target_val}', color='green' if correct else 'red')
                    ax.axis('off')
                
                vis_figure.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.3, h_pad=0.3)
                vis_figure.canvas.draw()
                plt.pause(0.001)
                self.vis_queue.task_done()
            except queue.Empty:
                plt.pause(0.1)
            except Exception as e:
                print(f"Visualization error: {e}")
                time.sleep(0.5)
        plt.close(vis_figure)

    def _visualize_batch(self, inputs, outputs, targets, epoch, is_training=True):
        if not self.visualization: return
        self.vis_queue.put({
            'inputs': inputs.detach(), 'outputs': outputs.detach(), 
            'targets': targets.detach(), 'epoch': epoch, 'is_training': is_training
        })

    def save_checkpoint(self, epoch, test_acc, is_best=False):
        checkpoint = {
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), # Note: optimizer state might be less relevant for line search
            'test_acc': test_acc, 'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies, 'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        latest_path = os.path.join(self.checkpoint_dir, 'model_latest.pt')
        shutil.copyfile(checkpoint_path, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
            shutil.copyfile(checkpoint_path, best_path)

    def train_epoch(self, epoch):
        self.model.train()
        train_loader = self.data_loader.get_train_loader()
        
        running_loss = 0.0
        correct = 0
        total = 0
        total_grad_norm = 0.0
        epoch_effective_lrs = []

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Initial forward pass to compute loss and gradients
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward() # Compute gradients

            # Calculate gradient norm (before any updates)
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.model.parameters() if p.grad is not None]))
            total_grad_norm += grad_norm.item()

            # --- Custom Line Search ---
            original_params_state_dict = copy.deepcopy(self.model.state_dict())
            loss_before_step_val = loss.item()
            
            best_lr_for_step = 0.0
            current_best_line_search_loss = loss_before_step_val
            test_lr = self.initial_line_search_lr # Start with the initial LR for this batch
            
            for _ in range(self.max_line_search_doublings):
                # Create a temporary model state for this test_lr
                temp_model_state = copy.deepcopy(original_params_state_dict)
                
                # Apply prospective update to the temporary state
                # We need to load the original_params_state_dict to ensure we apply the gradient step from the same point
                self.model.load_state_dict(original_params_state_dict) 
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.data.add_(param.grad.data, alpha=-test_lr)
                
                # Evaluate loss with this test_lr using the updated model
                with torch.no_grad():
                    outputs_test = self.model(inputs)
                    loss_with_test_lr = self.criterion(outputs_test, targets).item()

                if loss_with_test_lr < current_best_line_search_loss:
                    current_best_line_search_loss = loss_with_test_lr
                    best_lr_for_step = test_lr
                    test_lr *= 2 # Try a larger step
                else:
                    # Loss did not decrease, so the previous lr was better (or no lr was good)
                    break 
            
            # Apply the best update found (if any) by reloading original and applying best_lr
            self.model.load_state_dict(original_params_state_dict) # Reset to params before line search started
            if best_lr_for_step > 0:
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.data.add_(param.grad.data, alpha=-best_lr_for_step)
                actual_loss_after_step = current_best_line_search_loss
                epoch_effective_lrs.append(best_lr_for_step)
                self.writer.add_scalar('Batch/EffectiveLR', best_lr_for_step, epoch * len(train_loader) + batch_idx)
            else:
                # No step improved the loss, parameters remain as they were (already reverted)
                actual_loss_after_step = loss_before_step_val
                epoch_effective_lrs.append(0.0) # Log 0 if no update was made
                self.writer.add_scalar('Batch/EffectiveLR', 0.0, epoch * len(train_loader) + batch_idx)

            self.writer.add_scalar('Batch/Loss', actual_loss_after_step, epoch * len(train_loader) + batch_idx)
            # --- End Custom Line Search ---
            
            # Visualize first batch if needed (using outputs before line search for consistency with original loss)
            if batch_idx == 0 and self.visualization and self.visualization_mode in ['train', 'both']:
                # To visualize, we need outputs from the model *after* the best step is applied.
                with torch.no_grad():
                    final_batch_outputs = self.model(inputs)
                self._visualize_batch(inputs, final_batch_outputs, targets, epoch, is_training=True)

            running_loss += actual_loss_after_step # Use loss after line search
            
            # Statistics based on model state *after* line search update
            with torch.no_grad():
                final_outputs = self.model(inputs) # Re-evaluate outputs if needed, or use outputs_test if best_lr_for_step > 0
                _, predicted = final_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'eff_lr': f'{best_lr_for_step:.2e}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        epoch_grad_norm = total_grad_norm / len(train_loader)
        avg_effective_lr = sum(epoch_effective_lrs) / len(epoch_effective_lrs) if epoch_effective_lrs else 0
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.gradient_norms.append(epoch_grad_norm)
        self.effective_lrs_used.append(avg_effective_lr)

        self.writer.add_scalar('Epoch/AvgEffectiveLR', avg_effective_lr, epoch)
        
        return epoch_loss, epoch_acc

    def test(self, epoch):
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
                
                if batch_idx == 0 and self.visualization and self.visualization_mode in ['test', 'both']:
                    self._visualize_batch(inputs, outputs, targets, epoch, is_training=False)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        return test_loss, test_acc

    def _plot_metrics(self):
        plt.figure(figsize=(10, 20)) # Increased height for the new plot
        
        plt.subplot(4, 1, 1) # Changed to 4 rows
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Accuracy over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
        plt.legend(); plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(self.gradient_norms, label='Gradient L2 Norm')
        plt.title('Gradient L2 Norm over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Gradient Norm')
        plt.legend(); plt.grid(True)

        plt.subplot(4, 1, 4) # New subplot for effective LR
        plt.plot(self.effective_lrs_used, label='Avg Effective LR per Epoch')
        plt.title('Average Effective Learning Rate over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
        plt.legend(); plt.grid(True); plt.yscale('log') # Log scale for LR often useful

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'))
        plt.close()

    def train(self):
        print(f"Training on {self.device} with LineSearchSGD Trainer")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Outputs will be saved to: {self.output_dir}")
        print(f"Training samples: {self.num_train_samples}")
        print(f"Initial LR for line search: {self.initial_line_search_lr}, Max doublings: {self.max_line_search_doublings}")
        if self.random_labels: print(f"Using random labels with seed: {self.random_seed}")
        
        for epoch_num_human_readable in range(1, self.num_epochs + 1): # Epochs 1-indexed for print
            self.current_epoch = epoch_num_human_readable # For visualization
            print(f"Epoch {epoch_num_human_readable}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch(epoch_num_human_readable) # Pass human-readable epoch
            test_loss, test_acc = self.test(epoch_num_human_readable)
            
            # Log to TensorBoard (epoch here is 0-indexed for consistency if add_scalar expects that)
            # For clarity, I'll use epoch_num_human_readable - 1 if 0-indexed is needed for writer
            epoch_idx_for_writer = epoch_num_human_readable -1
            self.writer.add_scalar('Loss/train', train_loss, epoch_idx_for_writer)
            self.writer.add_scalar('Loss/test', test_loss, epoch_idx_for_writer)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch_idx_for_writer)
            self.writer.add_scalar('Accuracy/test', test_acc, epoch_idx_for_writer)
            # AvgEffectiveLR already logged per epoch in train_epoch using human-readable epoch
            
            is_best = test_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = test_acc
                self.best_epoch = epoch_idx_for_writer # Store 0-indexed best epoch
            
            self.save_checkpoint(epoch_idx_for_writer, test_acc, is_best)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            avg_lr_this_epoch = self.effective_lrs_used[-1] if self.effective_lrs_used else 0
            print(f"Avg Effective LR: {avg_lr_this_epoch:.2e}")
            if is_best: print(f"New best model! Best accuracy: {self.best_accuracy:.2f}%")
        
        self._plot_metrics()
        self.writer.close()
        
        if self.visualization:
            self.vis_running = False
            self.vis_thread.join(timeout=1.0)
        
        print(f"Training completed. Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch+1}")

    def get_history(self):
        return {
            'train_losses': self.train_losses, 'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses, 'test_accuracies': self.test_accuracies,
            'gradient_norms': self.gradient_norms, 'effective_lrs_used': self.effective_lrs_used,
            'best_accuracy': self.best_accuracy, 'best_epoch': self.best_epoch
        }

# Example usage (similar to how BasicTrainer might be used)
if __name__ == '__main__':
    from torchvision import models # Example model
    # This is a placeholder example; ensure your model and data are compatible.
    # For MNIST, a simpler model than resnet18 would be appropriate.
    
    # Define a simple model for MNIST
    class SimpleMNISTCNN(nn.Module):
        def __init__(self):
            super(SimpleMNISTCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50) # 20*4*4 = 320
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
            x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = torch.relu(self.fc1(x))
            x = torch.dropout(x, training=self.training)
            x = self.fc2(x)
            return torch.log_softmax(x, dim=1)

    print("Running LineSearchSGDTrainer example...")
    # model_to_train = models.resnet18(weights=None) # Example for different dataset
    # model_to_train.fc = nn.Linear(model_to_train.fc.in_features, 10) # Adjust for 10 classes
    
    model_to_train = SimpleMNISTCNN()

    # Parameters for the trainer
    # Use a small initial LR for line search as it will try to increase it
    trainer = LineSearchSGDTrainer(
        model=model_to_train,
        model_name="mnist_linesearch_test",
        learning_rate=1e-7, # Initial LR for line search
        batch_size=128,
        num_epochs=5, # Keep epochs low for a quick test
        preload_gpu=False, # Set to True if you have enough GPU memory for entire dataset
        visualization=False, # Enable if you want to see training progress visually
        max_line_search_doublings=25 
    )
    
    trainer.train()
    
    history = trainer.get_history()
    print("Training History (Line Search SGD):")
    for key, value in history.items():
        if isinstance(value, list):
            print(f"{key}: (last 5 values) {value[-5:]}")
        else:
            print(f"{key}: {value}") 