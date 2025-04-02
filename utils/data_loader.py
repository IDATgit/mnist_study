import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

class MNISTDataLoader:
    """
    A class to handle MNIST dataset loading with consistent preprocessing.
    """
    def __init__(self, batch_size=64, num_workers=2, root_dir='./data', preload_gpu=False):
        """
        Initialize the data loader with given parameters.
        
        Args:
            batch_size (int): Number of samples per batch
            num_workers (int): Number of subprocesses for data loading
            root_dir (str): Root directory for storing the dataset
            preload_gpu (bool): If True, load entire dataset into GPU memory
        """
        self.batch_size = batch_size
        self.num_workers = num_workers if not preload_gpu else 0  # No workers needed for GPU preloaded data
        self.root_dir = root_dir
        self.preload_gpu = preload_gpu
        
        # Standard MNIST normalization values
        self.mean = 0.1307
        self.std = 0.3081
        
        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))
        ])
        
        # Load datasets
        self.train_dataset = datasets.MNIST(
            root=self.root_dir,
            train=True,
            transform=self.transform,
            download=True
        )
        
        self.test_dataset = datasets.MNIST(
            root=self.root_dir,
            train=False,
            transform=self.transform,
            download=True
        )
        
        # Preload to GPU if requested
        if preload_gpu:
            print("Preloading MNIST dataset to GPU...")
            self.train_dataset = self._preload_to_gpu(self.train_dataset)
            self.test_dataset = self._preload_to_gpu(self.test_dataset)
            print("Dataset loaded to GPU successfully!")
    
    def _preload_to_gpu(self, dataset):
        """
        Preload an entire dataset to GPU memory.
        
        Args:
            dataset: PyTorch dataset
        
        Returns:
            TensorDataset: Dataset with tensors in GPU memory
        """
        # Load all data at once
        dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=self.num_workers)
        data, targets = next(iter(dataloader))
        
        # Move to GPU
        data = data.cuda()
        targets = targets.cuda()
        
        return TensorDataset(data, targets)
    
    def get_data_loader(self, train=True):
        """
        Get DataLoader for either training or test set.
        
        Args:
            train (bool): If True, load training data, else load test data
        
        Returns:
            DataLoader: PyTorch DataLoader containing the MNIST dataset
        """
        dataset = self.train_dataset if train else self.test_dataset
        
        # For GPU preloaded data, we don't need pin_memory or num_workers
        if self.preload_gpu:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=train,
                pin_memory=False,
                num_workers=0
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=train,
                pin_memory=True,
                num_workers=self.num_workers
            )
    
    def denormalize(self, tensor):
        """
        Denormalize the tensor (convert from normalized space back to [0,1] range).
        
        Args:
            tensor (torch.Tensor): Input tensor in normalized space
        
        Returns:
            torch.Tensor: Denormalized tensor
        """
        return tensor * self.std + self.mean
    
    def get_train_loader(self):
        """Get the training data loader."""
        return self.get_data_loader(train=True)
    
    def get_test_loader(self):
        """Get the test data loader."""
        return self.get_data_loader(train=False) 