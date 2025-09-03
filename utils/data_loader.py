import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

class MNISTDataLoader:
    """
    A class to handle MNIST dataset loading with consistent preprocessing.
    """
    def __init__(self, batch_size=64, num_workers=2, root_dir='./data', preload_gpu=False, 
                 random_labels=False, random_seed=42, num_train_samples=60000, random_images=False):
        """
        Initialize the data loader with given parameters.
        
        Args:
            batch_size (int): Number of samples per batch
            num_workers (int): Number of subprocesses for data loading
            root_dir (str): Root directory for storing the dataset
            preload_gpu (bool): If True, load entire dataset into GPU memory
            random_labels (bool): If True, assign random labels to training data
            random_seed (int): Seed for random operations
            num_train_samples (int): Number of training samples to use
            random_images (bool): If True, replace images with random noise
        """
        self.batch_size = batch_size
        self.num_workers = num_workers if not preload_gpu else 0  # No workers needed for GPU preloaded data
        self.root_dir = root_dir
        self.preload_gpu = preload_gpu
        self.random_labels = random_labels
        self.random_images = random_images
        self.random_seed = random_seed
        self.num_train_samples = num_train_samples
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
        self._preprocess_training_set()

        # Preload to GPU if requested
        if preload_gpu:
            print("Preloading MNIST dataset to GPU...")
            self.train_dataset = self._preload_to_gpu(self.train_dataset)
            self.test_dataset = self._preload_to_gpu(self.test_dataset)
            print("Dataset loaded to GPU successfully!")
    
    
    def _preprocess_training_set(self):
        """Preprocess training set by limiting samples and optionally randomizing labels/images"""
        # Limit number of training samples
        indices = list(range(len(self.train_dataset.data)))[:self.num_train_samples]
        self.train_dataset.data = self.train_dataset.data[indices]
        self.train_dataset.targets = self.train_dataset.targets[indices]

        # Randomize labels if requested
        if self.random_labels:
            torch.manual_seed(self.random_seed)
            random_targets = torch.randint(0, 10, (self.num_train_samples,))
            self.train_dataset.targets = random_targets
            
        # Replace images with random noise if requested
        if self.random_images:
            self._replace_with_random_images(self.train_dataset)
            
    def _replace_with_random_images(self, dataset):
        """
        Replace real images with random uniform noise.
        
        Args:
            dataset: PyTorch dataset
        """
        # Set the random seed for reproducibility
        torch.manual_seed(self.random_seed)
        
        # Create random noise images with the same shape as MNIST (28x28)
        # We use uniform noise in range [0, 255] to match MNIST's original range
        random_data = torch.randint(0, 256, dataset.data.shape, dtype=torch.uint8)
        
        print(f"Replacing {len(dataset.data)} real images with random uniform noise...")
        dataset.data = random_data

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


class CIFAR10DataLoader:
    """
    A class to handle CIFAR-10 dataset loading with consistent preprocessing.
    """
    def __init__(self, batch_size=64, num_workers=2, root_dir='./data', preload_gpu=False, 
                 random_labels=False, random_seed=42, num_train_samples=50000, random_images=False):
        """
        Initialize the data loader with given parameters.
        
        Args:
            batch_size (int): Number of samples per batch
            num_workers (int): Number of subprocesses for data loading
            root_dir (str): Root directory for storing the dataset
            preload_gpu (bool): If True, load entire dataset into GPU memory
            random_labels (bool): If True, assign random labels to training data
            random_seed (int): Seed for random operations
            num_train_samples (int): Number of training samples to use
            random_images (bool): If True, replace images with random noise
        """
        self.batch_size = batch_size
        self.num_workers = num_workers if not preload_gpu else 0  # No workers needed for GPU preloaded data
        self.root_dir = root_dir
        self.preload_gpu = preload_gpu
        self.random_labels = random_labels
        self.random_images = random_images
        self.random_seed = random_seed
        self.num_train_samples = num_train_samples
        # Standard CIFAR-10 normalization values
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        
        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(28),  # Crop from 32x32 to 28x28
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Load datasets
        self.train_dataset = datasets.CIFAR10(
            root=self.root_dir,
            train=True,
            transform=self.transform,
            download=True
        )
        
        self.test_dataset = datasets.CIFAR10(
            root=self.root_dir,
            train=False,
            transform=self.transform,
            download=True
        )
        self._preprocess_training_set()

        # Preload to GPU if requested
        if preload_gpu:
            print("Preloading CIFAR-10 dataset to GPU...")
            self.train_dataset = self._preload_to_gpu(self.train_dataset)
            self.test_dataset = self._preload_to_gpu(self.test_dataset)
            print("Dataset loaded to GPU successfully!")
    
    
    def _preprocess_training_set(self):
        """Preprocess training set by limiting samples and optionally randomizing labels/images"""
        # Limit number of training samples
        indices = list(range(len(self.train_dataset.data)))[:self.num_train_samples]
        self.train_dataset.data = self.train_dataset.data[indices]
        self.train_dataset.targets = [self.train_dataset.targets[i] for i in indices]

        # Randomize labels if requested
        if self.random_labels:
            torch.manual_seed(self.random_seed)
            random_targets = torch.randint(0, 10, (self.num_train_samples,)).tolist()
            self.train_dataset.targets = random_targets
            
        # Replace images with random noise if requested
        if self.random_images:
            self._replace_with_random_images(self.train_dataset)
            
    def _replace_with_random_images(self, dataset):
        """
        Replace real images with random uniform noise.
        
        Args:
            dataset: PyTorch dataset
        """
        # Set the random seed for reproducibility
        torch.manual_seed(self.random_seed)
        
        # Create random noise images with the same shape as CIFAR-10 (32x32x3)
        # We use uniform noise in range [0, 255] to match CIFAR-10's original range
        random_data = torch.randint(0, 256, dataset.data.shape, dtype=torch.uint8)
        
        print(f"Replacing {len(dataset.data)} real images with random uniform noise...")
        # Convert tensor back to numpy array for CIFAR-10 dataset compatibility
        dataset.data = random_data.numpy()

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
            DataLoader: PyTorch DataLoader containing the CIFAR-10 dataset
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
        # For CIFAR-10, we need to denormalize each channel separately
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        if tensor.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        return tensor * std + mean
    
    def get_train_loader(self):
        """Get the training data loader."""
        return self.get_data_loader(train=True)
    
    def get_test_loader(self):
        """Get the test data loader."""
        return self.get_data_loader(train=False) 