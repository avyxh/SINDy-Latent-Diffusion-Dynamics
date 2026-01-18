"""
Data loading and preprocessing utilities for MNIST.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(batch_size=128, data_dir='./data'):
    """
    Load MNIST dataset and create dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        data_dir: Directory to store/load MNIST data
        
    Returns:
        train_loader: Training dataloader
        test_loader: Test dataloader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor, scales to [0,1]
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_class_specific_dataloader(class_label, batch_size=128, data_dir='./data'):
    """
    Get dataloader for a specific digit class.
    
    Args:
        class_label: Which digit (0-9)
        batch_size: Batch size
        data_dir: Data directory
        
    Returns:
        DataLoader with only samples from specified class
    """
    # Load full dataset
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Filter for specific class
    indices = [i for i, (_, label) in enumerate(dataset) if label == class_label]
    
    # Create subset
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    
    # Create dataloader
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return loader


def flatten_image(image):
    """
    Flatten image tensor for VAE input.
    
    Args:
        image: Image tensor (batch_size, 1, 28, 28)
        
    Returns:
        Flattened tensor (batch_size, 784)
    """
    return image.view(image.size(0), -1)


def unflatten_image(flat_image):
    """
    Unflatten image tensor for visualization.
    
    Args:
        flat_image: Flattened tensor (batch_size, 784)
        
    Returns:
        Image tensor (batch_size, 1, 28, 28)
    """
    return flat_image.view(flat_image.size(0), 1, 28, 28)