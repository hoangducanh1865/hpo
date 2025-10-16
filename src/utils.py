import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import Config as config


class Utils:
    @staticmethod
    def load_fashion_mnist(batch_size=config.batch_size):
        """Load FashionMNIST dataset with train/validation/test split."""
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Load full datasets
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, transform=transform, download=True)

        # Split training data into train and validation sets
        train_size = int(0.8 * len(train_dataset))  # 80% for training
        val_size = len(train_dataset) - train_size   # 20% for validation
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    
    @staticmethod
    def accuracy(y_hat, y):
        """Compute number of correct predictions."""
        preds = torch.argmax(y_hat, dim=1)
        return (preds == y).float().sum()
