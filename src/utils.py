import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import Config as config


def load_fashion_mnist(batch_size=config.batch_size):
    """Load FashionMNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def accuracy(y_hat, y):
    """Compute number of correct predictions."""
    preds = torch.argmax(y_hat, dim=1)
    return (preds == y).float().sum()
