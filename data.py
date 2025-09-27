
import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()

def load_fashion_mnist():
    # Training dataset
    train_data = datasets.FashionMNIST(
        root="data",       
        train=True,        
        download=True,     
        transform=transform
    )

    # Test dataset
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,       
        download=True,
        transform=transform
    )
    return train_data, test_data
