import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()

def load_fashion_mnist(datasets):

    # Training dataset
    train_data = datasets.FashionMNIST(
        root="data",       # folder to save the dataset
        train=True,        # True = training data
        download=True,     # download if not already present
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
