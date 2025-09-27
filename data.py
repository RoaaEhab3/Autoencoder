import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

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

def preprocess(imgs):
    imgs = torch.tensor(imgs, dtype=torch.float32) / 255.0
    imgs = F.pad(imgs.unsqueeze(1), (2, 2, 2, 2)) 
    return imgs