from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # ensure dimensions divisible by 8
    transforms.ToTensor()
])

def load_fashion_mnist():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    return train_data, test_data
