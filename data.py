import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import math

def load_data(path , batch_size = 32):
    transform_cifar = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data_cifar = datasets.CIFAR10(
        root=path, train=True, download=False, transform=transform_cifar
    )
    test_data_cifar = datasets.CIFAR10(
        root=path , train=False, download=False, transform=transform_cifar
    )

    train_loader_cifar = torch.utils.data.DataLoader(train_data_cifar, batch_size=batch_size, shuffle=False)
    test_loader_cifar = torch.utils.data.DataLoader(test_data_cifar, batch_size=batch_size, shuffle=False)
    print(f"batch_Size : {batch_size}---------------data loaded successfully----------------------")


    return train_loader_cifar , test_loader_cifar
