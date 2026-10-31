"""CIFAR-10 dataset loading utilities."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(
    batch_size: int = 128,
    data_root: str = "data/CIFAR10/raw",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test data loaders."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
