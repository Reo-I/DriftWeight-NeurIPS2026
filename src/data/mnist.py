"""MNIST dataset loading utilities."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 64,
    data_root: str = "data",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create MNIST train and test data loaders.

    Downloads the MNIST dataset if not already present and creates
    DataLoader instances for training and testing.

    Args:
        batch_size: Number of samples per batch.
        data_root: Root directory for storing the dataset.
        num_workers: Number of subprocesses for data loading.

    Returns:
        Tuple of (train_loader, test_loader).

    Example:
        >>> train_loader, test_loader = get_mnist_loaders(batch_size=128)
        >>> for images, labels in train_loader:
        ...     # Training loop
        ...     pass
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
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
