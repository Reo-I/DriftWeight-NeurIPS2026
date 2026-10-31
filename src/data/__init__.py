"""Data loading utilities."""

from src.data.cifar10 import get_cifar10_loaders
from src.data.fashion_mnist import get_fashion_mnist_loaders
from src.data.mnist import get_mnist_loaders

__all__ = ["get_mnist_loaders", "get_fashion_mnist_loaders", "get_cifar10_loaders"]
