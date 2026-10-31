"""Device and hardware utility functions."""

import torch

from src.utils.logger import logger


def get_available_device() -> str:
    """Get the best available device automatically.

    Priority: CUDA > MPS > CPU

    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def log_device_info() -> None:
    """Log device and GPU information.

    Logs device availability and information for CUDA, MPS, or CPU.
    Also warms up the device to avoid lazy initialization warnings.
    """
    device_str = get_available_device()

    if device_str == "cuda" and torch.cuda.is_available():
        # Warm up CUDA to avoid cuBLAS initialization warnings
        torch.cuda.init()
        _ = torch.zeros(1).cuda()

        gpu_count = torch.cuda.device_count()
        logger.info("Device: CUDA")
        logger.info(f"GPU count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"CUDA version: {torch.version.cuda}")

    elif device_str == "mps" and torch.backends.mps.is_available():
        # Warm up MPS
        _ = torch.zeros(1, device="mps")
        logger.info("Device: MPS (Apple Silicon GPU)")
        logger.info("MPS available: True")

    else:
        logger.info("Device: CPU")
