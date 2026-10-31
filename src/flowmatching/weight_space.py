"""Weight space objects for neural network weights.

This module provides classes to handle neural network weights as vectors
in a high-dimensional weight space, enabling operations like flattening
and reconstruction.
"""

from __future__ import annotations

import torch


class WeightSpaceObjectStateDict:
    """Generic weight-space object backed by named model parameters.

    This class is architecture-agnostic and can represent MLP, ResNet, etc.
    It stores only trainable parameters (not running buffers).
    """

    def __init__(
        self,
        params: dict[str, torch.Tensor],
        key_order: list[str] | None = None,
    ) -> None:
        self.params = {k: v.detach().clone() for k, v in params.items()}
        self.key_order = key_order or list(self.params.keys())

    def flatten(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Flatten parameters into a single 1D tensor."""
        flat = torch.cat([self.params[k].reshape(-1) for k in self.key_order])
        return flat.to(device) if device else flat

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> WeightSpaceObjectStateDict:
        """Create object from model named parameters."""
        params = {name: p.detach().clone() for name, p in model.named_parameters()}
        return cls(params=params, key_order=list(params.keys()))

    @classmethod
    def from_flat(
        cls,
        flat: torch.Tensor,
        template_model: torch.nn.Module,
        device: str | torch.device | None = None,
    ) -> WeightSpaceObjectStateDict:
        """Reconstruct object from flat tensor using a template model."""
        named_params = list(template_model.named_parameters())
        expected_size = sum(param.numel() for _, param in named_params)
        if flat.numel() != expected_size:
            raise ValueError(
                f"Flat tensor size {flat.numel()} doesn't match expected "
                f"size {expected_size} for template model."
            )

        params: dict[str, torch.Tensor] = {}
        offset = 0
        for name, param in named_params:
            size = param.numel()
            restored = flat[offset : offset + size].reshape(param.shape)
            params[name] = restored.to(device) if device else restored
            offset += size
        return cls(params=params, key_order=[name for name, _ in named_params])

    def apply_to_model(self, model: torch.nn.Module) -> None:
        """Copy stored parameter tensors into model parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.params:
                    raise KeyError(f"Parameter '{name}' not found in weight object.")
                param.copy_(self.params[name])

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters represented by this object."""
        return sum(self.params[k].numel() for k in self.key_order)

    def __repr__(self) -> str:
        shapes = {k: tuple(self.params[k].shape) for k in self.key_order}
        return (
            "WeightSpaceObjectStateDict("
            f"num_tensors={len(self.key_order)}, total_params={self.num_params}, "
            f"shapes={shapes})"
        )
