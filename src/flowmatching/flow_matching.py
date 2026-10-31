"""Flow Matching training and generation for weight space.

This module implements the core Flow Matching algorithm for learning
to generate neural network weights. Based on the approach described in
"DeepWeightFlow: Re-Basined Flow Matching for Generating Neural Network Weights"
(https://arxiv.org/pdf/2601.05052).
"""

from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.flowmatching.flow_model import Bunch
from src.utils.logger import logger


class FlowMatching:
    """Flow Matching trainer for weight space generation.

    Implements Conditional Flow Matching (CFM) for learning to transport
    samples from a source distribution (Gaussian noise) to a target
    distribution (trained model weights).

    The flow is defined by the linear interpolation path:
        x_t = (1 - t) * x_0 + t * x_1
    where x_0 ~ source (noise), x_1 ~ target (trained weights).

    The velocity field u_t = x_1 - x_0 is learned by the model.

    Attributes:
        model: Neural network that predicts the velocity field.
        device: Device for computation.
        mode: Prediction mode ('velocity' or 'target').
        t_dist: Time distribution ('uniform' or 'beta').
        sigma: Noise level for the flow.
        best_loss: Best training loss achieved.
        best_model_state: State dict of model at best loss.
        input_dim: Dimension of weight space (set after training).

    Example:
        >>> fm = FlowMatching(
        ...     targetloader=weight_loader,
        ...     model=flow_model,
        ...     source_std=0.01,
        ...     device='cuda'
        ... )
        >>> fm.train(n_iters=10000, optimizer=optimizer)
        >>> generated_weights = fm.generate_weights(n_samples=100)
    """

    def __init__(
            self,
            targetloader: DataLoader,
            model: nn.Module,
            mode: Literal["velocity", "target"] = "velocity",
            t_dist: Literal["uniform", "beta"] = "uniform",
            device: str | torch.device | None = None,
            source_std: float = 0.01,
            sigma: float = 0.001,
        ) -> None:
        """Initialize FlowMatching.

        Args:
            targetloader: DataLoader for target distribution (trained weights).
            model: Neural network model for predicting velocity.
            mode: Prediction mode - 'velocity' predicts u_t, 'target' predicts x_1.
            t_dist: Time distribution - 'uniform' or 'beta'.
            device: Device for computation.
            source_std: Std of Gaussian source noise sampled each iteration.
            sigma: Noise level added to interpolated points.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.mode = mode
        self.t_dist = t_dist
        self.source_std = source_std
        self.sigma = sigma

        self.best_loss = float("inf")
        self.best_model_state: dict | None = None
        self.input_dim: int | None = None

        # Initialize iterators
        self._target_iter: iter | None = None

    def _sample_from_loader(self, loader: DataLoader) -> torch.Tensor:
        """Sample a batch from dataloader.

        Args:
            loader: DataLoader to sample from.

        Returns:
            Batch tensor on the correct device.
        """
        try:
            iterator = self._target_iter
            if iterator is None:
                iterator = iter(loader)
                self._target_iter = iterator
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            self._target_iter = iterator
            batch = next(iterator)

        # Handle TensorDataset (returns tuple)
        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        return batch.to(self.device)

    def sample_time_and_flow(self) -> Bunch:
        """Sample time t and compute flow data for training.

        Returns:
            Bunch object containing:
                - t: Time values (batch_size, 1)
                - x0: Source samples (batch_size, dim)
                - xt: Interpolated points (batch_size, dim)
                - x1: Target samples (batch_size, dim)
                - ut: Velocity (x1 - x0) (batch_size, dim)
                - eps: Noise added (batch_size, dim)
                - batch_size: Batch size
        """
        x1 = self._sample_from_loader(self.targetloader)
        x0 = torch.randn_like(x1) * self.source_std
        batch_size = x1.size(0)

        # Sample time
        if self.t_dist == "beta":
            # Beta distribution biased towards t=0
            alpha, beta_param = 2.0, 5.0
            t = torch.distributions.Beta(alpha, beta_param).sample((batch_size,))
            t = t.to(self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        # Expand t for broadcasting
        t_expanded = t.view(-1, *([1] * (x0.dim() - 1)))

        # Linear interpolation path: x_t = (1-t)*x_0 + t*x_1
        mu_t = (1 - t_expanded) * x0 + t_expanded * x1

        # Add small noise
        epsilon = torch.randn_like(x0) * self.sigma
        xt = mu_t + epsilon

        # Velocity field: u_t = x_1 - x_0
        ut = x1 - x0

        return Bunch(
            t=t.unsqueeze(-1),
            x0=x0,
            xt=xt,
            x1=x1,
            ut=ut,
            eps=epsilon,
            batch_size=batch_size,
        )

    def forward(self, flow: Bunch) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            flow: Bunch object with xt and t.

        Returns:
            Model prediction (velocity or target).
        """
        return self.model(flow.xt, flow.t)

    def loss_fn(self, pred: torch.Tensor, flow: Bunch) -> torch.Tensor:
        """Compute loss for flow matching.

        Args:
            pred: Model prediction.
            flow: Bunch object with target values.

        Returns:
            MSE loss between prediction and target.
        """
        if self.mode == "target":
            target = flow.x1
        else:
            target = flow.ut

        return torch.mean((pred.squeeze() - target) ** 2)

    def train(
            self,
            n_iters: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
            patience: int = 1000000,
            log_freq: int = 100,
        ) -> None:
        """Train the flow model.

        Args:
            n_iters: Number of training iterations.
            optimizer: Optimizer for model parameters.
            scheduler: Optional learning rate scheduler.
            patience: Early stopping patience (iterations without improvement).
            log_freq: Frequency of logging training progress.
        """
        self.model.train()
        last_loss = float("inf")
        patience_count = 0

        logger.info(f"Starting Flow Matching training for {n_iters} iterations")

        for i in range(n_iters):
            # Sample and compute loss
            flow = self.sample_time_and_flow()
            pred = self.forward(flow)
            loss = self.loss_fn(pred, flow)

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at iteration {i}: {loss.item()}")
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # Track best model
            loss_value = loss.item()
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            # Early stopping check
            if loss_value > last_loss:
                patience_count += 1
                if patience_count >= patience:
                    logger.info(f"Early stopping at iteration {i}")
                    break
            else:
                patience_count = 0

            last_loss = loss_value

            # Logging
            if i % log_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Iter {i}/{n_iters} | Loss: {loss_value:.6f} | "
                    f"Best: {self.best_loss:.6f} | LR: {lr:.2e}"
                )

        logger.info(f"Training complete. Best loss: {self.best_loss:.6f}")

    def map(
            self,
            x0: torch.Tensor,
            n_steps: int = 100,
            method: Literal["euler", "rk4"] = "euler",
            return_trajectory: bool = False,
        ) -> torch.Tensor | list[torch.Tensor]:
        """Map source samples to target distribution via ODE integration.

        Integrates the learned velocity field from t=0 to t=1 to transform
        source samples (noise) into target samples (trained weights).

        Args:
            x0: Source samples of shape (n_samples, dim).
            n_steps: Number of integration steps.
            method: Integration method ('euler' or 'rk4').
            return_trajectory: If True, return list of intermediate states.

        Returns:
            Generated samples at t=1, or trajectory if return_trajectory=True.
        """
        # Load best model if available
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        batch_size = x0.size(0)
        xt = x0.clone().to(self.device)

        trajectory = [xt.detach().clone()] if return_trajectory else None

        times = torch.linspace(0, 1, n_steps, device=self.device)
        dt = times[1] - times[0]

        with torch.no_grad():
            for i, t in enumerate(times[:-1]):
                t_tensor = torch.ones(batch_size, 1, device=self.device) * t

                if method == "euler":
                    vt = self.model(xt, t_tensor)
                    if vt.dim() > 2:
                        vt = vt.squeeze(-1)
                    xt = xt + vt * dt

                elif method == "rk4":
                    # Runge-Kutta 4th order
                    k1 = self.model(xt, t_tensor)
                    if k1.dim() > 2:
                        k1 = k1.squeeze(-1)

                    k2 = self.model(xt + 0.5 * dt * k1, t_tensor + 0.5 * dt)
                    if k2.dim() > 2:
                        k2 = k2.squeeze(-1)

                    k3 = self.model(xt + 0.5 * dt * k2, t_tensor + 0.5 * dt)
                    if k3.dim() > 2:
                        k3 = k3.squeeze(-1)

                    k4 = self.model(xt + dt * k3, t_tensor + dt)
                    if k4.dim() > 2:
                        k4 = k4.squeeze(-1)

                    xt = xt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                if return_trajectory:
                    trajectory.append(xt.detach().clone())

        # Restore current model state
        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()

        return trajectory if return_trajectory else xt

    def generate_weights(
            self,
            n_samples: int,
            source_std: float = 0.01,
            n_steps: int = 100,
            method: Literal["euler", "rk4"] = "rk4",
        ) -> torch.Tensor:
        """Generate new weight samples.

        Args:
            n_samples: Number of weight samples to generate.
            source_std: Standard deviation of source noise.
            n_steps: Number of ODE integration steps.
            method: Integration method.

        Returns:
            Generated weights of shape (n_samples, input_dim).

        Raises:
            ValueError: If input_dim is not set.
        """
        if self.input_dim is None:
            raise ValueError(
                "input_dim must be set before generating weights. "
                "Set it manually or train the model first."
            )

        source_samples = (
            torch.randn(n_samples, self.input_dim, device=self.device) * source_std
        )
        return self.map(source_samples, n_steps=n_steps, method=method)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.best_model_state or self.model.state_dict(),
            "best_loss": self.best_loss,
            "input_dim": self.input_dim,
            "mode": self.mode,
            "t_dist": self.t_dist,
            "source_std": self.source_std,
            "sigma": self.sigma,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_model_state = checkpoint["model_state_dict"]
        self.best_loss = checkpoint["best_loss"]
        self.input_dim = checkpoint.get("input_dim")
        self.mode = checkpoint.get("mode", "velocity")
        self.t_dist = checkpoint.get("t_dist", "uniform")
        self.source_std = checkpoint.get("source_std", 0.01)
        self.sigma = checkpoint.get("sigma", 0.001)
        logger.info(f"Checkpoint loaded from {path}")
