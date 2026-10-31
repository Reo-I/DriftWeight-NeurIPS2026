"""Weight data generator for creating trained model checkpoints."""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import GenerateConfig
from src.utils.logger import logger


class WeightGenerator:
    """Generator class for creating trained model weight data.

    Generates model checkpoints at specified iteration intervals during
    the training process. Used to create datasets of model weights.

    Attributes:
        model: PyTorch model to generate weights for.
        config: Generation configuration.
        device: Device to use for computation.
        criterion: Loss function.
        optimizer: Optimizer for training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: GenerateConfig,
        seed: int,
    ) -> None:
        """Initialize the WeightGenerator.

        Args:
            model: PyTorch model to generate weights for.
            config: Generation configuration.
            seed: Random seed for this generation run.
        """
        self.model = model
        self.config = config
        self.seed = seed
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None
        if config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
            )
        elif config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
            )

        self._global_iteration = 0
        self._seed_checkpoint_dir = self.config.checkpoint_dir / f"seed{seed:04d}"
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self) -> None:
        """Create checkpoint directory for this seed if it doesn't exist."""
        self._seed_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, iteration: int, epoch: int, loss: float) -> None:
        """Save model checkpoint.

        Args:
            iteration: Current global iteration number.
            epoch: Current epoch number.
            loss: Current loss value.
        """
        checkpoint_path = self._seed_checkpoint_dir / f"checkpoint_iter{iteration}.pt"
        checkpoint = {
            "iteration": iteration,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "seed": self.seed,
            "config": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "optimizer": self.config.optimizer,
                "hidden_dim": self.config.model.hidden_dim,
                "init_type": self.config.model.init_type,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _run_epoch(self, data_loader: DataLoader, epoch: int) -> float:
        """Run one epoch of weight generation.

        Args:
            data_loader: DataLoader for training data.
            epoch: Current epoch number.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(data_loader)

        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self._global_iteration += 1

            # Save checkpoint at specified intervals
            if self._global_iteration % self.config.save_every_n_iterations == 0:
                self._save_checkpoint(
                    iteration=self._global_iteration,
                    epoch=epoch,
                    loss=loss.item(),
                )

        return total_loss / num_batches

    def _evaluate(self, data_loader: DataLoader) -> tuple[float, float]:
        """Evaluate model accuracy.

        Args:
            data_loader: DataLoader for test data.

        Returns:
            Tuple of (average_loss, accuracy_percentage).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def generate(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> dict[str, float]:
        """Run weight data generation.

        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for test data.

        Returns:
            Dictionary containing final metrics.
        """
        logger.info(f"Starting weight generation for seed={self.seed}")

        best_accuracy = 0.0

        for epoch in range(1, self.config.num_epochs + 1):
            avg_loss = self._run_epoch(train_loader, epoch)
            test_loss, accuracy = self._evaluate(test_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            logger.info(
                f"Seed {self.seed} | Epoch {epoch}/{self.config.num_epochs} | "
                f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%"
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy

        # Save final checkpoint
        self._save_checkpoint(
            iteration=self._global_iteration,
            epoch=self.config.num_epochs,
            loss=avg_loss,
        )

        # Save final model weights
        final_path = self._seed_checkpoint_dir / "final.pt"
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"Final weights saved: {final_path}")

        return {
            "seed": self.seed,
            "final_loss": avg_loss,
            "best_accuracy": best_accuracy,
        }
