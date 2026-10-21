from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.statistics import TrainStats


class TemplateModelTrainer:
    def __init__(
            self, 
            model: nn.Module,
            n_epochs: int,
            learning_rate: float,
            momentum: float,
            device: str = "cpu"
    ) -> None:
        assert device in {"cpu", "cuda", "mps"}, (
            f"Unkdown device, expected device in [`cpu`, `cuda`, `mps`], got: {device}."
        )
        assert isinstance(n_epochs, int) and n_epochs > 0, (
            f"n_epochs should be a positiv integer, got: {n_epochs}"
        )
        assert isinstance(learning_rate, float) and learning_rate > 0, (
            f"learning_rate should be a positiv (> 0.0) float, got: {learning_rate}"
        )
        assert isinstance(momentum, float) and momentum >= 0, (
            f"momentum should be a positiv (>= 0.0) float, got: {momentum}"
        )

        self.device = torch.device(device)
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate, momentum)
        
        # Best achived accuracy on validation set
        self.curr_incumbent = -float("inf")
        self.stats = TrainStats()

    def train_one_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_correct, total_samples, total_loss = 0, 0, 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward propagation and loss computation
            y_batch_logits = self.model(x_batch)
            y_batch_preds = torch.argmax(y_batch_logits, dim=-1)
            loss = self.criterion(y_batch_logits, y_batch)

            # Backpropagation and SGD update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track stats
            batch_size = x_batch.size(0)
            total_samples += batch_size
            total_loss += float(loss.item() * batch_size)
            total_correct += int((y_batch == y_batch_preds).sum().item())

        mean_accuracy = total_correct / total_samples
        mean_cross_entropy = total_loss / total_samples

        return mean_accuracy, mean_cross_entropy

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_correct, total_samples, total_loss = 0, 0, 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward propagation and loss computation
            y_batch_logits = self.model(x_batch)
            y_batch_preds = torch.argmax(y_batch_logits, dim=-1)
            loss = self.criterion(y_batch_logits, y_batch)

            # Track stats
            batch_size = x_batch.size(0)
            total_samples += batch_size
            total_loss += float(loss.item() * batch_size)
            total_correct += int((y_batch == y_batch_preds).sum().item())

        mean_accuracy = total_correct / total_samples
        mean_cross_entropy = total_loss / total_samples

        return mean_accuracy, mean_cross_entropy

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        for epoch in range(self.n_epochs):
            train_acc_approx, train_loss_approx = self.train_one_epoch(train_loader)

            train_acc, train_loss = self.evaluate(train_loader)
            val_acc, val_loss = self.evaluate(val_loader)
            self.stats.update(train_acc, val_acc, train_loss, val_loss)

            if val_acc > self.curr_incumbent:
                self.curr_incumbent_acc = val_acc
                self.save()

    def save(self, directory: str = "checkpoints") -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "TemplateModel.pt")
        self.stats.to_csv(path / "TrainingStats.csv")
