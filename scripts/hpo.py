from typing import Tuple
import argparse
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision import transforms

import optuna
from optuna import Trial

from src import TemplateModel, TemplateModelTrainer, MNISTDataset


# Dataset configuration
ROOT = "./data/"
TRAIN_FRAC = 0.9         # Fraction of data used for training
N_WORKERS = 1            # Number of DataLoader worker processes

# Model architecture
IN_CHANNELS = 1          # Number of input channels
HIDDEN_DIM = 256         # Number of hidden features
OUT_DIM = 10             # Number of output classes (digits 0–9)

# Hyperparameter optimization
N_TRIALS = 25            # Number of Optuna trials
N_EPOCHS = 5             # Number of training epochs per trial

# Reproducibility and hardware
SEED = 42                # Random seed for reproducibility
DEVICE = "cuda" if torch.cuda.is_available() else "mps"


def load_dataset(root: str, train: bool = True) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNISTDataset(root=root, train=train, transform=transform)
    return dataset


def split_train_val_dataset(dataset: Dataset, train_frac: int, seed: int) -> Tuple[Dataset, Dataset]:
    train_size = int(train_frac * len(dataset))

    rng = np.random.default_rng(seed=seed)
    indices = torch.from_numpy(rng.permutation(len(dataset)))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def objective(trial: Trial) -> float:
    set_seed(SEED)

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    model = TemplateModel(IN_CHANNELS, HIDDEN_DIM, OUT_DIM)
    trainer = TemplateModelTrainer(model, N_EPOCHS, learning_rate, momentum, DEVICE)

    # Loading the dataset
    dataset = load_dataset(ROOT, train=True)
    train_set, val_set = split_train_val_dataset(dataset, TRAIN_FRAC, SEED)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=N_WORKERS)

    for epoch in range(N_EPOCHS):
        trainer.train_one_epoch(train_loader)
        val_accuracy, _ = trainer.evaluate(val_loader)
        trial.report(val_accuracy, step=epoch)

    return float(val_accuracy)


def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=N_TRIALS)

    print("Best trial:", study.best_trial.number)
    print("Best accuracy:", study.best_value)
    print("Best parameters:", study.best_params)


if __name__ == "__main__":
    main()
