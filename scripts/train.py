from typing import Tuple
import argparse
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src import (
    TemplateModel, 
    TemplateModelTrainer,
    TemplateModelEvaluator,
    MNISTDataset
)


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST LeNet-5 Style Training")

    # Dataset
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train_frac", type=int, default=0.9)

    # Model
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=10)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps")

    return parser.parse_args()


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


def main():
    args = parse_args()
    set_seed(args.seed)

    # Datasets
    train_set = load_dataset(args.root, train=True)
    test_set = load_dataset(args.root, train=False)
    train_set, val_set = split_train_val_dataset(train_set, args.train_frac, args.seed)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = TemplateModel(args.in_channels, args.hidden_dim, args.num_classes)
    
    # Training
    trainer = TemplateModelTrainer(
        model=model,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        momentum=args.momentum,
        device=args.device
    )
    trainer.train(train_loader, val_loader)

    # Evaluate on test
    evaluator = TemplateModelEvaluator(model, args.device)
    evaluator.evaluate(test_loader)
    evaluator.save()


if __name__ == "__main__":
    main()
