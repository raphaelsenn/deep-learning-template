from typing import Callable, Any
from pathlib import Path

from torchvision.datasets import MNIST
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(
            self, 
            root: str | Path,
            train: bool = True,
            transform: Callable | None = None
    ) -> None:
        super().__init__()
        self.dataset = MNIST(root, train, transform, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: Any) -> Any:
        return self.dataset[index]
