from typing import List, Dict
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TrainStats:
    train_accuracy_scores_: List[float] = field(default_factory=list)
    val_accuracy_scores_: List[float] = field(default_factory=list)
    
    train_losses_: List[float] = field(default_factory=list)
    val_losses_: List[float] = field(default_factory=list)

    curr_incumbent_: float = float("-inf") 
    incumbent_trajectory_: List[float] = field(default_factory=list)

    def update(self, train_acc: float, val_acc: float, train_loss: float, val_loss: float) -> None:
        self.train_accuracy_scores_.append(train_acc)
        self.val_accuracy_scores_.append(val_acc)
        self.train_losses_.append(train_loss)
        self.val_losses_.append(val_loss)

        self.curr_incumbent_ = max(self.curr_incumbent_, val_acc)
        self.incumbent_trajectory_.append(self.curr_incumbent_)

    def to_csv(self, path: str) -> None:
        data = {
            "train_accuracy_scores": self.train_accuracy_scores_,
            "val_accuracy_scores": self.val_accuracy_scores_,
            "train_losses": self.train_losses_,
            "val_losses": self.val_losses_,
            "incumbent_trajectory": self.incumbent_trajectory_,
        }
        pd.DataFrame.from_dict(data).to_csv(path, index=False)

    @property
    def last_train_acc(self) -> float:
        return self.train_accuracy_scores_[-1]

    @property
    def last_val_acc(self) -> float:
        return self.val_accuracy_scores_[-1]
    
    @property
    def last_train_loss(self) -> float:
        return self.train_losses_[-1]

    @property
    def last_val_loss(self) -> float:
        return self.val_losses_[-1]

    @property
    def latest(self) -> Dict[str, float]:
        return {
            "train_acc": self.train_accuracy[-1],
            "val_acc": self.val_accuracy[-1],
            "train_loss": self.train_loss[-1],
            "val_loss": self.val_loss[-1],
        }
