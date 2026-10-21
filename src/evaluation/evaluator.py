from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


class TemplateModelEvaluator:
    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.results = {}

    @torch.inference_mode()
    def predict(self, dataloader: DataLoader):
        self.model.eval()
        y_true, y_prob = [], []

        for x, y in dataloader:
            logits = self.model(x.to(self.device))
            y_true.append(y.cpu().numpy())
            y_prob.append(logits.softmax(dim=1).cpu().numpy())

        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        y_pred = y_prob.argmax(axis=1)

        return y_true, y_pred, y_prob

    def evaluate(self, dataloader: DataLoader) -> dict:
        y_true, y_pred, y_prob = self.predict(dataloader)
        labels = np.arange(y_prob.shape[1])

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "roc_auc_ovr": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels),
            "roc_auc_ovo": roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro", labels=labels),
            "log_loss": log_loss(y_true, y_prob, labels=labels),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        }

        self.results = {
            "metrics": pd.DataFrame([metrics]),
            "classification_report": pd.DataFrame(
                classification_report(
                    y_true, y_pred, labels=labels,
                    output_dict=True, zero_division=0
                )
            ).T,
            "confusion_matrix": pd.DataFrame(
                confusion_matrix(y_true, y_pred, labels=labels),
                index=labels, columns=labels
            ),
        }

        return metrics

    def save(self, directory: str = "checkpoints") -> None:
        if not self.results:
            raise RuntimeError("Run evaluate() before save().")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "TemplateModel.pt")

        for name, df in self.results.items():
            df.to_csv(path / f"{name}.csv", index=False)
