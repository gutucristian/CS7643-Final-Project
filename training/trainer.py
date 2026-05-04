# training loop for all three model architectures

import numpy as np
import torch
from torch.utils.data import DataLoader


def resolve_device(device_str="auto"):
    if device_str != "auto":
        return device_str

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


class Trainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.device = torch.device(resolve_device(device))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device) if hasattr(criterion, "to") else criterion

    def train(self, dataloader, epochs=10, val_dataloader=None):
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for windows, labels, fwd_returns in dataloader:
                windows = windows.to(self.device)
                labels = labels.to(self.device)
                fwd_returns = fwd_returns.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(windows)
                loss = self.criterion(logits, labels, fwd_returns)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(labels)

            avg_loss = total_loss / len(dataloader.dataset)
            history["train_loss"].append(avg_loss)

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={avg_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f}")
            else:
                print(f"Epoch {epoch:3d}/{epochs} | train_loss={avg_loss:.4f}")

        return history

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for windows, labels, fwd_returns in dataloader:
                windows = windows.to(self.device)
                labels = labels.to(self.device)
                fwd_returns = fwd_returns.to(self.device)

                logits = self.model(windows)
                loss = self.criterion(logits, labels, fwd_returns)

                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def predict(self, dataloader):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for windows, labels, fwd_returns in dataloader:
                windows = windows.to(self.device)
                logits = self.model(windows)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds)
