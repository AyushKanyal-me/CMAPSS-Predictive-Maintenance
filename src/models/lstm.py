"""PyTorch LSTM utilities for sequence-to-RUL modeling."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..evaluation.metrics import regression_metrics


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lightweight dataset wrapper for sequence features and RUL targets."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.ndim != 3:
            raise ValueError(f"features must be 3D (samples, timesteps, features), got {features.shape}.")
        if targets.ndim != 1:
            raise ValueError(f"targets must be 1D (samples,), got {targets.shape}.")
        if len(features) != len(targets):
            raise ValueError(
                f"features and targets must have the same number of samples. Got {len(features)} and {len(targets)}."
            )

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class LSTMRegressor(nn.Module):
    """Many-to-one LSTM regressor that predicts RUL from a sequence window."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be a positive integer.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        projection_size = max(hidden_size // 2, 16)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        last_hidden = outputs[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


@dataclass
class LSTMTrainingResult:
    """Container for trained LSTM model and tracked training metadata."""

    model: LSTMRegressor
    history: pd.DataFrame
    best_epoch: int
    best_validation_metrics: dict[str, float]
    validation_predictions: np.ndarray


def _run_epoch(
    model: LSTMRegressor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    cumulative_loss = 0.0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            outputs = model(features)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        cumulative_loss += float(loss.item()) * features.size(0)
        predictions.append(outputs.detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(predictions).astype(np.float32, copy=False)
    y_true = np.concatenate(targets).astype(np.float32, copy=False)

    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = cumulative_loss / len(loader.dataset)
    return metrics, y_pred


def train_lstm_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 30,
    weight_decay: float = 1e-5,
    patience: int = 5,
    random_state: int = 42,
    device: str | None = None,
) -> LSTMTrainingResult:
    """Train an LSTM regressor with optional early stopping."""
    if x_train.ndim != 3 or x_validation.ndim != 3:
        raise ValueError("x_train and x_validation must be 3D arrays.")
    if y_train.ndim != 1 or y_validation.ndim != 1:
        raise ValueError("y_train and y_validation must be 1D arrays.")
    if x_train.shape[0] != y_train.shape[0] or x_validation.shape[0] != y_validation.shape[0]:
        raise ValueError("Sequence arrays and target arrays must have matching sample counts.")
    if x_train.shape[2] != x_validation.shape[2]:
        raise ValueError("Training and validation sequences must have the same feature dimension.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if patience < 0:
        raise ValueError("patience must be >= 0.")

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        SequenceDataset(x_validation, y_validation),
        batch_size=batch_size,
        shuffle=False,
    )

    model = LSTMRegressor(
        input_size=x_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(resolved_device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history_rows: list[dict[str, float]] = []
    best_epoch = 0
    best_validation_metrics: dict[str, float] = {}
    best_validation_predictions = np.empty((0,), dtype=np.float32)
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics, _ = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=resolved_device,
            optimizer=optimizer,
        )
        validation_metrics, validation_predictions = _run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=resolved_device,
            optimizer=None,
        )

        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_rmse": train_metrics["rmse"],
                "validation_loss": validation_metrics["loss"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_mae": validation_metrics["mae"],
                "validation_nasa_score": validation_metrics["nasa_score"],
            }
        )

        if validation_metrics["rmse"] < best_rmse:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            best_validation_predictions = validation_predictions.copy()
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if patience > 0 and stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)
    return LSTMTrainingResult(
        model=model,
        history=history_df,
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
        validation_predictions=best_validation_predictions,
    )


def predict_lstm(
    model: LSTMRegressor,
    sequences: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """Run batched inference for precomputed sequence windows."""
    if sequences.ndim != 3:
        raise ValueError("sequences must be a 3D array (samples, timesteps, features).")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(resolved_device)
    model.eval()

    loader = DataLoader(SequenceDataset(sequences, np.zeros(len(sequences), dtype=np.float32)), batch_size=batch_size)
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for features, _ in loader:
            outputs = model(features.to(resolved_device))
            predictions.append(outputs.cpu().numpy())

    return np.concatenate(predictions).astype(np.float32, copy=False)