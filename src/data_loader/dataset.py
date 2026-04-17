"""IMU dataset with sliding-window segmentation.

Given a continuous stream of (normalised) IMU frames, the dataset produces
pairs of:
  * ``X`` : input window  – shape (window_size, input_dim)
  * ``y`` : target window – shape (predict_frames, output_dim)

The windows are extracted with a configurable stride (step_size) so that
consecutive windows overlap.

Usage example
-------------
>>> from src.data_loader.preprocessing import generate_synthetic_data, IMUPreprocessor
>>> from src.data_loader.dataset import IMUDataset, build_dataloaders
>>> X_raw, y_raw = generate_synthetic_data(num_frames=10_000)
>>> pre = IMUPreprocessor().fit(X_raw, y_raw)
>>> X_norm = pre.transform_X(X_raw)
>>> y_norm = pre.transform_y(y_raw)
>>> train_dl, val_dl, test_dl = build_dataloaders(X_norm, y_norm,
...                                               window_size=100,
...                                               predict_frames=1,
...                                               step_size=5,
...                                               batch_size=64)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class IMUDataset(Dataset):
    """Sliding-window IMU dataset.

    Parameters
    ----------
    X            : (N, input_dim)  – normalised IMU frames
    y            : (N, output_dim) – normalised joint targets
    window_size  : number of historical frames per sample
    predict_frames : number of future frames to predict
    step_size    : stride between consecutive windows
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 100,
        predict_frames: int = 1,
        step_size: int = 5,
    ) -> None:
        super().__init__()
        assert X.shape[0] == y.shape[0], "X and y must have the same number of frames."
        self.window_size = window_size
        self.predict_frames = predict_frames
        self.step_size = step_size

        # Pre-compute all windows for fast __getitem__
        total = X.shape[0]
        starts = range(0, total - window_size - predict_frames + 1, step_size)
        indices = list(starts)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        x_window = self.X[start : start + self.window_size]          # (W, input_dim)
        y_window = self.y[
            start + self.window_size : start + self.window_size + self.predict_frames
        ]  # (P, output_dim)
        return x_window, y_window


def build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 100,
    predict_frames: int = 1,
    step_size: int = 5,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders from continuous IMU arrays.

    Parameters
    ----------
    X, y         : normalised sensor and target arrays (N, *)
    window_size  : history window in frames
    predict_frames : future frames to predict
    step_size    : sliding-window stride
    batch_size   : mini-batch size
    train_ratio  : fraction of windows used for training
    val_ratio    : fraction used for validation (remainder → test)
    num_workers  : DataLoader worker processes
    seed         : random seed for deterministic split

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    dataset = IMUDataset(
        X, y, window_size=window_size, predict_frames=predict_frames, step_size=step_size
    )
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False)


def build_dataloaders_from_config(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience wrapper that reads parameters from the config dict."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    sample_rate = data_cfg["sample_rate"]
    window_size = int(data_cfg["window_seconds"] * sample_rate)
    step_size = max(1, int(data_cfg["step_seconds"] * sample_rate))
    return build_dataloaders(
        X,
        y,
        window_size=window_size,
        predict_frames=data_cfg["predict_frames"],
        step_size=step_size,
        batch_size=train_cfg["batch_size"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=train_cfg["seed"],
    )
