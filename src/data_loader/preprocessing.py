"""IMU data preprocessing pipeline.

Responsibilities
----------------
* Load raw CSV / NumPy files produced by the exoskeleton IMU sensors.
* Validate channel layout: 5 sensors × 6 channels = 30 channels.
* Fit a per-channel StandardScaler on training data and apply it to all splits.
* Persist the scaler to disk so that inference can reproduce the same transform.

Sensor layout (column order assumed in raw data)
-------------------------------------------------
  0- 5  : spine     [ax ay az gx gy gz]
  6-11  : left arm  [ax ay az gx gy gz]
 12-17  : right arm [ax ay az gx gy gz]
 18-23  : left leg  [ax ay az gx gy gz]
 24-29  : right leg [ax ay az gx gy gz]

Target layout (joint angles in radians, or positions in metres)
---------------------------------------------------------------
  0- 2  : spine  (roll, pitch, yaw)
  3- 5  : left arm
  6- 8  : right arm
  9-11  : left leg
 12-14  : right leg
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class IMUPreprocessor:
    """Normalise raw IMU signals to zero-mean / unit-variance per channel.

    Parameters
    ----------
    input_dim  : expected number of IMU channels (default 30)
    output_dim : expected number of joint output channels (default 15)
    """

    SCALER_FILE = "imu_scaler.pkl"

    def __init__(self, input_dim: int = 30, output_dim: int = 15) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._output_mean: Optional[np.ndarray] = None
        self._output_std: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IMUPreprocessor":
        """Compute per-channel statistics from training data.

        Parameters
        ----------
        X : (N, input_dim) or (N, T, input_dim) – raw IMU samples
        y : (N, output_dim) or (N, T, output_dim) – optional joint targets
        """
        X_flat = X.reshape(-1, self.input_dim)
        self._input_mean = X_flat.mean(axis=0)
        self._input_std = X_flat.std(axis=0) + 1e-8  # avoid division by zero

        if y is not None:
            y_flat = y.reshape(-1, self.output_dim)
            self._output_mean = y_flat.mean(axis=0)
            self._output_std = y_flat.std(axis=0) + 1e-8

        self._fitted = True
        logger.info("IMUPreprocessor fitted on %d samples.", X_flat.shape[0])
        return self

    # ------------------------------------------------------------------
    # Transform / inverse-transform
    # ------------------------------------------------------------------

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Normalise input IMU data."""
        self._check_fitted()
        return (X - self._input_mean) / self._input_std

    def inverse_transform_X(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return X * self._input_std + self._input_mean

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Normalise joint targets (if output statistics were fitted)."""
        if self._output_mean is None:
            return y
        return (y - self._output_mean) / self._output_std

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        if self._output_mean is None:
            return y
        return y * self._output_std + self._output_mean

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, self.SCALER_FILE)
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh)
        logger.info("Scaler saved to %s", path)

    @classmethod
    def load(cls, directory: str) -> "IMUPreprocessor":
        path = os.path.join(directory, cls.SCALER_FILE)
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        logger.info("Scaler loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("IMUPreprocessor must be fitted before transform is called.")

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Utility: generate synthetic IMU + joint data (for tests / demos)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    num_frames: int = 5000,
    input_dim: int = 30,
    output_dim: int = 15,
    sample_rate: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic IMU sensor data and corresponding joint angles.

    The signal consists of a mixture of sinusoids at biomechanically plausible
    frequencies (0.5 – 3 Hz) plus Gaussian noise.

    Returns
    -------
    X : (num_frames, input_dim)  – IMU channels
    y : (num_frames, output_dim) – joint angle targets (radians)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_frames) / sample_rate

    # Base frequencies for each channel
    freqs_X = rng.uniform(0.5, 3.0, size=input_dim)
    phases_X = rng.uniform(0, 2 * np.pi, size=input_dim)
    amps_X = rng.uniform(0.5, 2.0, size=input_dim)

    freqs_y = rng.uniform(0.5, 2.0, size=output_dim)
    phases_y = rng.uniform(0, 2 * np.pi, size=output_dim)
    amps_y = rng.uniform(0.1, 0.5, size=output_dim)

    X = (
        amps_X * np.sin(2 * np.pi * freqs_X * t[:, None] + phases_X)
        + rng.normal(0, 0.05, size=(num_frames, input_dim))
    )
    y = (
        amps_y * np.sin(2 * np.pi * freqs_y * t[:, None] + phases_y)
        + rng.normal(0, 0.01, size=(num_frames, output_dim))
    )
    return X.astype(np.float32), y.astype(np.float32)
