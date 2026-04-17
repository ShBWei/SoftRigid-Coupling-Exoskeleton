"""Visualisation utilities for skeleton pose and IMU sensor curves.

Functions
---------
plot_skeleton          – render a stick-figure skeleton from joint angles/positions
plot_sensor_curves     – time-series plot of IMU channels
plot_training_curves   – training / validation loss and MPJPE over epochs
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------

# Joint indices in the 5-joint model:  0=spine  1=left_arm  2=right_arm  3=left_leg  4=right_leg
JOINT_NAMES = ["Spine", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]

# (parent, child) pairs – root joint is spine (index 0)
SKELETON_EDGES = [
    (0, 1),  # spine  → left arm
    (0, 2),  # spine  → right arm
    (0, 3),  # spine  → left leg
    (0, 4),  # spine  → right leg
]

# 2-D reference positions (used when we only have angles, not Cartesian coords)
JOINT_REF_2D: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),    # spine (origin)
    1: (-0.4, 0.3),   # left arm
    2: (0.4, 0.3),    # right arm
    3: (-0.2, -0.5),  # left leg
    4: (0.2, -0.5),   # right leg
}


def plot_skeleton(
    joint_values: np.ndarray,
    dof_per_joint: int = 3,
    ax: plt.Axes | None = None,
    title: str = "Skeleton Pose",
    pred: np.ndarray | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Render a schematic 2-D stick figure.

    When Cartesian positions (dof_per_joint >= 2) are available the first two
    coordinates (x, y) are used directly.  Otherwise the reference layout is
    perturbed by the first angle of each joint for illustration.

    Parameters
    ----------
    joint_values  : (num_joints * dof_per_joint,)  – ground-truth pose
    dof_per_joint : DoF per joint
    ax            : existing matplotlib Axes (or None to create a new figure)
    title         : figure title
    pred          : optional predicted pose (same shape) drawn in a different colour
    save_path     : if given, save the figure to this path

    Returns
    -------
    matplotlib.figure.Figure
    """
    num_joints = len(joint_values) // dof_per_joint
    jv = joint_values.reshape(num_joints, dof_per_joint)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))
    else:
        fig = ax.get_figure()

    def _get_xy(jv_: np.ndarray, j: int) -> tuple[float, float]:
        if dof_per_joint >= 2:
            return float(jv_[j, 0]), float(jv_[j, 1])
        # Perturb reference position by first angle
        rx, ry = JOINT_REF_2D[j]
        angle = float(jv_[j, 0])
        return rx + 0.05 * np.cos(angle), ry + 0.05 * np.sin(angle)

    # Draw ground-truth skeleton
    for parent, child in SKELETON_EDGES:
        if parent >= num_joints or child >= num_joints:
            continue
        px, py = _get_xy(jv, parent)
        cx, cy = _get_xy(jv, child)
        ax.plot([px, cx], [py, cy], "b-o", linewidth=2, markersize=6, label="GT" if parent == 0 else "")

    # Draw predicted skeleton
    if pred is not None:
        pjv = pred.reshape(num_joints, dof_per_joint)
        for parent, child in SKELETON_EDGES:
            if parent >= num_joints or child >= num_joints:
                continue
            px, py = _get_xy(pjv, parent)
            cx, cy = _get_xy(pjv, child)
            ax.plot([px, cx], [py, cy], "r--o", linewidth=2, markersize=6, label="Pred" if parent == 0 else "")

    # Annotate joint names
    for j in range(num_joints):
        x, y = _get_xy(jv, j)
        ax.text(x, y + 0.04, JOINT_NAMES[j] if j < len(JOINT_NAMES) else str(j), ha="center", fontsize=7)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys())

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_sensor_curves(
    data: np.ndarray,
    sample_rate: int = 50,
    sensor_names: Sequence[str] | None = None,
    channel_labels: Sequence[str] | None = None,
    channels_per_sensor: int = 6,
    max_sensors: int = 5,
    title: str = "IMU Sensor Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot time-series IMU channels, grouped by sensor.

    Parameters
    ----------
    data               : (T, input_dim)  – raw or normalised IMU data
    sample_rate        : Hz (used for x-axis)
    sensor_names       : list of sensor labels (default: Spine, L-Arm, R-Arm, L-Leg, R-Leg)
    channel_labels     : list of 6 per-sensor channel names
    channels_per_sensor: channels per sensor (default 6)
    max_sensors        : number of sensors to plot
    title              : figure title
    save_path          : if given, save the figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    T = data.shape[0]
    t = np.arange(T) / sample_rate

    if sensor_names is None:
        sensor_names = ["Spine", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]
    if channel_labels is None:
        channel_labels = ["ax", "ay", "az", "gx", "gy", "gz"]

    n_sensors = min(max_sensors, data.shape[1] // channels_per_sensor)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 2.5 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, channels_per_sensor))

    for i in range(n_sensors):
        ax = axes[i]
        start_ch = i * channels_per_sensor
        for k in range(channels_per_sensor):
            ch_idx = start_ch + k
            if ch_idx >= data.shape[1]:
                break
            ax.plot(t, data[:, ch_idx], color=colors[k], linewidth=0.8,
                    label=channel_labels[k] if k < len(channel_labels) else str(k))
        ax.set_ylabel(sensor_names[i] if i < len(sensor_names) else f"Sensor {i}")
        ax.legend(loc="upper right", fontsize=7, ncol=channels_per_sensor)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_mpjpe: Sequence[float] | None = None,
    val_mpjpe: Sequence[float] | None = None,
    title: str = "Training Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot loss (and optionally MPJPE) curves over training epochs.

    Parameters
    ----------
    train_losses : per-epoch training loss
    val_losses   : per-epoch validation loss
    train_mpjpe  : (optional) per-epoch training MPJPE
    val_mpjpe    : (optional) per-epoch validation MPJPE
    title        : figure title
    save_path    : if given, save the figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    has_mpjpe = (train_mpjpe is not None) and (val_mpjpe is not None)
    n_rows = 2 if has_mpjpe else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax.plot(epochs, val_losses, label="Val Loss", color="darkorange")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if has_mpjpe:
        ax2 = axes[1]
        ax2.plot(epochs, train_mpjpe, label="Train MPJPE", color="steelblue")
        ax2.plot(epochs, val_mpjpe, label="Val MPJPE", color="darkorange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MPJPE (rad / m)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("Epoch")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
