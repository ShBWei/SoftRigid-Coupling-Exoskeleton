#!/usr/bin/env python3
"""Real-time inference demo for the exoskeleton motion predictor.

Simulates a live IMU data stream and runs the model frame-by-frame,
displaying a continuously updated skeleton pose and sensor plot.

Usage
-----
    python scripts/demo.py --config configs/config.yaml
    python scripts/demo.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
    python scripts/demo.py --config configs/config.yaml --save-gif demo_output/demo.gif
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import numpy as np
import torch
import yaml

from src.data_loader.preprocessing import IMUPreprocessor, generate_synthetic_data
from src.models.lstm_model import LSTMPredictor
from src.models.transformer_model import TransformerPredictor
from src.utils.logger import get_logger
from src.utils.metrics import mpjpe
from src.utils.visualization import JOINT_NAMES, JOINT_REF_2D, SKELETON_EDGES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exoskeleton motion prediction demo")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--save-gif", default=None, help="Save animation as GIF to this path")
    p.add_argument("--no-display", action="store_true", help="Disable interactive window")
    return p.parse_args()


def build_model(cfg: dict) -> torch.nn.Module:
    model_type = cfg["model"]["type"]
    if model_type == "lstm":
        return LSTMPredictor.from_config(cfg)
    elif model_type == "transformer":
        return TransformerPredictor.from_config(cfg)
    raise ValueError(f"Unknown model type: {model_type}")


def _draw_skeleton_on_ax(
    ax,
    joints: np.ndarray,
    num_joints: int,
    dof_per_joint: int,
    color: str = "b",
    label: str = "",
) -> None:
    """Draw a stick-figure onto an existing Axes object."""
    jv = joints.reshape(num_joints, dof_per_joint)

    def xy(j: int) -> tuple[float, float]:
        if dof_per_joint >= 2:
            return float(jv[j, 0]), float(jv[j, 1])
        rx, ry = JOINT_REF_2D.get(j, (0.0, 0.0))
        a = float(jv[j, 0])
        return rx + 0.05 * np.cos(a), ry + 0.05 * np.sin(a)

    first = True
    for parent, child in SKELETON_EDGES:
        if parent >= num_joints or child >= num_joints:
            continue
        px, py = xy(parent)
        cx, cy = xy(child)
        lbl = label if first else ""
        ax.plot([px, cx], [py, cy], f"{color}-o", linewidth=2, markersize=6, label=lbl)
        first = False

    for j in range(num_joints):
        x, y = xy(j)
        ax.text(x, y + 0.04, JOINT_NAMES[j] if j < len(JOINT_NAMES) else str(j),
                ha="center", fontsize=7)


def run_demo(
    cfg: dict,
    model: torch.nn.Module,
    preprocessor: IMUPreprocessor,
    device: torch.device,
    save_gif: str | None,
    no_display: bool,
    logger,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    data_cfg = cfg["data"]
    demo_cfg = cfg["demo"]

    sample_rate = data_cfg["sample_rate"]
    input_dim = data_cfg["input_dim"]
    output_dim = data_cfg["output_dim"]
    num_joints = data_cfg["num_joints"]
    dof_per_joint = data_cfg["dof_per_joint"]
    window_size = int(data_cfg["window_seconds"] * sample_rate)
    sim_len = int(demo_cfg["sim_length_seconds"] * sample_rate)

    # Generate simulated sensor stream
    X_raw, y_raw = generate_synthetic_data(
        num_frames=sim_len + window_size,
        input_dim=input_dim,
        output_dim=output_dim,
        sample_rate=sample_rate,
        seed=42,
    )

    X_stream = preprocessor.transform_X(X_raw)
    y_stream = preprocessor.transform_y(y_raw)

    # Pre-compute predictions for the whole stream
    model.eval()
    preds_list: list[np.ndarray] = []
    mpjpe_list: list[float] = []

    with torch.no_grad():
        for t in range(window_size, sim_len + window_size):
            window = torch.from_numpy(X_stream[t - window_size : t]).unsqueeze(0).to(device)
            pred = model(window)
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = y_stream[t]
            preds_list.append(pred_np)
            err = float(mpjpe(pred[0, 0], torch.from_numpy(gt_np).to(device)).item())
            mpjpe_list.append(err)

    preds_arr = np.array(preds_list)
    gt_arr = y_stream[window_size:]
    time_arr = np.arange(sim_len) / sample_rate

    logger.info("Demo – mean MPJPE over stream: %.4f", float(np.mean(mpjpe_list)))

    # ---- Animation ----
    if no_display:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")

    fig = plt.figure(figsize=(14, 6))
    ax_skel = fig.add_subplot(1, 3, 1)
    ax_curve = fig.add_subplot(1, 3, 2)
    ax_mpjpe = fig.add_subplot(1, 3, 3)

    # Number of frames shown in the sensor window
    window_plot = min(window_size, 100)

    def update(frame: int):
        ax_skel.cla()
        ax_curve.cla()
        ax_mpjpe.cla()

        # Skeleton
        gt_joints = preprocessor.inverse_transform_y(gt_arr[frame])
        pred_joints = preprocessor.inverse_transform_y(preds_arr[frame])
        _draw_skeleton_on_ax(ax_skel, gt_joints, num_joints, dof_per_joint, "b", "GT")
        _draw_skeleton_on_ax(ax_skel, pred_joints, num_joints, dof_per_joint, "r", "Pred")
        ax_skel.set_title(f"Skeleton  t={time_arr[frame]:.2f}s")
        ax_skel.set_aspect("equal")
        ax_skel.legend(fontsize=8)
        ax_skel.set_xlim(-0.8, 0.8)
        ax_skel.set_ylim(-0.8, 0.8)

        # Sensor curves (recent window)
        t_start = max(0, frame + window_size - window_plot)
        t_end = frame + window_size
        raw_clip = X_stream[t_start:t_end, :6]  # spine only (6 channels)
        t_clip = np.arange(raw_clip.shape[0]) / sample_rate
        ch_labels = ["ax", "ay", "az", "gx", "gy", "gz"]
        for k in range(6):
            ax_curve.plot(t_clip, raw_clip[:, k], linewidth=0.8, label=ch_labels[k])
        ax_curve.set_title("Spine IMU (recent)")
        ax_curve.set_xlabel("Time (s)")
        ax_curve.legend(fontsize=7, ncol=2)
        ax_curve.grid(True, alpha=0.3)

        # MPJPE over time
        ax_mpjpe.plot(time_arr[: frame + 1], mpjpe_list[: frame + 1], color="darkgreen")
        ax_mpjpe.set_title("MPJPE over time")
        ax_mpjpe.set_xlabel("Time (s)")
        ax_mpjpe.set_ylabel("MPJPE")
        ax_mpjpe.grid(True, alpha=0.3)

        fig.tight_layout()

    n_frames = sim_len
    interval_ms = max(1, int(1000 / demo_cfg["fps"]))
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)

    if save_gif:
        os.makedirs(os.path.dirname(save_gif) or ".", exist_ok=True)
        logger.info("Saving GIF to %s …", save_gif)
        ani.save(save_gif, writer=PillowWriter(fps=demo_cfg["fps"]))
        logger.info("GIF saved.")

    if not no_display:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device(cfg["inference"]["device"])
    logger = get_logger("demo")

    # ---- Preprocessing ----
    processed_dir = cfg["data"]["processed_dir"]
    scaler_path = os.path.join(processed_dir, IMUPreprocessor.SCALER_FILE)
    if os.path.exists(scaler_path):
        preprocessor = IMUPreprocessor.load(processed_dir)
    else:
        logger.warning("Scaler not found, fitting on synthetic data for demo.")
        X_raw, y_raw = generate_synthetic_data(
            num_frames=5_000,
            input_dim=cfg["data"]["input_dim"],
            output_dim=cfg["data"]["output_dim"],
            sample_rate=cfg["data"]["sample_rate"],
        )
        preprocessor = IMUPreprocessor(
            input_dim=cfg["data"]["input_dim"],
            output_dim=cfg["data"]["output_dim"],
        ).fit(X_raw, y_raw)

    # ---- Model ----
    model = build_model(cfg).to(device)
    ckpt_path = args.checkpoint or cfg["inference"]["checkpoint"]
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint: %s", ckpt_path)
    else:
        logger.warning("No checkpoint found at %s – using random weights for demo.", ckpt_path)

    run_demo(
        cfg=cfg,
        model=model,
        preprocessor=preprocessor,
        device=device,
        save_gif=args.save_gif,
        no_display=args.no_display,
        logger=logger,
    )


if __name__ == "__main__":
    main()
