#!/usr/bin/env python3
"""Evaluation script – compute MPJPE on the held-out test split.

Usage
-----
    python scripts/eval.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from src.data_loader.dataset import build_dataloaders_from_config
from src.data_loader.preprocessing import IMUPreprocessor, generate_synthetic_data
from src.models.lstm_model import LSTMPredictor
from src.models.transformer_model import TransformerPredictor
from src.utils.logger import get_logger
from src.utils.metrics import mpjpe
from src.utils.visualization import plot_skeleton, plot_sensor_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate exoskeleton motion predictor")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pth)")
    p.add_argument("--output-dir", default="eval_outputs", help="Where to save eval figures")
    p.add_argument("--no-synthetic", action="store_true")
    return p.parse_args()


def build_model(cfg: dict) -> torch.nn.Module:
    model_type = cfg["model"]["type"]
    if model_type == "lstm":
        return LSTMPredictor.from_config(cfg)
    elif model_type == "transformer":
        return TransformerPredictor.from_config(cfg)
    raise ValueError(f"Unknown model type: {model_type}")


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device(cfg["inference"]["device"])
    logger = get_logger("eval")

    # ---- Data ----
    data_dir = cfg["data"]["data_dir"]
    X_path = os.path.join(data_dir, "imu_data.npy")
    y_path = os.path.join(data_dir, "joint_targets.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        X_raw = np.load(X_path).astype(np.float32)
        y_raw = np.load(y_path).astype(np.float32)
    else:
        if args.no_synthetic:
            raise FileNotFoundError(f"Data not found at {data_dir}")
        logger.warning("Real data not found. Using synthetic data.")
        X_raw, y_raw = generate_synthetic_data(
            num_frames=5_000,
            input_dim=cfg["data"]["input_dim"],
            output_dim=cfg["data"]["output_dim"],
            sample_rate=cfg["data"]["sample_rate"],
            seed=0,
        )

    # ---- Preprocessing ----
    processed_dir = cfg["data"]["processed_dir"]
    scaler_path = os.path.join(processed_dir, IMUPreprocessor.SCALER_FILE)
    if os.path.exists(scaler_path):
        preprocessor = IMUPreprocessor.load(processed_dir)
    else:
        logger.warning("Scaler not found, fitting on evaluation data (sub-optimal).")
        preprocessor = IMUPreprocessor(
            input_dim=cfg["data"]["input_dim"],
            output_dim=cfg["data"]["output_dim"],
        ).fit(X_raw, y_raw)

    X = preprocessor.transform_X(X_raw)
    y = preprocessor.transform_y(y_raw)

    _, _, test_loader = build_dataloaders_from_config(X, y, cfg)
    logger.info("Test batches: %d", len(test_loader))

    # ---- Model ----
    model = build_model(cfg).to(device)
    ckpt_path = args.checkpoint or cfg["inference"]["checkpoint"]
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", ckpt_path, ckpt.get("epoch", -1))
    else:
        logger.warning("Checkpoint not found at %s – evaluating with random weights.", ckpt_path)

    model.eval()

    # ---- Evaluation ----
    all_mpjpe = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            all_mpjpe.append(mpjpe(pred, y_batch).item())
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    mean_mpjpe = float(np.mean(all_mpjpe))
    std_mpjpe = float(np.std(all_mpjpe))
    logger.info("Test MPJPE: %.4f ± %.4f", mean_mpjpe, std_mpjpe)

    # ---- Visualisation ----
    os.makedirs(args.output_dir, exist_ok=True)

    preds_arr = np.concatenate(all_preds, axis=0)   # (N, P, output_dim)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Plot skeleton for the first sample
    sample_pred = preds_arr[0, 0]    # first sample, first predicted frame
    sample_gt = targets_arr[0, 0]
    # Inverse-transform for physical units
    sample_pred_phys = preprocessor.inverse_transform_y(sample_pred)
    sample_gt_phys = preprocessor.inverse_transform_y(sample_gt)

    fig = plot_skeleton(
        sample_gt_phys,
        dof_per_joint=cfg["data"]["dof_per_joint"],
        pred=sample_pred_phys,
        title="Test Sample – Skeleton Pose",
        save_path=os.path.join(args.output_dir, "skeleton_pose.png"),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    # Plot sensor curves for a short clip
    clip_len = min(250, X.shape[0])
    fig2 = plot_sensor_curves(
        X[:clip_len],
        sample_rate=cfg["data"]["sample_rate"],
        title="Normalised IMU Sensor Curves (eval clip)",
        save_path=os.path.join(args.output_dir, "sensor_curves.png"),
    )
    plt.close(fig2)

    logger.info("Saved visualisations to %s", args.output_dir)
    print(f"\nTest MPJPE: {mean_mpjpe:.4f} ± {std_mpjpe:.4f}")


if __name__ == "__main__":
    main()
