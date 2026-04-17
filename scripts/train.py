#!/usr/bin/env python3
"""Training script for the exoskeleton motion prediction model.

Usage
-----
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --model transformer
    python scripts/train.py --config configs/config.yaml --resume checkpoints/last_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure the project root is on sys.path when called as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.data_loader.dataset import build_dataloaders_from_config
from src.data_loader.preprocessing import IMUPreprocessor, generate_synthetic_data
from src.models.lstm_model import LSTMPredictor
from src.models.transformer_model import TransformerPredictor
from src.utils.logger import get_logger
from src.utils.metrics import MPJPELoss, mpjpe
from src.utils.visualization import plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train exoskeleton motion predictor")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p.add_argument("--model", choices=["lstm", "transformer"], default=None,
                   help="Override model type from config")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--data-dir", default=None, help="Override data directory")
    p.add_argument("--no-synthetic", action="store_true",
                   help="Disable synthetic data fallback (fail if real data not found)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_model(cfg: dict) -> torch.nn.Module:
    model_type = cfg["model"]["type"]
    if model_type == "lstm":
        return LSTMPredictor.from_config(cfg)
    elif model_type == "transformer":
        return TransformerPredictor.from_config(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_data(cfg: dict, logger, no_synthetic: bool = False):
    """Load raw data from disk or fall back to synthetic generation."""
    data_dir = cfg["data"]["data_dir"]
    input_dim = cfg["data"]["input_dim"]
    output_dim = cfg["data"]["output_dim"]
    sample_rate = cfg["data"]["sample_rate"]

    X_path = os.path.join(data_dir, "imu_data.npy")
    y_path = os.path.join(data_dir, "joint_targets.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        logger.info("Loading data from %s", data_dir)
        X_raw = np.load(X_path).astype(np.float32)
        y_raw = np.load(y_path).astype(np.float32)
    else:
        if no_synthetic:
            raise FileNotFoundError(
                f"Data files not found at {data_dir} and --no-synthetic is set."
            )
        logger.warning(
            "Real data not found at %s. Generating synthetic data for demonstration.", data_dir
        )
        X_raw, y_raw = generate_synthetic_data(
            num_frames=10_000,
            input_dim=input_dim,
            output_dim=output_dim,
            sample_rate=sample_rate,
            seed=cfg["training"]["seed"],
        )

    return X_raw, y_raw


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    cfg: dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg,
        },
        path,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    clip_grad: float,
    log_interval: int,
    logger,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    n_batches = 0

    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        batch_mpjpe = mpjpe(pred.detach(), y_batch).item()
        total_loss += loss.item()
        total_mpjpe += batch_mpjpe
        n_batches += 1
        global_step += 1

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | Loss %.4f | MPJPE %.4f",
                epoch, batch_idx + 1, len(loader), loss.item(), batch_mpjpe,
            )
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/batch_mpjpe", batch_mpjpe, global_step)

    return total_loss / max(n_batches, 1), total_mpjpe / max(n_batches, 1), global_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch)
        total_loss += criterion(pred, y_batch).item()
        total_mpjpe += mpjpe(pred, y_batch).item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mpjpe / max(n_batches, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model is not None:
        cfg["model"]["type"] = args.model
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir

    # Reproducibility
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg["training"]["device"])
    log_dir = cfg["training"]["log_dir"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]

    logger = get_logger("train", log_dir=log_dir)
    logger.info("Config: %s", cfg)
    logger.info("Device: %s", device)

    # Data
    X_raw, y_raw = load_data(cfg, logger, no_synthetic=args.no_synthetic)
    logger.info("Data shapes: X=%s  y=%s", X_raw.shape, y_raw.shape)

    # Preprocessing
    preprocessor = IMUPreprocessor(
        input_dim=cfg["data"]["input_dim"],
        output_dim=cfg["data"]["output_dim"],
    )
    preprocessor.fit(X_raw, y_raw)
    if cfg["data"]["normalize"]:
        X = preprocessor.transform_X(X_raw)
        y = preprocessor.transform_y(y_raw)
    else:
        X, y = X_raw, y_raw

    preprocessor.save(cfg["data"]["processed_dir"])

    # DataLoaders
    train_loader, val_loader, _ = build_dataloaders_from_config(X, y, cfg)
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | Parameters: %d", cfg["model"]["type"], n_params)

    # Optimiser & scheduler
    train_cfg = cfg["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    if train_cfg["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["num_epochs"]
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg["lr_step_size"],
            gamma=train_cfg["lr_gamma"],
        )

    criterion = MPJPELoss(dof_per_joint=cfg["data"]["dof_per_joint"])

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        logger.info("Resumed from %s (epoch %d)", args.resume, ckpt["epoch"])

    writer = SummaryWriter(log_dir=log_dir)

    # Training history for visualisation
    train_losses, val_losses, train_mpjpes, val_mpjpes = [], [], [], []
    patience_counter = 0
    global_step = 0

    for epoch in range(start_epoch, train_cfg["num_epochs"] + 1):
        t0 = time.time()
        train_loss, train_m, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            train_cfg["clip_grad_norm"], train_cfg["log_interval"],
            logger, epoch, writer, global_step,
        )

        val_loss, val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d | Train Loss %.4f | Val Loss %.4f | Train MPJPE %.4f | Val MPJPE %.4f | %.1fs",
            epoch, train_cfg["num_epochs"], train_loss, val_loss, train_m, val_m, elapsed,
        )

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("mpjpe", {"train": train_m, "val": val_m}, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mpjpes.append(train_m)
        val_mpjpes.append(val_m)

        # Checkpoint: always save latest
        save_checkpoint(
            os.path.join(ckpt_dir, "last_model.pth"),
            model, optimizer, epoch, best_val_loss, cfg,
        )

        # Checkpoint: save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pth"),
                model, optimizer, epoch, best_val_loss, cfg,
            )
            logger.info("  ↑ New best model saved (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                logger.info("Early stopping triggered after %d epochs without improvement.", epoch)
                break

    # Save training curves figure
    os.makedirs(log_dir, exist_ok=True)
    fig = plot_training_curves(
        train_losses, val_losses, train_mpjpes, val_mpjpes,
        save_path=os.path.join(log_dir, "training_curves.png"),
    )
    plt_close(fig)
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    writer.close()


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
