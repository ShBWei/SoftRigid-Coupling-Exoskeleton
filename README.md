# 刚柔并脊外骨骼运动预测 · Gangrou-Bingji Exoskeleton Motion Prediction

A sequence-to-sequence deep learning pipeline for predicting next-frame joint
motion from IMU sensor data mounted on a rigid-flexible spine exoskeleton.

---

## Features

| Feature | Detail |
|---|---|
| **Sensor layout** | Spine + 4 limbs (left/right arm & leg) – 5 × 6 channels = 30-D input |
| **Models** | LSTM encoder-decoder and Transformer encoder-decoder (switchable via config) |
| **Preprocessing** | Per-channel StandardScaler, sliding-window segmentation |
| **Loss / metric** | MPJPE (Mean Per-Joint Position/Angle Error) |
| **Checkpointing** | Best & latest checkpoint saved after each epoch |
| **Logging** | Console + file logging; TensorBoard integration |
| **Visualisation** | 2-D skeleton pose, IMU sensor time-series, training curves, real-time demo animation |

---

## Repository Structure

```
gangrou-bingji-exoskeleton/
├── configs/
│   └── config.yaml          ← All hyperparameters (model, data, training)
├── data/
│   ├── raw/                 ← Place imu_data.npy & joint_targets.npy here
│   └── processed/           ← Normalisation scaler saved here
├── scripts/
│   ├── train.py             ← Training entry-point
│   ├── eval.py              ← Evaluation / MPJPE on test split
│   └── demo.py              ← Real-time inference simulation + animation
├── src/
│   ├── models/
│   │   ├── lstm_model.py        ← LSTM encoder-decoder
│   │   └── transformer_model.py ← Transformer encoder-decoder
│   ├── data_loader/
│   │   ├── dataset.py           ← Sliding-window PyTorch Dataset
│   │   └── preprocessing.py     ← IMU normalisation pipeline
│   └── utils/
│       ├── metrics.py           ← MPJPE loss
│       ├── visualization.py     ← Skeleton & sensor plots
│       └── logger.py            ← Logging helpers
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ShBWei/gangrou-bingji-exoskeleton.git
cd gangrou-bingji-exoskeleton

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **CUDA note** – `requirements.txt` installs the default PyTorch wheel (CPU-capable).
> To enable GPU training install the matching CUDA wheel from https://pytorch.org/get-started/locally/
> and set `training.device: cuda` in `configs/config.yaml`.

---

## Data Format

Place your raw recordings in `data/raw/`:

| File | Shape | Description |
|---|---|---|
| `imu_data.npy` | `(N, 30)` float32 | Raw IMU frames (5 sensors × 6 channels) |
| `joint_targets.npy` | `(N, 15)` float32 | Joint angles/positions (5 joints × 3 DoF) |

**Channel order per sensor:** `ax  ay  az  gx  gy  gz`  
**Sensor order:** spine · left arm · right arm · left leg · right leg

If the files are absent the scripts automatically generate synthetic data for
demonstration purposes.

---

## Quick Start

### Training

```bash
# LSTM model (default)
python scripts/train.py --config configs/config.yaml

# Transformer model
python scripts/train.py --config configs/config.yaml --model transformer

# Resume from a checkpoint
python scripts/train.py --config configs/config.yaml --resume checkpoints/last_model.pth
```

TensorBoard logs are written to `runs/`.  Launch with:

```bash
tensorboard --logdir runs
```

### Evaluation

```bash
python scripts/eval.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output-dir eval_outputs
```

Outputs a per-joint MPJPE score and saves skeleton + sensor-curve figures to
`eval_outputs/`.

### Real-time Demo

```bash
# Interactive animated window (requires a display)
python scripts/demo.py --config configs/config.yaml

# Headless – save as GIF
python scripts/demo.py --config configs/config.yaml \
    --no-display --save-gif demo_output/demo.gif
```

---

## Configuration

All hyperparameters live in `configs/config.yaml`.  Key sections:

```yaml
data:
  sample_rate: 50          # Hz
  window_seconds: 2.0      # history window fed to the model
  step_seconds: 0.1        # sliding-window stride
  predict_frames: 1        # future frames to predict
  input_dim: 30            # 5 sensors × 6 channels
  output_dim: 15           # 5 joints × 3 DoF

model:
  type: lstm               # "lstm" or "transformer"
  lstm:
    hidden_dim: 256
    num_layers: 2
  transformer:
    d_model: 128
    nhead: 8

training:
  device: cpu              # "cpu" or "cuda"
  batch_size: 64
  num_epochs: 100
  learning_rate: 1.0e-3
  early_stopping_patience: 20
```

---

## Model Architectures

### LSTM Encoder-Decoder
```
IMU window (B, T, 30)
    → LSTMEncoder  (bidirectional option, multi-layer)
    → context (h_n, c_n)
    → LSTMDecoder  (autoregressive, seeded with last input frame)
    → predicted joints (B, P, 15)
```

### Transformer Encoder-Decoder
```
IMU window (B, T, 30)
    → Linear projection + Positional Encoding
    → Transformer Encoder  (N × self-attention)
    → memory
    → Transformer Decoder  (autoregressive, causal mask)
    → Linear projection
    → predicted joints (B, P, 15)
```

---

## License

[MIT](LICENSE)
