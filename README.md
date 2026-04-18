# SoftRigid Coupling Exoskeleton
**刚柔并"脊"——新一代万向刚柔耦合仿生外骨骼运动预判系统**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> Internet+ Competition Project | Real-time human motion prediction for soft-rigid coupled exoskeletons

## Overview

A **physics-guided deep learning system** for real-time prediction of spinal posture and rigidity states in segmented exoskeletons.

- **Input**: 8 IMU streams @60Hz, 24-frame (400ms) observation window
- **Output**: 12-frame (200ms) future prediction of posture + mode + switch probability
- **Response target**: delay under 200ms (aligned with plan section 3.2.4)

## 8 IMU Configuration (Plan 3.2.2)

- Count: `8`
- Locations: cervical, left/right shoulder, left/right elbow, left/right hip, left knee
- Channel per IMU: `6` (3-axis acc + 3-axis gyro)
- Input tensor: `8 x 6 x 24 = 1152` dimensions

## Four-Segment Spine Structure

```text
[Cervical x7] -- [Thoracic x12] -- [Lumbar x5] -- [Sacral fused]
   C1-C7             T1-T12            L1-L5         fixed base
```

```mermaid
graph LR
  C[Cervical] --> T[Thoracic]
  T --> L[Lumbar]
  L --> S[Sacral]
```

## Scenario Data and Validation Reference

- Reference cohort: **200 subjects**
- Reported effect baseline: **ROM +42%**, **burden -65%**
- Scenario distribution:
  - Walk/flexible mode: 60%
  - Lift/rigid mode: 20%
  - Fall-protection transition: 10%
  - Rest/mixed mode: 10%

## Model Outputs

- Posture head: `23 x 3` joint angles
- Mode head: rigid/flexible classification (`0/1`)
- Switch head: switching probability (`0-1`)
- Physics layer: differentiable soft ROM constraints + hard clipping

## Training Metrics

- MPJPE (pose error proxy)
- Mode-switch accuracy
- Ratio of response delay `<200ms`
- Best checkpoint selected by validation switch F1

## Quick Start

```bash
pip install -r requirements.txt
python main.py --samples 300 --epochs 3 --repeat_seeds 1 --batch_size 16 --device cpu
```

## Latest Run (Local Verified)

已在本机完成一次端到端运行（轻量配置：`samples=300, epochs=3, repeat_seeds=1`），并成功生成模型、图表、论文素材文件。

- 数据分布: walking/scoliosis/squat/fall = 180/30/60/30（60/10/20/10）
- 物理合规率: 100.00%
- ROM 违规率: 0.00%
- 生成文件:
  - `medical_data.npz`
  - `baseline_best.pth`, `ours_best.pth`
  - `learning_curves.png`
  - `fig1_anatomical_violation_cases.png`
  - `fig2_fall_prevention_timeline.png`
  - `fig3_coupling_heatmap.png`
  - `table_comparison.tex`
  - `discussion_auto.txt`

### Comparison Snapshot

| Experiment | Metric | Baseline | Ours |
|---|---:|---:|---:|
| A (normal->normal) | MPJPE | 3.2440 | 2.8866 |
| A (normal->normal) | AVR | 0.0000 | 0.0000 |
| B (normal->scoliosis) | MPJPE | 3.8515 | 3.3494 |
| B (normal->scoliosis) | AVR | 0.0000 | 0.0000 |
| C (danger action) | MPJPE | 2.5817 | 2.7552 |
| C (danger action) | InjuryPreventionRate | 0.0000 | 0.0000 |

> 注: 上述是快速可运行验证，主要用于确认“代码整体连通 + 结果可落盘”。论文版结果建议使用 `repeat_seeds=5` 和更长训练轮数（如 `epochs>=30`）获取稳定显著性统计。
