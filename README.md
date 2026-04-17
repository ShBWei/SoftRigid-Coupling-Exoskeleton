# GangRouBingJi Exoskeleton
**刚柔并"脊"——新一代万向刚柔耦合仿生外骨骼运动预判系统**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

&gt; Internet+ Competition Project | Real-time human motion prediction for soft-rigid coupled exoskeletons

**GitHub Repository**: `https://github.com/ShBWei/gangrou-bingji-exoskeleton`

## Overview

A **physics-guided deep learning system** for real-time prediction of spinal posture and rigidity states in segmented exoskeletons. 

**Input**: 400ms historical IMU streams from 4-6 sensor nodes (spine + limbs)  
**Output**: 200ms future spine posture (3D angles) + stiffness switching commands (soft/rigid mode)

**Core Innovations:**
- **Segmented Biomimetic Spine**: Cervical/Thoracic/Lumbar/Sacral 4-section articulated structure
- **Soft-Rigid Coupling**: Dual-mode control prediction (adaptive flexibility vs. rigid locking)
- **Dual Attention Mechanism**: Temporal attention + Cross-segment biomechanical constraints
- **Physics-Informed**: Anatomically feasible joint limits (0-150°) and spine kinematic chains

## Quick Start

```bash
# Clone repo
git clone https://github.com/ShBWei/gangrou-bingji-exoskeleton.git
cd gangrou-bingji-exoskeleton

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data and train
python train.py

# View results
# Check generated: gangrou_bingji_results.png
