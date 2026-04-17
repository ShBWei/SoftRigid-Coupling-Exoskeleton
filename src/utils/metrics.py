"""Evaluation metrics for exoskeleton motion prediction.

MPJPE (Mean Per-Joint Position Error)
--------------------------------------
Standard metric in human motion prediction literature.  Given predicted and
ground-truth joint positions / angles:

    MPJPE = mean over joints of ||pred_j - gt_j||_2

When the targets are joint *angles* (not Cartesian positions) the Euclidean
norm is taken over the DoF per joint, which gives a Mean Per-Joint Angle
Error in radians.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    dof_per_joint: int = 3,
) -> torch.Tensor:
    """Mean Per-Joint Position / Angle Error.

    Parameters
    ----------
    pred          : (..., num_joints * dof_per_joint) – predicted values
    target        : same shape as pred
    dof_per_joint : degrees of freedom per joint (default 3)

    Returns
    -------
    Scalar tensor (mean over batch and time dimensions).
    """
    # (..., num_joints, dof_per_joint)
    pred_j = pred.reshape(*pred.shape[:-1], -1, dof_per_joint)
    target_j = target.reshape(*target.shape[:-1], -1, dof_per_joint)
    # Per-joint L2 norm
    per_joint_err = torch.norm(pred_j - target_j, dim=-1)  # (..., num_joints)
    return per_joint_err.mean()


def weighted_mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    dof_per_joint: int = 3,
) -> torch.Tensor:
    """Weighted MPJPE – joints can carry different importance weights.

    Parameters
    ----------
    pred, target  : (..., num_joints * dof_per_joint)
    weights       : (num_joints,) – non-negative joint weights (will be normalised)
    dof_per_joint : degrees of freedom per joint

    Returns
    -------
    Scalar tensor.
    """
    pred_j = pred.reshape(*pred.shape[:-1], -1, dof_per_joint)
    target_j = target.reshape(*target.shape[:-1], -1, dof_per_joint)
    per_joint_err = torch.norm(pred_j - target_j, dim=-1)  # (..., num_joints)
    w = weights / (weights.sum() + 1e-8)
    return (per_joint_err * w).sum(dim=-1).mean()


class MPJPELoss(nn.Module):
    """nn.Module wrapper around :func:`mpjpe`."""

    def __init__(self, dof_per_joint: int = 3) -> None:
        super().__init__()
        self.dof_per_joint = dof_per_joint

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mpjpe(pred, target, self.dof_per_joint)
