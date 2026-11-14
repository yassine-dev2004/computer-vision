"""Landmark features and the landmark MLP, shared by training and live inference."""
import numpy as np
import torch
from torch import nn

_PAIRS = torch.triu_indices(21, 21, offset=1)  # 210 joint pairs
FEATURE_DIM = 2 * 63 + 210


def _normalize(points):
    """(B, 21, 3) -> wrist at the origin, wrist-to-middle-knuckle distance = 1."""
    points = points - points[:, :1]
    return points / points[:, 9].norm(dim=-1).clamp_min(1e-6)[:, None, None]


def _random_transforms(batch, device, max_rot_deg=15.0, scale_jitter=0.1):
    """Random mirror, in-plane rotation and x/y stretch, as (B, 3, 3) matrices."""
    angle = (torch.rand(batch, device=device) * 2 - 1) * np.deg2rad(max_rot_deg)
    cos, sin = angle.cos(), angle.sin()
    rot = torch.zeros(batch, 3, 3, device=device)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = cos, -sin, sin, cos, 1
    mirror = torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
    stretch = 1 + (torch.rand(batch, 2, device=device) * 2 - 1) * scale_jitter
    diag = torch.stack([mirror * stretch[:, 0], stretch[:, 1], torch.ones(batch, device=device)], 1)
    return rot @ torch.diag_embed(diag)


def landmark_features(image_lm, world_lm, augment=False, noise=0.02):
    """Image landmarks (pixel units) and world landmarks (metres), both (B, 21, 3)
    -> (B, 336): normalized coordinates of both + all pairwise joint distances.
    Orientation is kept on purpose (P vs K, Q vs G differ only by orientation);
    left/right hands are handled by training on mirrored copies instead."""
    image_lm, world_lm = _normalize(image_lm), _normalize(world_lm)
    if augment:
        transform = _random_transforms(image_lm.shape[0], image_lm.device).transpose(1, 2)
        image_lm = image_lm @ transform + noise * torch.randn_like(image_lm)
        world_lm = world_lm @ transform + noise * torch.randn_like(world_lm)
    i, j = _PAIRS.to(image_lm.device)
    distances = (image_lm[:, i] - image_lm[:, j]).norm(dim=-1)
    return torch.cat([image_lm.flatten(1), world_lm.flatten(1), distances], dim=1)


class LandmarkMLP(nn.Module):
    def __init__(self, num_classes, in_dim=FEATURE_DIM, hidden=512, dropout=0.3):
        super().__init__()

        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.GELU(), nn.Dropout(dropout)]

        self.net = nn.Sequential(nn.BatchNorm1d(in_dim), *block(in_dim, hidden), *block(hidden, hidden),
                                 *block(hidden, hidden // 2), nn.Linear(hidden // 2, num_classes))

    def forward(self, x):
        return self.net(x)
