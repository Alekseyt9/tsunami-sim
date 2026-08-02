"""Host-side fitting used when a released deformable fragment becomes rigid.

The expensive per-substep motion remains on CUDA.  Conversion is deliberately
rare and uses a double-precision least-squares fit so that the initial rigid
body preserves the fragment's mass, centre of mass and linear/angular momentum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidClusterFit:
    mass: float
    center: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    inverse_inertia: np.ndarray
    local_positions: np.ndarray
    internal_velocity_rms: float


def fit_rigid_cluster(position: np.ndarray, velocity: np.ndarray, mass: np.ndarray) -> RigidClusterFit:
    """Fit the momentum-equivalent rigid motion of one particle cluster."""
    x = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    total_mass = float(m.sum())
    if len(x) < 2 or total_mass <= 0.0:
        raise ValueError("a rigid cluster needs at least two positive-mass particles")

    center = (m[:, None] * x).sum(axis=0) / total_mass
    linear_velocity = (m[:, None] * v).sum(axis=0) / total_mass
    local = x - center
    relative_velocity = v - linear_velocity
    inertia = np.zeros((3, 3), dtype=np.float64)
    angular_momentum = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for r, dv, particle_mass in zip(local, relative_velocity, m):
        inertia += particle_mass * (float(np.dot(r, r)) * identity - np.outer(r, r))
        angular_momentum += np.cross(r, particle_mass * dv)

    # Planar panels legitimately have a small principal inertia.  A tiny
    # trace-relative regularizer avoids a singular inverse without changing
    # resolved axes in any meaningful way.
    regularizer = max(float(np.trace(inertia)) * 1.0e-9, 1.0e-8)
    regularized = inertia + identity * regularizer
    inverse_inertia = np.linalg.inv(regularized)
    angular_velocity = inverse_inertia @ angular_momentum
    rigid_velocity = linear_velocity + np.cross(angular_velocity[None, :], local)
    residual = v - rigid_velocity
    internal_velocity_rms = float(np.sqrt((m * np.sum(residual * residual, axis=1)).sum() / total_mass))

    return RigidClusterFit(
        mass=total_mass,
        center=center.astype(np.float32),
        linear_velocity=linear_velocity.astype(np.float32),
        angular_velocity=angular_velocity.astype(np.float32),
        inverse_inertia=inverse_inertia.astype(np.float32),
        local_positions=local.astype(np.float32),
        internal_velocity_rms=internal_velocity_rms,
    )
