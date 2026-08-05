"""Host-side fitting used when a released deformable fragment becomes rigid.

The expensive per-substep motion remains on CUDA.  Conversion is deliberately
rare and uses a double-precision least-squares fit so that the initial rigid
body preserves the fragment's mass, centre of mass and linear/angular momentum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.scene import STRUCT_GLASS, STRUCT_SLAB, STRUCT_WALL


@dataclass(frozen=True)
class RigidClusterFit:
    mass: float
    center: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    inverse_inertia: np.ndarray
    local_positions: np.ndarray
    internal_velocity_rms: float
    reference_position_rms: float


@dataclass(frozen=True)
class RigidCollisionProxy:
    """Body-local convex OBB enclosing the fragment's physical samples."""

    local_center: np.ndarray
    half_extent: np.ndarray
    material: int


def limit_rigid_release_motion(
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    mass: float,
    half_extent: np.ndarray,
    maximum_linear_speed: float,
    maximum_upward_speed: float,
    upward_speed_reference_mass: float,
    minimum_mass_upward_speed: float,
    maximum_angular_speed: float,
    maximum_tip_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the runtime rigid limits before the first rigid substep.

    Conversion used to upload the unconstrained deformable fit and clamp it
    only in ``integrate_rigid_bodies``.  A newly converted heavy slab could
    therefore be rendered and collide once with an impossible release impulse.
    This host equivalent keeps the transition frame inside the same mass- and
    size-aware bounds as every later CUDA substep.
    """
    linear = np.asarray(linear_velocity, dtype=np.float64).copy()
    angular = np.asarray(angular_velocity, dtype=np.float64).copy()
    body_mass = max(float(mass), 1.0)
    if maximum_upward_speed > 0.0:
        upward_limit = float(maximum_upward_speed)
        if upward_speed_reference_mass > 0.0:
            upward_limit *= np.sqrt(
                float(upward_speed_reference_mass)
                / max(body_mass, float(upward_speed_reference_mass))
            )
            upward_limit = max(float(minimum_mass_upward_speed), upward_limit)
        linear[1] = min(float(linear[1]), upward_limit)
    linear_speed = float(np.linalg.norm(linear))
    if maximum_linear_speed > 0.0 and linear_speed > maximum_linear_speed:
        linear *= float(maximum_linear_speed) / linear_speed

    angular_speed = float(np.linalg.norm(angular))
    angular_limit = float(maximum_angular_speed)
    if maximum_tip_speed > 0.0:
        body_radius = max(float(np.linalg.norm(half_extent)), 0.25)
        size_limit = float(maximum_tip_speed) / body_radius
        angular_limit = size_limit if angular_limit <= 0.0 else min(angular_limit, size_limit)
    if angular_limit > 0.0 and angular_speed > angular_limit:
        angular *= angular_limit / angular_speed
    return linear.astype(np.float32), angular.astype(np.float32)


def fit_rigid_collision_proxy(
    local_positions: np.ndarray,
    radius: np.ndarray,
    material: np.ndarray,
    mass: np.ndarray | None = None,
    padding_scale: float = 1.0,
    normal_axis: np.ndarray | None = None,
    structural_class: np.ndarray | None = None,
    maximum_sheet_thickness: float | None = None,
) -> RigidCollisionProxy:
    """Fit the eight-vertex convex proxy used after rigid conversion.

    The box lives in the fitted body's local coordinates, so later rotation is
    exact. Particle radii are included in the envelope; the underlying samples
    remain present for water/deformable coupling and reactivation fallback.
    """
    local = np.asarray(local_positions, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    material = np.asarray(material, dtype=np.int32)
    if len(local) == 0 or len(radius) != len(local) or len(material) != len(local):
        raise ValueError("collision proxy needs equally sized non-empty particle arrays")
    padding = max(float(np.median(radius)) * float(padding_scale), 1.0e-4)
    lower = np.min(local, axis=0) - padding
    upper = np.max(local, axis=0) + padding
    local_center = 0.5 * (lower + upper)
    half_extent = np.maximum(0.5 * (upper - lower), padding)
    weights = np.ones(len(material), dtype=np.float64) if mass is None else np.asarray(mass, dtype=np.float64)
    if len(weights) != len(material):
        raise ValueError("collision proxy material weights must match the particle arrays")
    maximum_material = max(3, int(np.max(material)))
    dominant = int(np.argmax(np.bincount(material, weights=weights, minlength=maximum_material + 1)))
    if (
        maximum_sheet_thickness is not None
        and normal_axis is not None
        and structural_class is not None
    ):
        axes = np.asarray(normal_axis, dtype=np.int32)
        roles = np.asarray(structural_class, dtype=np.int32)
        if len(axes) != len(local) or len(roles) != len(local):
            raise ValueError("sheet proxy metadata must match the particle arrays")
        sheet = np.isin(roles, (STRUCT_WALL, STRUCT_GLASS, STRUCT_SLAB)) & (axes >= 0) & (axes < 3)
        sheet_weight = float(np.sum(weights[sheet]))
        total_weight = float(np.sum(weights))
        if sheet_weight >= 0.80 * max(total_weight, 1.0e-12):
            axis_weight = np.bincount(axes[sheet], weights=weights[sheet], minlength=3)
            axis = int(np.argmax(axis_weight))
            center_span = float(np.ptp(local[sheet, axis]))
            thickness = max(float(maximum_sheet_thickness), 0.04)
            # Only collapse a genuinely planar fragment.  A malformed cluster
            # containing two parallel walls must retain an enclosing proxy
            # rather than silently losing collision coverage between them.
            if axis_weight[axis] >= 0.80 * sheet_weight and center_span <= thickness:
                local_center[axis] = 0.5 * (
                    float(np.min(local[sheet, axis])) + float(np.max(local[sheet, axis]))
                )
                half_extent[axis] = 0.5 * thickness
    return RigidCollisionProxy(
        local_center=local_center.astype(np.float32),
        half_extent=half_extent.astype(np.float32),
        material=dominant,
    )


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return an xyzw quaternion for a proper 3x3 rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.asarray(
                [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale]
            )
        elif axis == 1:
            scale = np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.asarray(
                [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale]
            )
        else:
            scale = np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quaternion = np.asarray(
                [(matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                 (matrix[1, 0] - matrix[0, 1]) / scale]
            )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quaternion / norm).astype(np.float32)


def _mass_properties(local: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    identity = np.eye(3, dtype=np.float64)
    inertia = np.zeros((3, 3), dtype=np.float64)
    for radius, particle_mass in zip(local, mass):
        inertia += particle_mass * (
            float(np.dot(radius, radius)) * identity - np.outer(radius, radius)
        )
    regularizer = max(float(np.trace(inertia)) * 1.0e-9, 1.0e-8)
    return inertia, np.linalg.inv(inertia + identity * regularizer)


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
    angular_momentum = np.zeros(3, dtype=np.float64)
    for r, dv, particle_mass in zip(local, relative_velocity, m):
        angular_momentum += np.cross(r, particle_mass * dv)

    # Planar panels legitimately have a small principal inertia.  A tiny
    # trace-relative regularizer avoids a singular inverse without changing
    # resolved axes in any meaningful way.
    _, inverse_inertia = _mass_properties(local, m)
    angular_velocity = inverse_inertia @ angular_momentum
    rigid_velocity = linear_velocity + np.cross(angular_velocity[None, :], local)
    residual = v - rigid_velocity
    internal_velocity_rms = float(np.sqrt((m * np.sum(residual * residual, axis=1)).sum() / total_mass))

    return RigidClusterFit(
        mass=total_mass,
        center=center.astype(np.float32),
        orientation=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        linear_velocity=linear_velocity.astype(np.float32),
        angular_velocity=angular_velocity.astype(np.float32),
        inverse_inertia=inverse_inertia.astype(np.float32),
        local_positions=local.astype(np.float32),
        internal_velocity_rms=internal_velocity_rms,
        reference_position_rms=0.0,
    )


def fit_rigid_cluster_to_reference(
    position: np.ndarray,
    rest_position: np.ndarray,
    velocity: np.ndarray,
    mass: np.ndarray,
) -> RigidClusterFit:
    """Fit momentum to the best rigid pose of a fragment's undeformed shape.

    A released deformable panel can be heavily stretched by its last cohesive
    contacts. Using that stretched cloud as a collision body creates enormous
    slabs. This fit keeps the current mass centre and momentum, while Kabsch
    alignment recovers a bounded body-local shape from ``rest_position``.
    """
    x = np.asarray(position, dtype=np.float64)
    rest = np.asarray(rest_position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    total_mass = float(m.sum())
    if len(x) < 2 or x.shape != rest.shape or len(v) != len(x) or len(m) != len(x):
        raise ValueError("reference rigid fit needs equally sized particle arrays")
    if total_mass <= 0.0:
        raise ValueError("a reference rigid cluster needs positive mass")

    center = (m[:, None] * x).sum(axis=0) / total_mass
    rest_center = (m[:, None] * rest).sum(axis=0) / total_mass
    local = rest - rest_center
    current_local = x - center
    covariance = (local * m[:, None]).T @ current_local
    left, _, right_t = np.linalg.svd(covariance, full_matrices=False)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    fitted_arms = (rotation @ local.T).T
    reference_residual = current_local - fitted_arms
    reference_position_rms = float(
        np.sqrt((m * np.sum(reference_residual * reference_residual, axis=1)).sum() / total_mass)
    )

    linear_velocity = (m[:, None] * v).sum(axis=0) / total_mass
    relative_velocity = v - linear_velocity
    angular_momentum_world = np.sum(
        np.cross(current_local, m[:, None] * relative_velocity), axis=0
    )
    _, inverse_inertia = _mass_properties(local, m)
    angular_velocity = rotation @ (inverse_inertia @ (rotation.T @ angular_momentum_world))
    rigid_velocity = linear_velocity + np.cross(angular_velocity[None, :], fitted_arms)
    residual = v - rigid_velocity
    internal_velocity_rms = float(
        np.sqrt((m * np.sum(residual * residual, axis=1)).sum() / total_mass)
    )

    return RigidClusterFit(
        mass=total_mass,
        center=center.astype(np.float32),
        orientation=_rotation_matrix_to_quaternion(rotation),
        linear_velocity=linear_velocity.astype(np.float32),
        angular_velocity=angular_velocity.astype(np.float32),
        inverse_inertia=inverse_inertia.astype(np.float32),
        local_positions=local.astype(np.float32),
        internal_velocity_rms=internal_velocity_rms,
        reference_position_rms=reference_position_rms,
    )
