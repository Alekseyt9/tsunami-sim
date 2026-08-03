"""Prepared high-impact optimization paths for DELUGE V3.

The production solver does not switch numerical models merely because these
objects exist.  They provide the buffers, conservative diagnostics, and
transfer accounting needed before enabling a larger-step implicit fluid solve
or replacing calm interior SPH samples by a coarse 3D volume grid.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from kernels.base import poly6, spiky_grad, viscosity_laplacian


@wp.kernel
def dfsph_density_factor_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    sound_speed: float,
    hydrostatic_depth: float,
    initial_wave_height: float,
    reservoir_z_max: float,
    max_support: float,
    bootstrap_reference: int,
    bootstrap_ratio_minimum: float,
    bootstrap_ratio_maximum: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        rho[i] = rest_density
        factor[i] = 0.0
        return
    xi = x[i]
    rhoi = float(0.0)
    gradient_mass_sum = wp.vec3(0.0)
    neighbour_inverse_mass_term = float(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if kind[j] == 0 and water_phase[j] == 2:
            continue
        delta = xi - x[j]
        distance_squared = wp.dot(delta, delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance_squared >= support * support or distance_squared >= max_support * max_support:
            continue
        effective_mass = mass[j]
        if kind[j] != 0:
            effective_mass = rest_density * volume[j]
        rhoi += effective_mass * poly6(distance_squared, support)
        if j == i:
            continue
        distance = wp.sqrt(distance_squared)
        if distance <= 1.0e-5:
            continue
        kernel_gradient = spiky_grad(delta, distance, support)
        gradient_mass_sum += effective_mass * kernel_gradient
        if kind[j] == 0:
            # For C_i=rho_i/reference_i-1, the neighbour gradient is
            # -m_j/reference_i*grad(W_ij).  Its inverse-mass contribution to
            # the generalized constraint denominator is m_j*|grad(W)|^2.
            neighbour_inverse_mass_term += mass[j] * wp.dot(
                kernel_gradient, kernel_gradient
            )
    rhoi = wp.max(rhoi, rest_density * 0.15)
    reference = rho_reference[i]
    if reference <= 0.0:
        gamma = 7.0
        stiffness = rest_density * sound_speed * sound_speed / gamma
        crest_dz = (xi[2] - reservoir_z_max + 5.0) / 7.5
        local_surface = hydrostatic_depth + initial_wave_height * wp.exp(-crest_dz * crest_dz)
        water_column = wp.max(local_surface - xi[1], 0.0)
        hydro_pressure = rest_density * 9.81 * water_column
        target_ratio = wp.pow(1.0 + hydro_pressure / stiffness, 1.0 / gamma)
        reference = rhoi / target_ratio
        rho_reference[i] = reference
        rho[i] = rest_density * target_ratio
    else:
        rho[i] = wp.max(rhoi * rest_density / reference, rest_density * 0.15)
    normalized_ratio = rho[i] / rest_density
    if (
        bootstrap_reference != 0
        and (
            normalized_ratio < bootstrap_ratio_minimum
            or normalized_ratio > bootstrap_ratio_maximum
        )
    ):
        # Legacy WCSPH checkpoints can contain invalid refinement references
        # hidden by the EOS pressure clamp. Rebase only on the first implicit
        # execution; doing this continuously would erase real compression.
        reference = rhoi
        rho_reference[i] = reference
        rho[i] = rest_density
    safe_reference = wp.max(reference, rest_density * 0.05)
    # Generalized PBD/DFSPH denominator for unequal particle masses:
    #   sum_k (1/m_k) |dC_i/dx_k|^2
    # Boundary volumes contribute to the central gradient but have infinite
    # mass, so they correctly have no neighbour inverse-mass term.
    denominator = (
        wp.dot(gradient_mass_sum, gradient_mass_sum)
        / wp.max(mass[i], 1.0e-8)
        + neighbour_inverse_mass_term
    ) / (safe_reference * safe_reference)
    if denominator > 1.0e-9:
        factor[i] = 1.0 / denominator
    else:
        factor[i] = 0.0


@wp.kernel
def dfsph_predict_velocity_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    predicted_velocity: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    viscosity: float,
    xsph_strength: float,
    max_support: float,
    dt: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    xi = x[i]
    vi = v[i]
    acceleration = wp.vec3(0.0, -9.81, 0.0)
    xsph = wp.vec3(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    if water_phase[i] == 2:
        for edge in range(start, end):
            j = neighbour_index[edge]
            if kind[j] == 0:
                continue
            delta = xi - x[j]
            distance = wp.length(delta)
            contact_distance = radius[i] + radius[j]
            if distance <= 1.0e-5 or distance >= contact_distance:
                continue
            normal = delta / distance
            penetration = contact_distance - distance
            approach_speed = wp.min(wp.dot(vi - v[j], normal), 0.0)
            contact_acceleration = normal * (
                2400.0 * penetration - 55.0 * approach_speed
            )
            acceleration += contact_acceleration
            wp.atomic_add(solid_force, j, -contact_acceleration * mass[i])
        predicted_velocity[i] = vi + acceleration * dt
        return
    inverse_rhoi = 1.0 / wp.max(rho[i], rest_density * 0.15)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        rhoj = rest_density
        neighbour_mass = rest_density * volume[j]
        if kind[j] == 0:
            rhoj = wp.max(rho[j], rest_density * 0.15)
            neighbour_mass = mass[j]
        viscous_acceleration = (
            viscosity * neighbour_mass * (v[j] - vi)
            * (1.0 / rhoj) * viscosity_laplacian(distance, support) * inverse_rhoi
        )
        acceleration += viscous_acceleration
        if kind[j] == 0:
            xsph += mass[j] / rhoj * (v[j] - vi) * poly6(
                wp.dot(delta, delta), support
            )
        else:
            wp.atomic_add(solid_force, j, -viscous_acceleration * mass[i])
    acceleration += xsph * (xsph_strength / wp.max(dt, 1.0e-7))
    predicted_velocity[i] = vi + acceleration * dt


@wp.kernel
def dfsph_density_advected_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    solid_velocity: wp.array(dtype=wp.vec3),
    predicted_velocity: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    density_advected: wp.array(dtype=float),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    max_support: float,
    dt: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        density_advected[i] = 1.0
        return
    xi = x[i]
    vi = predicted_velocity[i]
    divergence = float(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        neighbour_velocity = solid_velocity[j]
        neighbour_volume = volume[j]
        if kind[j] == 0:
            neighbour_velocity = predicted_velocity[j]
            neighbour_volume = mass[j] / rest_density
        divergence += neighbour_volume * wp.dot(
            vi - neighbour_velocity, spiky_grad(delta, distance, support)
        )
    normalization = rest_density / wp.max(
        rho_reference[i], rest_density * 0.05
    )
    density_advected[i] = rho[i] / rest_density + dt * normalization * divergence


@wp.kernel
def dfsph_initialize_kappa(
    fluid_particle: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    density_advected: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
    kappa_warmstart: wp.array(dtype=float),
    inverse_dt_squared: float,
    warmstart_blend: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        kappa[i] = 0.0
        return
    initial = (
        wp.max(density_advected[i] - 1.0, 0.0)
        * factor[i] * inverse_dt_squared
    )
    warm = kappa_warmstart[i] * warmstart_blend * inverse_dt_squared
    if warm > 0.0:
        kappa[i] = warm
    else:
        kappa[i] = initial


@wp.kernel
def dfsph_store_warmstart(
    fluid_particle: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    kappa: wp.array(dtype=float),
    kappa_warmstart: wp.array(dtype=float),
    dt_squared: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        kappa_warmstart[i] = 0.0
    else:
        kappa_warmstart[i] = kappa[i] * dt_squared


@wp.kernel
def dfsph_pressure_acceleration_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    rho_reference: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
    pressure_acceleration: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    max_support: float,
    boundary_reaction_scale: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        pressure_acceleration[i] = wp.vec3(0.0)
        return
    xi = x[i]
    acceleration = wp.vec3(0.0)
    reference_i = wp.max(rho_reference[i], rest_density * 0.05)
    inverse_mass_i = 1.0 / wp.max(mass[i], 1.0e-8)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        effective_mass_j = rest_density * volume[j]
        neighbour_term = float(0.0)
        if kind[j] == 0:
            effective_mass_j = mass[j]
            # Contribution from neighbour constraint C_j to particle i.
            # Unlike the central term, m_i cancels with particle i's inverse
            # mass. This is the part lost by the equal-volume formulation.
            neighbour_term = kappa[j] / wp.max(
                rho_reference[j], rest_density * 0.05
            )
        pair_acceleration = -(
            kappa[i] * effective_mass_j * inverse_mass_i / reference_i
            + neighbour_term
        ) * spiky_grad(delta, distance, support)
        acceleration += pair_acceleration
        if kind[j] != 0 and boundary_reaction_scale > 0.0:
            wp.atomic_add(
                solid_force, j,
                -pair_acceleration * mass[i] * boundary_reaction_scale,
            )
    pressure_acceleration[i] = acceleration


@wp.kernel
def dfsph_jacobi_update_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    density_advected: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    pressure_acceleration: wp.array(dtype=wp.vec3),
    kappa: wp.array(dtype=float),
    compression_residual: wp.array(dtype=float),
    error_accumulator: wp.array(dtype=float),
    sample_counter: wp.array(dtype=wp.int32),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    max_support: float,
    dt_squared: float,
    relaxation: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        return
    xi = x[i]
    ai = pressure_acceleration[i]
    pressure_density_change = float(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        neighbour_acceleration = wp.vec3(0.0)
        neighbour_volume = volume[j]
        if kind[j] == 0:
            neighbour_acceleration = pressure_acceleration[j]
            neighbour_volume = mass[j] / rest_density
        pressure_density_change += neighbour_volume * wp.dot(
            ai - neighbour_acceleration,
            spiky_grad(delta, distance, support),
        )
    normalization = rest_density / wp.max(
        rho_reference[i], rest_density * 0.05
    )
    residual = (
        density_advected[i] - 1.0
        + dt_squared * normalization * pressure_density_change
    )
    compression_error = wp.max(residual, 0.0)
    compression_residual[i] = compression_error
    kappa[i] = wp.max(
        kappa[i] + relaxation * residual * factor[i] / wp.max(dt_squared, 1.0e-12),
        0.0,
    )
    wp.atomic_add(error_accumulator, 0, compression_error)
    wp.atomic_max(error_accumulator, 1, compression_error)
    wp.atomic_add(sample_counter, 0, 1)


@wp.kernel
def dfsph_apply_velocity_correction(
    fluid_particle: wp.array(dtype=wp.int32),
    predicted_velocity: wp.array(dtype=wp.vec3),
    correction: wp.array(dtype=wp.vec3),
    correction_scale: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    predicted_velocity[i] += correction[i] * correction_scale


@wp.kernel
def dfsph_finalize_predicted_acceleration(
    fluid_particle: wp.array(dtype=wp.int32),
    v: wp.array(dtype=wp.vec3),
    predicted_velocity: wp.array(dtype=wp.vec3),
    acceleration: wp.array(dtype=wp.vec3),
    inverse_dt: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    acceleration[i] = (predicted_velocity[i] - v[i]) * inverse_dt


@wp.kernel
def dfsph_divergence_advected_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    solid_velocity: wp.array(dtype=wp.vec3),
    predicted_velocity: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    rho_reference: wp.array(dtype=float),
    divergence_advected: wp.array(dtype=float),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    max_support: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        divergence_advected[i] = 0.0
        return
    xi = x[i]
    vi = predicted_velocity[i]
    inverse_reference = 1.0 / wp.max(
        rho_reference[i], rest_density * 0.05
    )
    divergence = float(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        neighbour_velocity = solid_velocity[j]
        effective_mass = rest_density * volume[j]
        if kind[j] == 0:
            neighbour_velocity = predicted_velocity[j]
            effective_mass = mass[j]
        divergence += effective_mass * inverse_reference * wp.dot(
            vi - neighbour_velocity,
            spiky_grad(delta, distance, support),
        )
    divergence_advected[i] = divergence


@wp.kernel
def dfsph_initialize_divergence_kappa(
    fluid_particle: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    divergence_advected: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        kappa[i] = 0.0
        return
    kappa[i] = wp.max(divergence_advected[i], 0.0) * factor[i]


@wp.kernel
def dfsph_divergence_jacobi_update_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    divergence_advected: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    velocity_correction: wp.array(dtype=wp.vec3),
    kappa: wp.array(dtype=float),
    compression_residual: wp.array(dtype=float),
    error_accumulator: wp.array(dtype=float),
    sample_counter: wp.array(dtype=wp.int32),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    max_support: float,
    relaxation: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        return
    xi = x[i]
    correction_i = velocity_correction[i]
    inverse_reference = 1.0 / wp.max(
        rho_reference[i], rest_density * 0.05
    )
    correction_divergence = float(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        neighbour_correction = wp.vec3(0.0)
        effective_mass = rest_density * volume[j]
        if kind[j] == 0:
            neighbour_correction = velocity_correction[j]
            effective_mass = mass[j]
        correction_divergence += effective_mass * inverse_reference * wp.dot(
            correction_i - neighbour_correction,
            spiky_grad(delta, distance, support),
        )
    residual = divergence_advected[i] + correction_divergence
    compression_error = wp.max(residual, 0.0)
    compression_residual[i] = compression_error
    kappa[i] = wp.max(
        kappa[i] + relaxation * residual * factor[i], 0.0
    )
    wp.atomic_add(error_accumulator, 0, compression_error)
    wp.atomic_max(error_accumulator, 1, compression_error)
    wp.atomic_add(sample_counter, 0, 1)


@wp.kernel
def dfsph_clear_selection(
    fluid_particle: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    fluid_slot: wp.array(dtype=wp.int32),
    selection: wp.array(dtype=wp.int32),
    expanded_selection: wp.array(dtype=wp.int32),
    kappa: wp.array(dtype=float),
    correction: wp.array(dtype=wp.vec3),
):
    """Reset cheap per-particle state and build particle -> fluid-slot map."""
    slot = wp.tid()
    i = fluid_particle[slot]
    fluid_slot[i] = slot
    selection[slot] = 0
    expanded_selection[slot] = 0
    kappa[i] = 0.0
    correction[i] = wp.vec3(0.0)


@wp.kernel
def dfsph_mark_density_compression(
    fluid_particle: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    density_advected: wp.array(dtype=float),
    selection: wp.array(dtype=wp.int32),
    threshold: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] != 2 and density_advected[i] - 1.0 >= threshold:
        selection[slot] = 1


@wp.kernel
def dfsph_mark_divergence_compression(
    fluid_particle: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    divergence_advected: wp.array(dtype=float),
    selection: wp.array(dtype=wp.int32),
    threshold: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] != 2 and divergence_advected[i] >= threshold:
        selection[slot] = 1


@wp.kernel
def dfsph_expand_selection_one_ring(
    fluid_particle: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    fluid_slot: wp.array(dtype=wp.int32),
    selection: wp.array(dtype=wp.int32),
    expanded_selection: wp.array(dtype=wp.int32),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
):
    slot = wp.tid()
    if selection[slot] == 0:
        return
    wp.atomic_max(expanded_selection, slot, 1)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if kind[j] == 0:
            neighbour_slot = fluid_slot[j]
            if neighbour_slot >= 0:
                wp.atomic_max(expanded_selection, neighbour_slot, 1)


@wp.kernel
def dfsph_collect_selected_slots(
    selection: wp.array(dtype=wp.int32),
    selected_slot: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
):
    slot = wp.tid()
    if selection[slot] != 0:
        destination = wp.atomic_add(selected_count, 0, 1)
        selected_slot[destination] = slot


@wp.kernel
def dfsph_initialize_kappa_selected(
    fluid_particle: wp.array(dtype=wp.int32),
    selected_slot: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
    constraint_selection: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    density_advected: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
    kappa_warmstart: wp.array(dtype=float),
    inverse_dt_squared: float,
    warmstart_blend: float,
):
    selected_index = wp.tid()
    if selected_index >= selected_count[0]:
        return
    slot = selected_slot[selected_index]
    i = fluid_particle[slot]
    if water_phase[i] == 2 or constraint_selection[slot] == 0:
        return
    initial = (
        wp.max(density_advected[i] - 1.0, 0.0)
        * factor[i] * inverse_dt_squared
    )
    warm = kappa_warmstart[i] * warmstart_blend * inverse_dt_squared
    if warm > 0.0:
        kappa[i] = warm
    else:
        kappa[i] = initial


@wp.kernel
def dfsph_initialize_divergence_kappa_selected(
    fluid_particle: wp.array(dtype=wp.int32),
    selected_slot: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
    constraint_selection: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    divergence_advected: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
):
    selected_index = wp.tid()
    if selected_index >= selected_count[0]:
        return
    slot = selected_slot[selected_index]
    i = fluid_particle[slot]
    if water_phase[i] != 2 and constraint_selection[slot] != 0:
        kappa[i] = wp.max(divergence_advected[i], 0.0) * factor[i]


@wp.kernel
def dfsph_pressure_acceleration_selected_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    selected_slot: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    rho_reference: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
    pressure_acceleration: wp.array(dtype=wp.vec3),
    solid_force: wp.array(dtype=wp.vec3),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    max_support: float,
    boundary_reaction_scale: float,
):
    selected_index = wp.tid()
    if selected_index >= selected_count[0]:
        return
    slot = selected_slot[selected_index]
    i = fluid_particle[slot]
    if water_phase[i] == 2:
        return
    xi = x[i]
    acceleration = wp.vec3(0.0)
    reference_i = wp.max(rho_reference[i], rest_density * 0.05)
    inverse_mass_i = 1.0 / wp.max(mass[i], 1.0e-8)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        effective_mass_j = rest_density * volume[j]
        neighbour_term = float(0.0)
        if kind[j] == 0:
            effective_mass_j = mass[j]
            neighbour_term = kappa[j] / wp.max(
                rho_reference[j], rest_density * 0.05
            )
        pair_acceleration = -(
            kappa[i] * effective_mass_j * inverse_mass_i / reference_i
            + neighbour_term
        ) * spiky_grad(delta, distance, support)
        acceleration += pair_acceleration
        if kind[j] != 0 and boundary_reaction_scale > 0.0:
            wp.atomic_add(
                solid_force, j,
                -pair_acceleration * mass[i] * boundary_reaction_scale,
            )
    pressure_acceleration[i] = acceleration


@wp.kernel
def dfsph_density_jacobi_update_selected_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    selected_slot: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
    constraint_selection: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    density_advected: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    pressure_acceleration: wp.array(dtype=wp.vec3),
    kappa: wp.array(dtype=float),
    compression_residual: wp.array(dtype=float),
    error_accumulator: wp.array(dtype=float),
    sample_counter: wp.array(dtype=wp.int32),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    max_support: float,
    dt_squared: float,
    relaxation: float,
):
    selected_index = wp.tid()
    if selected_index >= selected_count[0]:
        return
    slot = selected_slot[selected_index]
    i = fluid_particle[slot]
    if water_phase[i] == 2 or constraint_selection[slot] == 0:
        return
    xi = x[i]
    ai = pressure_acceleration[i]
    pressure_density_change = float(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        neighbour_acceleration = wp.vec3(0.0)
        neighbour_volume = volume[j]
        if kind[j] == 0:
            neighbour_acceleration = pressure_acceleration[j]
            neighbour_volume = mass[j] / rest_density
        pressure_density_change += neighbour_volume * wp.dot(
            ai - neighbour_acceleration,
            spiky_grad(delta, distance, support),
        )
    normalization = rest_density / wp.max(
        rho_reference[i], rest_density * 0.05
    )
    residual = (
        density_advected[i] - 1.0
        + dt_squared * normalization * pressure_density_change
    )
    compression_error = wp.max(residual, 0.0)
    compression_residual[i] = compression_error
    kappa[i] = wp.max(
        kappa[i] + relaxation * residual * factor[i]
        / wp.max(dt_squared, 1.0e-12),
        0.0,
    )
    wp.atomic_add(error_accumulator, 0, compression_error)
    wp.atomic_max(error_accumulator, 1, compression_error)
    wp.atomic_add(sample_counter, 0, 1)


@wp.kernel
def dfsph_divergence_jacobi_update_selected_verlet(
    x: wp.array(dtype=wp.vec3),
    fluid_particle: wp.array(dtype=wp.int32),
    selected_slot: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
    constraint_selection: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    divergence_advected: wp.array(dtype=float),
    rho_reference: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    velocity_correction: wp.array(dtype=wp.vec3),
    kappa: wp.array(dtype=float),
    compression_residual: wp.array(dtype=float),
    error_accumulator: wp.array(dtype=float),
    sample_counter: wp.array(dtype=wp.int32),
    neighbour_count: wp.array(dtype=wp.int32),
    neighbour_offset: wp.array(dtype=wp.int32),
    neighbour_index: wp.array(dtype=wp.int32),
    neighbour_capacity: int,
    rest_density: float,
    max_support: float,
    relaxation: float,
):
    selected_index = wp.tid()
    if selected_index >= selected_count[0]:
        return
    slot = selected_slot[selected_index]
    i = fluid_particle[slot]
    if water_phase[i] == 2 or constraint_selection[slot] == 0:
        return
    xi = x[i]
    correction_i = velocity_correction[i]
    inverse_reference = 1.0 / wp.max(
        rho_reference[i], rest_density * 0.05
    )
    correction_divergence = float(0.0)
    start = neighbour_offset[slot]
    end = wp.min(start + neighbour_count[slot], neighbour_capacity)
    for edge in range(start, end):
        j = neighbour_index[edge]
        if j == i or (kind[j] == 0 and water_phase[j] == 2):
            continue
        delta = xi - x[j]
        distance = wp.length(delta)
        support = 4.0 * wp.max(radius[i], radius[j])
        if distance <= 1.0e-5 or distance >= support or distance >= max_support:
            continue
        neighbour_correction = wp.vec3(0.0)
        effective_mass = rest_density * volume[j]
        if kind[j] == 0:
            neighbour_correction = velocity_correction[j]
            effective_mass = mass[j]
        correction_divergence += effective_mass * inverse_reference * wp.dot(
            correction_i - neighbour_correction,
            spiky_grad(delta, distance, support),
        )
    residual = divergence_advected[i] + correction_divergence
    compression_error = wp.max(residual, 0.0)
    compression_residual[i] = compression_error
    kappa[i] = wp.max(
        kappa[i] + relaxation * residual * factor[i], 0.0
    )
    wp.atomic_add(error_accumulator, 0, compression_error)
    wp.atomic_max(error_accumulator, 1, compression_error)
    wp.atomic_add(sample_counter, 0, 1)


@wp.kernel
def dfsph_apply_velocity_correction_selected(
    fluid_particle: wp.array(dtype=wp.int32),
    selected_slot: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
    predicted_velocity: wp.array(dtype=wp.vec3),
    correction: wp.array(dtype=wp.vec3),
    correction_scale: float,
):
    selected_index = wp.tid()
    if selected_index >= selected_count[0]:
        return
    i = fluid_particle[selected_slot[selected_index]]
    predicted_velocity[i] += correction[i] * correction_scale


@wp.kernel
def dfsph_store_warmstart_selected(
    fluid_particle: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    selection: wp.array(dtype=wp.int32),
    kappa: wp.array(dtype=float),
    kappa_warmstart: wp.array(dtype=float),
    dt_squared: float,
):
    slot = wp.tid()
    i = fluid_particle[slot]
    if water_phase[i] == 2 or selection[slot] == 0:
        kappa_warmstart[i] = 0.0
    else:
        kappa_warmstart[i] = kappa[i] * dt_squared


@wp.kernel
def classify_and_deposit_narrow_band_interior(
    neighbour_grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    surface_mask: wp.array(dtype=wp.int32),
    water_phase: wp.array(dtype=wp.int32),
    interior_mask: wp.array(dtype=wp.int32),
    grid_mass: wp.array3d(dtype=float),
    grid_volume: wp.array3d(dtype=float),
    grid_momentum_x: wp.array3d(dtype=float),
    grid_momentum_y: wp.array3d(dtype=float),
    grid_momentum_z: wp.array3d(dtype=float),
    counters: wp.array(dtype=wp.int32),
    lower: wp.vec3,
    inverse_cell_size: float,
    nx: int,
    ny: int,
    nz: int,
    detail_distance: float,
    maximum_velocity_rms: float,
):
    particle = wp.tid()
    interior_mask[particle] = 0
    if (
        kind[particle] != 0 or water_phase[particle] != 0
        or surface_mask[particle] != 0
    ):
        return
    position = x[particle]
    velocity = v[particle]

    # A particle remains detailed if a free-surface or solid boundary sample
    # lies inside the configured band.  This deliberately over-preserves SPH
    # around impacts; the diagnostic must never overstate removable particles.
    near_detail = int(0)
    velocity_delta_squared = float(0.0)
    velocity_samples = int(0)
    query = wp.hash_grid_query(neighbour_grid, position, detail_distance)
    for neighbour in query:
        if surface_mask[neighbour] != 0 or kind[neighbour] != 0:
            near_detail = 1
            break
        # Uniform translation is safe for a coarse volume grid, even at
        # tsunami velocity. Local shear/rotation is not: retain SPH wherever
        # nearby connected water has a materially different velocity.
        if water_phase[neighbour] != 0:
            near_detail = 1
            break
        velocity_delta = v[neighbour] - velocity
        velocity_delta_squared += wp.dot(velocity_delta, velocity_delta)
        velocity_samples += 1
    if near_detail != 0:
        return
    if velocity_samples > 0:
        velocity_rms = wp.sqrt(velocity_delta_squared / float(velocity_samples))
        if velocity_rms > maximum_velocity_rms:
            return

    cell = wp.vec3i(
        int(wp.floor((position[0] - lower[0]) * inverse_cell_size)),
        int(wp.floor((position[1] - lower[1]) * inverse_cell_size)),
        int(wp.floor((position[2] - lower[2]) * inverse_cell_size)),
    )
    if (
        cell[0] < 0 or cell[0] >= nx or cell[1] < 0 or cell[1] >= ny
        or cell[2] < 0 or cell[2] >= nz
    ):
        return
    interior_mask[particle] = 1
    particle_mass = mass[particle]
    wp.atomic_add(grid_mass, cell[0], cell[1], cell[2], particle_mass)
    wp.atomic_add(grid_volume, cell[0], cell[1], cell[2], volume[particle])
    wp.atomic_add(
        grid_momentum_x, cell[0], cell[1], cell[2], particle_mass * velocity[0]
    )
    wp.atomic_add(
        grid_momentum_y, cell[0], cell[1], cell[2], particle_mass * velocity[1]
    )
    wp.atomic_add(
        grid_momentum_z, cell[0], cell[1], cell[2], particle_mass * velocity[2]
    )
    wp.atomic_add(counters, 0, 1)


class ImplicitFluidPreparation:
    """Feature-gated constant-density DFSPH projection and CFL audit."""

    def __init__(self, policy: dict[str, Any], capacity: int, device: str):
        self.policy = policy
        self.enabled = bool(policy.get("enabled", False))
        self.mode = str(policy.get("mode", "diagnostic"))
        self.execution_enabled = self.enabled and self.mode.lower() in (
            "density_projection", "dfsph", "execute"
        )
        self.target_dt_multiplier = max(
            1.0, float(policy.get("target_dt_multiplier", 2.0))
        )
        self.maximum_dt = max(1.0e-7, float(policy.get("maximum_dt", 0.00036)))
        self.advection_cfl = max(0.05, float(policy.get("advection_cfl", 0.40)))
        self.acceleration_cfl = max(
            0.05, float(policy.get("acceleration_cfl", 0.25))
        )
        self.minimum_pressure_iterations = max(
            1, int(policy.get("minimum_pressure_iterations", 2))
        )
        self.maximum_pressure_iterations = max(
            self.minimum_pressure_iterations,
            int(policy.get("maximum_pressure_iterations", 12)),
        )
        self.pressure_relaxation = min(
            1.0, max(0.05, float(policy.get("pressure_relaxation", 0.5)))
        )
        self.warmstart_blend = min(
            1.0, max(0.0, float(policy.get("warmstart_blend", 0.5)))
        )
        self.divergence_projection_enabled = bool(
            policy.get("divergence_projection", True)
        )
        self.divergence_iterations = max(
            1, int(policy.get("divergence_iterations", 2))
        )
        self.divergence_relaxation = min(
            1.0, max(0.05, float(policy.get("divergence_relaxation", 0.25)))
        )
        selective_policy = policy.get("selective_compression", {})
        self.selective_compression_enabled = bool(
            selective_policy.get("enabled", False)
        )
        self.density_selection_threshold = max(
            0.0, float(selective_policy.get("density_threshold", 0.002))
        )
        self.divergence_selection_threshold = max(
            0.0,
            float(selective_policy.get("divergence_threshold_per_s", 2.0)),
        )
        self.selection_neighbor_rings = min(
            1, max(0, int(selective_policy.get("expand_neighbor_rings", 1)))
        )
        self.bootstrap_ratio_minimum = min(
            1.0, max(0.1, float(policy.get("bootstrap_ratio_minimum", 0.80)))
        )
        self.bootstrap_ratio_maximum = max(
            1.0, float(policy.get("bootstrap_ratio_maximum", 1.20))
        )
        # These arrays match the standard DFSPH state layout.  In diagnostic
        # mode they are deliberately not touched by the production integrator.
        allocation = capacity if self.enabled else 1
        self.factor = wp.zeros(allocation, dtype=float, device=device)
        self.kappa_density = wp.zeros(allocation, dtype=float, device=device)
        self.kappa_density_warmstart = wp.zeros(
            allocation, dtype=float, device=device
        )
        self.kappa_divergence = wp.zeros(allocation, dtype=float, device=device)
        self.density_advected = wp.zeros(allocation, dtype=float, device=device)
        self.divergence_advected = wp.zeros(
            allocation, dtype=float, device=device
        )
        self.velocity_predicted = wp.zeros(
            allocation, dtype=wp.vec3, device=device
        )
        self.pressure_acceleration = wp.zeros(
            allocation, dtype=wp.vec3, device=device
        )
        self.compression_residual = wp.zeros(
            allocation, dtype=float, device=device
        )
        self.error_accumulator = wp.zeros(2, dtype=float, device=device)
        self.sample_counter = wp.zeros(1, dtype=wp.int32, device=device)
        self.divergence_error_accumulator = wp.zeros(
            2, dtype=float, device=device
        )
        self.divergence_sample_counter = wp.zeros(
            1, dtype=wp.int32, device=device
        )
        self.fluid_slot = wp.zeros(
            allocation, dtype=wp.int32, device=device
        )
        self.compression_selection = wp.zeros(
            allocation, dtype=wp.int32, device=device
        )
        self.expanded_compression_selection = wp.zeros(
            allocation, dtype=wp.int32, device=device
        )
        self.selected_slot = wp.zeros(
            allocation, dtype=wp.int32, device=device
        )
        self.density_selected_count = wp.zeros(
            1, dtype=wp.int32, device=device
        )
        self.divergence_selected_count = wp.zeros(
            1, dtype=wp.int32, device=device
        )
        self.last_execution_iterations = 0
        self.last_divergence_iterations = 0
        self.last_density_selected_count = 0
        self.last_divergence_selected_count = 0
        self.last_fluid_particle_count = 0
        self.execution_calls = 0
        self.last_diagnostics: dict[str, float | int | str] = {}

    def _select_compressed_particles(
        self,
        arrays: dict[str, wp.array],
        fluid_particle: wp.array,
        fluid_particle_count: int,
        neighbour_count: wp.array,
        neighbour_offset: wp.array,
        neighbour_index: wp.array,
        neighbour_capacity: int,
        kappa: wp.array,
        selected_count: wp.array,
        density_mode: bool,
        count: int,
        device: str,
    ) -> wp.array:
        """Build an entirely device-side compact high-compression work list."""
        wp.launch(
            dfsph_clear_selection, dim=fluid_particle_count,
            inputs=[
                fluid_particle, arrays["water_phase"][:count],
                self.fluid_slot, self.compression_selection,
                self.expanded_compression_selection, kappa,
                self.pressure_acceleration,
            ], device=device,
        )
        if density_mode:
            wp.launch(
                dfsph_mark_density_compression, dim=fluid_particle_count,
                inputs=[
                    fluid_particle, arrays["water_phase"][:count],
                    self.density_advected, self.compression_selection,
                    self.density_selection_threshold,
                ], device=device,
            )
        else:
            wp.launch(
                dfsph_mark_divergence_compression, dim=fluid_particle_count,
                inputs=[
                    fluid_particle, arrays["water_phase"][:count],
                    self.divergence_advected, self.compression_selection,
                    self.divergence_selection_threshold,
                ], device=device,
            )
        active_selection = self.compression_selection
        if self.selection_neighbor_rings > 0:
            wp.launch(
                dfsph_expand_selection_one_ring, dim=fluid_particle_count,
                inputs=[
                    fluid_particle, arrays["kind"][:count], self.fluid_slot,
                    self.compression_selection,
                    self.expanded_compression_selection, neighbour_count,
                    neighbour_offset, neighbour_index, neighbour_capacity,
                ], device=device,
            )
            active_selection = self.expanded_compression_selection
        selected_count.zero_()
        wp.launch(
            dfsph_collect_selected_slots, dim=fluid_particle_count,
            inputs=[active_selection, self.selected_slot, selected_count],
            device=device,
        )
        return active_selection

    def execute_density_projection(
        self,
        arrays: dict[str, wp.array],
        fluid_particle: wp.array,
        fluid_particle_count: int,
        neighbour_count: wp.array,
        neighbour_offset: wp.array,
        neighbour_index: wp.array,
        neighbour_capacity: int,
        solid_force: wp.array,
        cfg: dict[str, Any],
        count: int,
        max_support: float,
        dt: float,
        device: str,
    ) -> bool:
        """Execute a fixed-iteration GPU density projection.

        Fixed iteration count avoids a CPU/GPU synchronization inside every
        substep. Convergence metrics are copied only when frame statistics are
        collected.
        """
        if (
            not self.execution_enabled or fluid_particle_count <= 0
            or neighbour_capacity <= 1
        ):
            return False
        rest_density = float(cfg["rest_density"])
        wp.launch(
            dfsph_density_factor_verlet, dim=fluid_particle_count,
            inputs=[
                arrays["x"][:count], fluid_particle,
                arrays["radius"][:count], arrays["mass"][:count],
                arrays["volume"][:count], arrays["kind"][:count],
                arrays["water_phase"][:count], arrays["rho"][:count],
                arrays["rho_reference"][:count], self.factor,
                neighbour_count, neighbour_offset, neighbour_index,
                neighbour_capacity, rest_density, float(cfg["sound_speed"]),
                float(cfg["water_depth"]), float(cfg["wave_height"]),
                float(cfg["reservoir_z_max"]), max_support,
                int(self.execution_calls == 0), self.bootstrap_ratio_minimum,
                self.bootstrap_ratio_maximum,
            ], device=device,
        )
        wp.launch(
            dfsph_predict_velocity_verlet, dim=fluid_particle_count,
            inputs=[
                arrays["x"][:count], fluid_particle, arrays["v"][:count],
                arrays["radius"][:count], arrays["mass"][:count],
                arrays["volume"][:count], arrays["kind"][:count],
                arrays["water_phase"][:count], arrays["rho"][:count],
                self.velocity_predicted, solid_force,
                neighbour_count, neighbour_offset, neighbour_index,
                neighbour_capacity, rest_density, float(cfg["viscosity"]),
                float(cfg.get("xsph_strength", 0.0)), max_support, dt,
            ], device=device,
        )
        wp.launch(
            dfsph_density_advected_verlet, dim=fluid_particle_count,
            inputs=[
                arrays["x"][:count], fluid_particle, arrays["v"][:count],
                self.velocity_predicted, arrays["radius"][:count],
                arrays["mass"][:count], arrays["volume"][:count],
                arrays["kind"][:count], arrays["water_phase"][:count],
                arrays["rho"][:count], arrays["rho_reference"][:count],
                self.density_advected,
                neighbour_count, neighbour_offset, neighbour_index,
                neighbour_capacity, rest_density, max_support, dt,
            ], device=device,
        )
        inverse_dt_squared = 1.0 / max(dt * dt, 1.0e-12)
        if self.selective_compression_enabled:
            self._select_compressed_particles(
                arrays, fluid_particle, fluid_particle_count,
                neighbour_count, neighbour_offset, neighbour_index,
                neighbour_capacity, self.kappa_density,
                self.density_selected_count, True, count, device,
            )
            wp.launch(
                dfsph_initialize_kappa_selected, dim=fluid_particle_count,
                inputs=[
                    fluid_particle, self.selected_slot,
                    self.density_selected_count,
                    self.compression_selection,
                    arrays["water_phase"][:count], self.density_advected,
                    self.factor, self.kappa_density,
                    self.kappa_density_warmstart, inverse_dt_squared,
                    self.warmstart_blend,
                ], device=device,
            )
        else:
            wp.launch(
                dfsph_initialize_kappa, dim=fluid_particle_count,
                inputs=[
                    fluid_particle, arrays["water_phase"][:count],
                    self.density_advected, self.factor, self.kappa_density,
                    self.kappa_density_warmstart, inverse_dt_squared,
                    self.warmstart_blend,
                ], device=device,
            )
        for _ in range(self.maximum_pressure_iterations):
            if self.selective_compression_enabled:
                self.pressure_acceleration.zero_()
                wp.launch(
                    dfsph_pressure_acceleration_selected_verlet,
                    dim=fluid_particle_count,
                    inputs=[
                        arrays["x"][:count], fluid_particle,
                        self.selected_slot, self.density_selected_count,
                        arrays["radius"][:count], arrays["mass"][:count],
                        arrays["volume"][:count], arrays["kind"][:count],
                        arrays["water_phase"][:count],
                        arrays["rho_reference"][:count], self.kappa_density,
                        self.pressure_acceleration, solid_force,
                        neighbour_count, neighbour_offset, neighbour_index,
                        neighbour_capacity, rest_density, max_support, 0.0,
                    ], device=device,
                )
            else:
                wp.launch(
                    dfsph_pressure_acceleration_verlet,
                    dim=fluid_particle_count,
                    inputs=[
                        arrays["x"][:count], fluid_particle,
                        arrays["radius"][:count], arrays["mass"][:count],
                        arrays["volume"][:count], arrays["kind"][:count],
                        arrays["water_phase"][:count],
                        arrays["rho_reference"][:count], self.kappa_density,
                        self.pressure_acceleration, solid_force,
                        neighbour_count, neighbour_offset, neighbour_index,
                        neighbour_capacity, rest_density, max_support, 0.0,
                    ], device=device,
                )
            self.error_accumulator.zero_()
            self.sample_counter.zero_()
            if self.selective_compression_enabled:
                wp.launch(
                    dfsph_density_jacobi_update_selected_verlet,
                    dim=fluid_particle_count,
                    inputs=[
                        arrays["x"][:count], fluid_particle,
                        self.selected_slot, self.density_selected_count,
                        self.compression_selection,
                        arrays["radius"][:count], arrays["mass"][:count],
                        arrays["volume"][:count], arrays["kind"][:count],
                        arrays["water_phase"][:count], self.density_advected,
                        arrays["rho_reference"][:count], self.factor,
                        self.pressure_acceleration, self.kappa_density,
                        self.compression_residual, self.error_accumulator,
                        self.sample_counter, neighbour_count, neighbour_offset,
                        neighbour_index, neighbour_capacity, rest_density,
                        max_support, dt * dt, self.pressure_relaxation,
                    ], device=device,
                )
            else:
                wp.launch(
                    dfsph_jacobi_update_verlet, dim=fluid_particle_count,
                    inputs=[
                        arrays["x"][:count], fluid_particle,
                        arrays["radius"][:count], arrays["mass"][:count],
                        arrays["volume"][:count], arrays["kind"][:count],
                        arrays["water_phase"][:count], self.density_advected,
                        arrays["rho_reference"][:count], self.factor,
                        self.pressure_acceleration,
                        self.kappa_density, self.compression_residual,
                        self.error_accumulator, self.sample_counter,
                        neighbour_count, neighbour_offset, neighbour_index,
                        neighbour_capacity, rest_density, max_support,
                        dt * dt, self.pressure_relaxation,
                    ], device=device,
                )
        if self.selective_compression_enabled:
            self.pressure_acceleration.zero_()
            wp.launch(
                dfsph_pressure_acceleration_selected_verlet,
                dim=fluid_particle_count,
                inputs=[
                    arrays["x"][:count], fluid_particle, self.selected_slot,
                    self.density_selected_count, arrays["radius"][:count],
                    arrays["mass"][:count], arrays["volume"][:count],
                    arrays["kind"][:count], arrays["water_phase"][:count],
                    arrays["rho_reference"][:count], self.kappa_density,
                    self.pressure_acceleration, solid_force,
                    neighbour_count, neighbour_offset, neighbour_index,
                    neighbour_capacity, rest_density, max_support, 1.0,
                ], device=device,
            )
            wp.launch(
                dfsph_apply_velocity_correction_selected,
                dim=fluid_particle_count,
                inputs=[
                    fluid_particle, self.selected_slot,
                    self.density_selected_count, self.velocity_predicted,
                    self.pressure_acceleration, dt,
                ], device=device,
            )
            wp.launch(
                dfsph_store_warmstart_selected, dim=fluid_particle_count,
                inputs=[
                    fluid_particle, arrays["water_phase"][:count],
                    self.compression_selection, self.kappa_density,
                    self.kappa_density_warmstart, dt * dt,
                ], device=device,
            )
        else:
            wp.launch(
                dfsph_pressure_acceleration_verlet,
                dim=fluid_particle_count,
                inputs=[
                    arrays["x"][:count], fluid_particle,
                    arrays["radius"][:count], arrays["mass"][:count],
                    arrays["volume"][:count], arrays["kind"][:count],
                    arrays["water_phase"][:count],
                    arrays["rho_reference"][:count], self.kappa_density,
                    self.pressure_acceleration, solid_force,
                    neighbour_count, neighbour_offset, neighbour_index,
                    neighbour_capacity, rest_density, max_support, 1.0,
                ], device=device,
            )
            wp.launch(
                dfsph_apply_velocity_correction, dim=fluid_particle_count,
                inputs=[
                    fluid_particle, self.velocity_predicted,
                    self.pressure_acceleration, dt,
                ], device=device,
            )
            wp.launch(
                dfsph_store_warmstart, dim=fluid_particle_count,
                inputs=[
                    fluid_particle, arrays["water_phase"][:count],
                    self.kappa_density, self.kappa_density_warmstart,
                    dt * dt,
                ], device=device,
            )
        if self.divergence_projection_enabled:
            wp.launch(
                dfsph_divergence_advected_verlet, dim=fluid_particle_count,
                inputs=[
                    arrays["x"][:count], fluid_particle, arrays["v"][:count],
                    self.velocity_predicted, arrays["radius"][:count],
                    arrays["mass"][:count], arrays["volume"][:count],
                    arrays["kind"][:count], arrays["water_phase"][:count],
                    arrays["rho_reference"][:count],
                    self.divergence_advected,
                    neighbour_count, neighbour_offset, neighbour_index,
                    neighbour_capacity, rest_density, max_support,
                ], device=device,
            )
            if self.selective_compression_enabled:
                self._select_compressed_particles(
                    arrays, fluid_particle, fluid_particle_count,
                    neighbour_count, neighbour_offset, neighbour_index,
                    neighbour_capacity, self.kappa_divergence,
                    self.divergence_selected_count, False, count, device,
                )
                wp.launch(
                    dfsph_initialize_divergence_kappa_selected,
                    dim=fluid_particle_count,
                    inputs=[
                        fluid_particle, self.selected_slot,
                        self.divergence_selected_count,
                        self.compression_selection,
                        arrays["water_phase"][:count],
                        self.divergence_advected, self.factor,
                        self.kappa_divergence,
                    ], device=device,
                )
            else:
                wp.launch(
                    dfsph_initialize_divergence_kappa,
                    dim=fluid_particle_count,
                    inputs=[
                        fluid_particle, arrays["water_phase"][:count],
                        self.divergence_advected, self.factor,
                        self.kappa_divergence,
                    ], device=device,
                )
            for _ in range(self.divergence_iterations):
                if self.selective_compression_enabled:
                    self.pressure_acceleration.zero_()
                    wp.launch(
                        dfsph_pressure_acceleration_selected_verlet,
                        dim=fluid_particle_count,
                        inputs=[
                            arrays["x"][:count], fluid_particle,
                            self.selected_slot, self.divergence_selected_count,
                            arrays["radius"][:count], arrays["mass"][:count],
                            arrays["volume"][:count], arrays["kind"][:count],
                            arrays["water_phase"][:count],
                            arrays["rho_reference"][:count],
                            self.kappa_divergence,
                            self.pressure_acceleration, solid_force,
                            neighbour_count, neighbour_offset,
                            neighbour_index, neighbour_capacity, rest_density,
                            max_support, 0.0,
                        ], device=device,
                    )
                else:
                    wp.launch(
                        dfsph_pressure_acceleration_verlet,
                        dim=fluid_particle_count,
                        inputs=[
                            arrays["x"][:count], fluid_particle,
                            arrays["radius"][:count], arrays["mass"][:count],
                            arrays["volume"][:count], arrays["kind"][:count],
                            arrays["water_phase"][:count],
                            arrays["rho_reference"][:count],
                            self.kappa_divergence,
                            self.pressure_acceleration, solid_force,
                            neighbour_count, neighbour_offset, neighbour_index,
                            neighbour_capacity, rest_density, max_support, 0.0,
                        ], device=device,
                    )
                self.divergence_error_accumulator.zero_()
                self.divergence_sample_counter.zero_()
                if self.selective_compression_enabled:
                    wp.launch(
                        dfsph_divergence_jacobi_update_selected_verlet,
                        dim=fluid_particle_count,
                        inputs=[
                            arrays["x"][:count], fluid_particle,
                            self.selected_slot, self.divergence_selected_count,
                            self.compression_selection,
                            arrays["radius"][:count], arrays["mass"][:count],
                            arrays["volume"][:count], arrays["kind"][:count],
                            arrays["water_phase"][:count],
                            self.divergence_advected,
                            arrays["rho_reference"][:count], self.factor,
                            self.pressure_acceleration,
                            self.kappa_divergence,
                            self.compression_residual,
                            self.divergence_error_accumulator,
                            self.divergence_sample_counter, neighbour_count,
                            neighbour_offset, neighbour_index,
                            neighbour_capacity, rest_density, max_support,
                            self.divergence_relaxation,
                        ], device=device,
                    )
                else:
                    wp.launch(
                        dfsph_divergence_jacobi_update_verlet,
                        dim=fluid_particle_count,
                        inputs=[
                            arrays["x"][:count], fluid_particle,
                            arrays["radius"][:count], arrays["mass"][:count],
                            arrays["volume"][:count], arrays["kind"][:count],
                            arrays["water_phase"][:count],
                            self.divergence_advected,
                            arrays["rho_reference"][:count], self.factor,
                            self.pressure_acceleration,
                            self.kappa_divergence,
                            self.compression_residual,
                            self.divergence_error_accumulator,
                            self.divergence_sample_counter, neighbour_count,
                            neighbour_offset, neighbour_index,
                            neighbour_capacity, rest_density, max_support,
                            self.divergence_relaxation,
                        ], device=device,
                    )
            boundary_scale = 1.0 / max(dt, 1.0e-12)
            if self.selective_compression_enabled:
                self.pressure_acceleration.zero_()
                wp.launch(
                    dfsph_pressure_acceleration_selected_verlet,
                    dim=fluid_particle_count,
                    inputs=[
                        arrays["x"][:count], fluid_particle,
                        self.selected_slot, self.divergence_selected_count,
                        arrays["radius"][:count], arrays["mass"][:count],
                        arrays["volume"][:count], arrays["kind"][:count],
                        arrays["water_phase"][:count],
                        arrays["rho_reference"][:count],
                        self.kappa_divergence, self.pressure_acceleration,
                        solid_force, neighbour_count, neighbour_offset,
                        neighbour_index, neighbour_capacity, rest_density,
                        max_support, boundary_scale,
                    ], device=device,
                )
                wp.launch(
                    dfsph_apply_velocity_correction_selected,
                    dim=fluid_particle_count,
                    inputs=[
                        fluid_particle, self.selected_slot,
                        self.divergence_selected_count,
                        self.velocity_predicted,
                        self.pressure_acceleration, 1.0,
                    ], device=device,
                )
            else:
                wp.launch(
                    dfsph_pressure_acceleration_verlet,
                    dim=fluid_particle_count,
                    inputs=[
                        arrays["x"][:count], fluid_particle,
                        arrays["radius"][:count], arrays["mass"][:count],
                        arrays["volume"][:count], arrays["kind"][:count],
                        arrays["water_phase"][:count],
                        arrays["rho_reference"][:count],
                        self.kappa_divergence, self.pressure_acceleration,
                        solid_force, neighbour_count, neighbour_offset,
                        neighbour_index, neighbour_capacity, rest_density,
                        max_support, boundary_scale,
                    ], device=device,
                )
                wp.launch(
                    dfsph_apply_velocity_correction,
                    dim=fluid_particle_count,
                    inputs=[
                        fluid_particle, self.velocity_predicted,
                        self.pressure_acceleration, 1.0,
                    ], device=device,
                )
            self.last_divergence_iterations = self.divergence_iterations
        else:
            self.last_divergence_iterations = 0
        wp.launch(
            dfsph_finalize_predicted_acceleration, dim=fluid_particle_count,
            inputs=[
                fluid_particle, arrays["v"][:count], self.velocity_predicted,
                arrays["acceleration"][:count],
                1.0 / max(dt, 1.0e-12),
            ], device=device,
        )
        self.last_execution_iterations = self.maximum_pressure_iterations
        self.last_fluid_particle_count = fluid_particle_count
        self.execution_calls += 1
        return True

    def execution_diagnostics(self) -> dict[str, float | int | str]:
        if not self.execution_enabled or self.last_execution_iterations <= 0:
            return {}
        error = self.error_accumulator.numpy()
        samples = max(int(self.sample_counter.numpy()[0]), 1)
        result = {
            "implicit_execution_mode": self.mode,
            "implicit_pressure_iterations": self.last_execution_iterations,
            "implicit_density_error_mean_percent": 100.0 * float(error[0]) / samples,
            "implicit_density_error_max_percent": 100.0 * float(error[1]),
        }
        if self.last_divergence_iterations > 0:
            divergence_error = self.divergence_error_accumulator.numpy()
            divergence_samples = max(
                int(self.divergence_sample_counter.numpy()[0]), 1
            )
            result.update({
                "implicit_divergence_iterations": self.last_divergence_iterations,
                "implicit_divergence_error_mean_per_s": float(
                    divergence_error[0]
                ) / divergence_samples,
                "implicit_divergence_error_max_per_s": float(
                    divergence_error[1]
                ),
            })
        if self.selective_compression_enabled:
            density_selected = int(self.density_selected_count.numpy()[0])
            divergence_selected = int(
                self.divergence_selected_count.numpy()[0]
            )
            fluid_count = max(self.last_fluid_particle_count, 1)
            self.last_density_selected_count = density_selected
            self.last_divergence_selected_count = divergence_selected
            result.update({
                "implicit_selective_compression": 1,
                "implicit_density_selected_particles": density_selected,
                "implicit_density_selected_percent": (
                    100.0 * density_selected / fluid_count
                ),
                "implicit_divergence_selected_particles": (
                    divergence_selected
                ),
                "implicit_divergence_selected_percent": (
                    100.0 * divergence_selected / fluid_count
                ),
            })
        return result

    def analyze(
        self,
        radius: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        kind: np.ndarray,
        base_dt: float,
        output_fps: float,
    ) -> dict[str, float | int | str]:
        if not self.enabled:
            return {}
        fluid = kind == 0
        if not np.any(fluid):
            return {}
        minimum_spacing = max(float(np.min(radius[fluid])) * 2.0, 1.0e-5)
        maximum_speed = max(
            float(np.max(np.linalg.norm(velocity[fluid], axis=1))), 1.0e-5
        )
        maximum_acceleration = max(
            float(np.max(np.linalg.norm(acceleration[fluid], axis=1))), 1.0e-5
        )
        advection_dt = self.advection_cfl * minimum_spacing / maximum_speed
        acceleration_dt = math.sqrt(
            self.acceleration_cfl * minimum_spacing / maximum_acceleration
        )
        requested_dt = min(base_dt * self.target_dt_multiplier, self.maximum_dt)
        recommended_dt = max(
            base_dt, min(requested_dt, advection_dt, acceleration_dt)
        )
        current_substeps = int(math.ceil((1.0 / output_fps) / base_dt))
        predicted_substeps = int(math.ceil((1.0 / output_fps) / recommended_dt))
        self.last_diagnostics = {
            "implicit_fluid_mode": self.mode,
            "implicit_requested_dt_s": requested_dt,
            "implicit_recommended_dt_s": recommended_dt,
            "implicit_advection_limit_s": advection_dt,
            "implicit_acceleration_limit_s": acceleration_dt,
            "implicit_current_substeps_per_frame": current_substeps,
            "implicit_predicted_substeps_per_frame": predicted_substeps,
            "implicit_predicted_step_reduction_percent": 100.0 * (
                1.0 - predicted_substeps / max(current_substeps, 1)
            ),
        }
        return dict(self.last_diagnostics)


class NarrowBandVolumePreparation:
    """Conservative SPH-to-grid eligibility audit for calm interior water."""

    def __init__(
        self,
        policy: dict[str, Any],
        cfg: dict[str, Any],
        capacity: int,
        device: str,
    ):
        self.policy = policy
        self.enabled = bool(policy.get("enabled", False))
        self.mode = str(policy.get("mode", "diagnostic"))
        self.device = device
        self.cell_size = max(0.5, float(policy.get("cell_size", 4.0)))
        self.detail_distance = max(
            self.cell_size * 0.25, float(policy.get("detail_distance", 2.0))
        )
        self.maximum_velocity_rms = max(
            0.1,
            float(
                policy.get(
                    "maximum_velocity_rms",
                    policy.get(
                        "maximum_velocity_delta",
                        policy.get("maximum_interior_speed", 3.0),
                    ),
                )
            ),
        )
        self.analyze_every_frames = max(
            1, int(policy.get("analyze_every_frames", 24))
        )
        self.lower = np.asarray(
            [
                -0.5 * float(cfg["domain_width"]),
                0.0,
                float(cfg["reservoir_z_min"]),
            ],
            dtype=np.float32,
        )
        upper = np.asarray(
            [
                0.5 * float(cfg["domain_width"]),
                float(cfg["domain_y_max"]),
                float(cfg["domain_z_max"]),
            ],
            dtype=np.float32,
        )
        domain_span = upper - self.lower
        shape = np.maximum(1, np.ceil(domain_span / self.cell_size)).astype(
            np.int32
        )
        self.nx, self.ny, self.nz = map(int, shape)
        allocation_shape = (
            (self.nx, self.ny, self.nz) if self.enabled else (1, 1, 1)
        )
        self.grid_mass = wp.zeros(allocation_shape, dtype=float, device=device)
        self.grid_volume = wp.zeros(allocation_shape, dtype=float, device=device)
        self.grid_momentum_x = wp.zeros(
            allocation_shape, dtype=float, device=device
        )
        self.grid_momentum_y = wp.zeros(
            allocation_shape, dtype=float, device=device
        )
        self.grid_momentum_z = wp.zeros(
            allocation_shape, dtype=float, device=device
        )
        self.interior_mask = wp.zeros(
            capacity if self.enabled else 1, dtype=wp.int32, device=device
        )
        self.counters = wp.zeros(1, dtype=wp.int32, device=device)
        self.neighbour_grid = None
        self.neighbour_grid_dims = (0, 0, 0)
        self.last_diagnostics: dict[str, float | int | str] = {}
        if self.enabled:
            # HashGrid dimensions must follow the neighbour-search cell width,
            # not the coarser volume-grid cell size.  Reusing `shape` caused
            # severe hash aliasing for a 1--2 m detail band inside 4 m volume
            # cells and produced non-monotonic, overly conservative masks.
            neighbour_shape = np.maximum(
                1, np.ceil(domain_span / self.detail_distance)
            ).astype(np.int32)
            dims = np.maximum(16, neighbour_shape + 8)
            self.neighbour_grid_dims = tuple(map(int, dims))
            self.neighbour_grid = wp.HashGrid(
                int(dims[0]), int(dims[1]), int(dims[2]), device=device
            )

    def analyze(self, arrays: dict[str, wp.array], count: int) -> None:
        if not self.enabled or count <= 0 or self.neighbour_grid is None:
            return
        self.grid_mass.zero_()
        self.grid_volume.zero_()
        self.grid_momentum_x.zero_()
        self.grid_momentum_y.zero_()
        self.grid_momentum_z.zero_()
        self.counters.zero_()
        positions = arrays["x"][:count]
        self.neighbour_grid.build(positions, self.detail_distance)
        wp.launch(
            classify_and_deposit_narrow_band_interior, dim=count,
            inputs=[
                self.neighbour_grid.id, positions, arrays["v"][:count],
                arrays["mass"][:count], arrays["volume"][:count],
                arrays["kind"][:count], arrays["water_surface_mask"][:count],
                arrays["water_phase"][:count], self.interior_mask[:count],
                self.grid_mass, self.grid_volume, self.grid_momentum_x,
                self.grid_momentum_y, self.grid_momentum_z, self.counters,
                wp.vec3(*map(float, self.lower)), 1.0 / self.cell_size,
                self.nx, self.ny, self.nz, self.detail_distance,
                self.maximum_velocity_rms,
            ], device=self.device,
        )

    def diagnostics(self, total_fluid_particles: int) -> dict[str, float | int | str]:
        if not self.enabled:
            return {}
        interior_particles = int(self.counters.numpy()[0])
        volume = self.grid_volume.numpy()
        active_cells = int(np.count_nonzero(volume > 0.0))
        self.last_diagnostics = {
            "narrow_band_mode": self.mode,
            "narrow_band_interior_particles": interior_particles,
            "narrow_band_interior_fraction_percent": 100.0
            * interior_particles
            / max(total_fluid_particles, 1),
            "narrow_band_active_grid_cells": active_cells,
            "narrow_band_grid_cells": self.nx * self.ny * self.nz,
            "narrow_band_grid_volume_m3": float(
                np.sum(volume, dtype=np.float64)
            ),
            "narrow_band_grid_mass_kg": float(
                np.sum(self.grid_mass.numpy(), dtype=np.float64)
            ),
        }
        return dict(self.last_diagnostics)
