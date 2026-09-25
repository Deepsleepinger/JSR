"""
Domain decomposition and overlapping subdomain partition module.

Generates physical coordinate-based subdomains with explicit overlap layers
and partition-of-unity weighting for Restricted Additive Schwarz (RAS)
and Symmetric Weighted Additive Schwarz (AS).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple
import numpy as np


def create_3d_overlapping_partition(
    coordinates: np.ndarray,
    grid: Tuple[int, int, int] = (2, 2, 2),
    overlap_layers: int = 1,
    mesh_n: int = 48,
    domain_bounds: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1.0e-12,
) -> Dict[str, object]:
    """
    Construct an overlapping 3-D Cartesian partition from spatial coordinates.

    Parameters
    ----------
    coordinates : np.ndarray of shape (N, 3)
        Spatial coordinates of free DOFs.
    grid : (kx, ky, kz)
        Number of subdomains along each coordinate axis (e.g. (2, 2, 2) = 8 octants).
    overlap_layers : int
        Number of mesh elements of overlap to add in each coordinate direction (delta >= 1).
    mesh_n : int
        Underlying grid resolution N for element size h = 1 / N.
    domain_bounds : (lo, hi)
        Coordinate bounding box.
    eps : float
        Numerical tolerance for bounding box boundary inclusion.

    Returns
    -------
    dict containing:
        - 'core_ids': array of shape (N,) assigning each DOF to exactly one core subdomain.
        - 'local_indices': list of 1D arrays with overlapping DOF indices for each subdomain.
        - 'memberships': array of shape (N,) indicating how many subdomains contain each DOF.
        - 'weights': array of shape (N,) with partition-of-unity weights (1.0 / memberships).
        - 'subdomain_count': total number of subdomains (kx * ky * kz).
    """
    kx, ky, kz = grid
    total_subdomains = kx * ky * kz
    n_dofs = coordinates.shape[0]

    lo_b, hi_b = domain_bounds
    dom_len = hi_b - lo_b
    h = dom_len / float(mesh_n)

    # 1. Non-overlapping core assignment
    # Map coordinates to [0, kx), [0, ky), [0, kz)
    norm_coords = (coordinates - lo_b) / dom_len
    bin_x = np.floor(np.minimum(norm_coords[:, 0] * kx, kx - eps)).astype(np.int64)
    bin_y = np.floor(np.minimum(norm_coords[:, 1] * ky, ky - eps)).astype(np.int64)
    bin_z = np.floor(np.minimum(norm_coords[:, 2] * kz, kz - eps)).astype(np.int64)

    core_ids = bin_x + kx * (bin_y + ky * bin_z)
    if np.any(core_ids < 0) or np.any(core_ids >= total_subdomains):
        raise ValueError("Invalid core subdomain assignment found in coordinates.")

    # 2. Overlapping subdomain index expansion
    local_indices: List[np.ndarray] = []
    memberships = np.zeros(n_dofs, dtype=np.int64)

    dx = dom_len / float(kx)
    dy = dom_len / float(ky)
    dz = dom_len / float(kz)
    overlap_dist = overlap_layers * h

    for iz in range(kz):
        for iy in range(ky):
            for ix in range(kx):
                sub_lo = np.array([
                    lo_b + ix * dx - overlap_dist,
                    lo_b + iy * dy - overlap_dist,
                    lo_b + iz * dz - overlap_dist,
                ], dtype=np.float64)
                sub_hi = np.array([
                    lo_b + (ix + 1) * dx + overlap_dist,
                    lo_b + (iy + 1) * dy + overlap_dist,
                    lo_b + (iz + 1) * dz + overlap_dist,
                ], dtype=np.float64)

                sub_lo = np.maximum(sub_lo, lo_b)
                sub_hi = np.minimum(sub_hi, hi_b)

                in_sub = np.all(
                    (coordinates >= sub_lo - eps) & (coordinates <= sub_hi + eps),
                    axis=1,
                )
                idx = np.flatnonzero(in_sub).astype(np.int64)
                if idx.size == 0:
                    raise RuntimeError(f"Empty overlapping subdomain at grid ({ix}, {iy}, {iz})")
                local_indices.append(idx)
                memberships[idx] += 1

    if np.any(memberships <= 0):
        raise RuntimeError("One or more DOFs are not covered by any overlapping subdomain.")

    # 3. Partition of unity weights
    weights = 1.0 / memberships.astype(np.float64)

    return {
        "core_ids": core_ids,
        "local_indices": local_indices,
        "memberships": memberships,
        "weights": weights,
        "subdomain_count": total_subdomains,
        "overlap_layers": overlap_layers,
        "grid": grid,
    }


def create_2d_overlapping_partition(
    coordinates: np.ndarray,
    grid: Tuple[int, int] = (4, 4),
    overlap_layers: int = 1,
    mesh_n: int = 48,
    domain_bounds: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1.0e-12,
) -> Dict[str, object]:
    """2-D counterpart of Cartesian overlapping partition."""
    kx, ky = grid
    total_subdomains = kx * ky
    n_dofs = coordinates.shape[0]

    lo_b, hi_b = domain_bounds
    dom_len = hi_b - lo_b
    h = dom_len / float(mesh_n)

    norm_coords = (coordinates - lo_b) / dom_len
    bin_x = np.floor(np.minimum(norm_coords[:, 0] * kx, kx - eps)).astype(np.int64)
    bin_y = np.floor(np.minimum(norm_coords[:, 1] * ky, ky - eps)).astype(np.int64)
    core_ids = bin_x + kx * bin_y

    local_indices: List[np.ndarray] = []
    memberships = np.zeros(n_dofs, dtype=np.int64)

    dx = dom_len / float(kx)
    dy = dom_len / float(ky)
    overlap_dist = overlap_layers * h

    for iy in range(ky):
        for ix in range(kx):
            sub_lo = np.array([lo_b + ix * dx - overlap_dist, lo_b + iy * dy - overlap_dist])
            sub_hi = np.array([lo_b + (ix + 1) * dx + overlap_dist, lo_b + (iy + 1) * dy + overlap_dist])
            sub_lo = np.maximum(sub_lo, lo_b)
            sub_hi = np.minimum(sub_hi, hi_b)

            in_sub = np.all((coordinates >= sub_lo - eps) & (coordinates <= sub_hi + eps), axis=1)
            idx = np.flatnonzero(in_sub).astype(np.int64)
            local_indices.append(idx)
            memberships[idx] += 1

    weights = 1.0 / memberships.astype(np.float64)
    return {
        "core_ids": core_ids,
        "local_indices": local_indices,
        "memberships": memberships,
        "weights": weights,
        "subdomain_count": total_subdomains,
        "overlap_layers": overlap_layers,
        "grid": grid,
    }
