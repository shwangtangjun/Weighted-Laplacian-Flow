"""Shared distributions, diagnostics, and plotting helpers for WLF experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

TWO_PI = 2.0 * np.pi
VON_MISES_KAPPA = 4.0
CAUCHY_SCALE = 1.0
CAUCHY_DOMAIN_MIN = -15.0
CAUCHY_DOMAIN_MAX = 15.0
CORNER_CENTER = 10.0
CORNER_HALF_WIDTH = 3.75


def domain_bounds(target: str) -> tuple[float, float]:
    if target == "bimodal":
        return 0.0, TWO_PI
    elif target == "cauchy":
        return CAUCHY_DOMAIN_MIN, CAUCHY_DOMAIN_MAX
    else:
        raise ValueError(f"unknown target distribution {target!r}")


def signed_power(values, exponent: float):
    """Return sign(values) * abs(values)**exponent for NumPy or PyTorch arrays."""

    if isinstance(values, np.ndarray):
        return np.sign(values) * np.abs(values) ** exponent
    return values.sign() * values.abs().pow(exponent)


def bimodal_means(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    first_mean = np.full(dimension, 0.5 * TWO_PI)
    second_mean = np.full(dimension, 0.5 * TWO_PI)
    first_mean[:2] = (0.25 * TWO_PI, 0.75 * TWO_PI)
    second_mean[:2] = (0.75 * TWO_PI, 0.25 * TWO_PI)
    return first_mean, second_mean


def log_target_numpy(points: np.ndarray, target: str) -> np.ndarray:
    """Evaluate the unnormalized log target density."""

    if target == "bimodal":
        first_mean, second_mean = bimodal_means(points.shape[-1])
        first_mode = VON_MISES_KAPPA * np.cos(points - first_mean)
        second_mode = VON_MISES_KAPPA * np.cos(points - second_mean)
        return np.logaddexp(np.log(0.5) + first_mode.sum(axis=-1), np.log(0.5) + second_mode.sum(axis=-1))
    elif target == "cauchy":
        return -np.log1p((points / CAUCHY_SCALE) ** 2).sum(axis=-1)
    else:
        raise ValueError(f"unknown target distribution {target!r}")


def grad_log_target_numpy(points: np.ndarray, target: str) -> np.ndarray:
    """Evaluate the gradient of the log target density."""

    if target == "bimodal":
        first_mean, second_mean = bimodal_means(points.shape[-1])
        first_log = np.log(0.5) + (VON_MISES_KAPPA * np.cos(points - first_mean)).sum(axis=-1)
        second_log = np.log(0.5) + (VON_MISES_KAPPA * np.cos(points - second_mean)).sum(axis=-1)
        total_log = np.logaddexp(first_log, second_log)
        first_responsibility = np.exp(first_log - total_log)
        first_grad = -VON_MISES_KAPPA * np.sin(points - first_mean)
        second_grad = -VON_MISES_KAPPA * np.sin(points - second_mean)
        return first_responsibility[..., None] * first_grad + (1.0 - first_responsibility[..., None]) * second_grad
    elif target == "cauchy":
        return -2.0 * points / (CAUCHY_SCALE**2 + points**2)
    else:
        raise ValueError(f"unknown target distribution {target!r}")


def log_target_torch(points: "torch.Tensor", target: str) -> "torch.Tensor":
    """Torch counterpart of :func:`log_target_numpy` for mesh-free training."""

    if target != "bimodal":
        raise ValueError(f"mesh-free target does not support {target!r}")
    first_mean, second_mean = bimodal_means(points.shape[-1])
    first_mean_tensor = torch.as_tensor(first_mean, dtype=points.dtype, device=points.device)
    second_mean_tensor = torch.as_tensor(second_mean, dtype=points.dtype, device=points.device)
    first_mode = VON_MISES_KAPPA * torch.cos(points - first_mean_tensor)
    second_mode = VON_MISES_KAPPA * torch.cos(points - second_mean_tensor)
    log_half = float(np.log(0.5))
    return torch.logaddexp(log_half + first_mode.sum(dim=-1), log_half + second_mode.sum(dim=-1))


def log_initial_torch(points: "torch.Tensor", initial: str) -> "torch.Tensor":
    """Log initial density up to an irrelevant additive constant."""

    if initial == "uniform":
        return torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    elif initial == "unimodal":
        first_mean, _ = bimodal_means(points.shape[-1])
        first_mean_tensor = torch.as_tensor(first_mean, dtype=points.dtype, device=points.device)
        return (VON_MISES_KAPPA * torch.cos(points - first_mean_tensor)).sum(dim=-1)
    else:
        raise ValueError(f"mesh-free initialization does not support {initial!r}")


def sample_initial(rng: np.random.Generator, count: int, dimension: int, initial: str, target: str) -> np.ndarray:
    """Draw particles from the selected initial distribution."""

    lower, upper = domain_bounds(target)
    shape = (count, dimension)
    if initial == "uniform":
        return rng.uniform(lower, upper, size=shape)
    elif initial == "unimodal":
        first_mean, _ = bimodal_means(dimension)
        return rng.vonmises(first_mean, VON_MISES_KAPPA, size=shape) % TWO_PI
    elif initial == "corner":
        center = np.full(dimension, CORNER_CENTER)
        return rng.uniform(center - CORNER_HALF_WIDTH, center + CORNER_HALF_WIDTH, size=shape)
    else:
        raise ValueError(f"unknown initial distribution {initial!r}")


def initial_density_on_grid(coords: np.ndarray, initial: str, target: str) -> np.ndarray:
    """Evaluate q0 on a two-dimensional grid."""

    lower, upper = domain_bounds(target)
    if initial == "uniform":
        return np.full(coords.shape[:-1], 1.0 / (upper - lower) ** 2)
    elif initial == "unimodal":
        first_mean, _ = bimodal_means(2)
        log_unnormalized = (VON_MISES_KAPPA * np.cos(coords - first_mean)).sum(axis=-1)
        normalizer = (TWO_PI * np.i0(VON_MISES_KAPPA)) ** 2
        return np.exp(log_unnormalized) / normalizer
    elif initial == "corner":
        center = np.full(2, CORNER_CENTER)
        inside = np.all(np.abs(coords - center) <= CORNER_HALF_WIDTH, axis=-1)
        area = (2.0 * CORNER_HALF_WIDTH) ** 2
        return np.where(inside, 1.0 / area, 1e-12)
    else:
        raise ValueError(f"unknown initial distribution {initial!r}")


def target_density_on_grid(coords: np.ndarray, cell_volume: float, target: str) -> np.ndarray:
    """Return the normalized target density on a 2D grid or 2D marginal."""

    if target == "bimodal":
        first_mean, second_mean = bimodal_means(2)
        first_log = (VON_MISES_KAPPA * np.cos(coords - first_mean)).sum(axis=-1)
        second_log = (VON_MISES_KAPPA * np.cos(coords - second_mean)).sum(axis=-1)
        normalizer = (TWO_PI * np.i0(VON_MISES_KAPPA)) ** 2
        return (0.5 * np.exp(first_log) + 0.5 * np.exp(second_log)) / normalizer
    elif target == "cauchy":
        unnormalized = np.exp(log_target_numpy(coords, target))
        return unnormalized / (unnormalized.sum() * cell_volume)
    else:
        raise ValueError(f"unknown target distribution {target!r}")


def sample_target_projection(rng: np.random.Generator, count: int, target: str) -> np.ndarray:
    """Draw the first two coordinates of target particles for KDE calibration."""

    if target == "bimodal":
        first_mean, second_mean = bimodal_means(2)
        first_count = int(rng.binomial(count, 0.5))
        return np.vstack(
            (
                rng.vonmises(first_mean, VON_MISES_KAPPA, size=(first_count, 2)) % TWO_PI,
                rng.vonmises(second_mean, VON_MISES_KAPPA, size=(count - first_count, 2)) % TWO_PI,
            )
        )
    elif target == "cauchy":
        angle_min = np.arctan(CAUCHY_DOMAIN_MIN / CAUCHY_SCALE)
        angle_max = np.arctan(CAUCHY_DOMAIN_MAX / CAUCHY_SCALE)
        return CAUCHY_SCALE * np.tan(rng.uniform(angle_min, angle_max, size=(count, 2)))
    else:
        raise ValueError(f"unknown target distribution {target!r}")


def apply_boundary(points: np.ndarray, target: str) -> None:
    """Apply periodic or reflecting boundaries in place."""

    if target == "bimodal":
        points[:] %= TWO_PI
    elif target == "cauchy":
        width = CAUCHY_DOMAIN_MAX - CAUCHY_DOMAIN_MIN
        reflected = (points - CAUCHY_DOMAIN_MIN) % (2.0 * width)
        points[:] = CAUCHY_DOMAIN_MIN + np.where(reflected <= width, reflected, 2.0 * width - reflected)
    else:
        raise ValueError(f"unknown target distribution {target!r}")


def kde_on_grid(particles: np.ndarray, n_grid: int, dx: float, cell_volume: float, target: str, bandwidth: float) -> np.ndarray:
    """Cloud-in-cell deposition followed by a Gaussian KDE filter."""

    if target == "bimodal":
        lower_bound = 0.0
        periodic = True
    elif target == "cauchy":
        lower_bound = CAUCHY_DOMAIN_MIN
        periodic = False
    else:
        raise ValueError(f"unknown target distribution {target!r}")
    counts = np.zeros((n_grid, n_grid))
    offset = 0.0 if periodic else 0.5
    cell_coords = (particles[:, :2] - lower_bound) / dx - offset
    lower = np.floor(cell_coords).astype(int)
    fraction = cell_coords - lower

    if periodic:
        i0 = lower[:, 0] % n_grid
        j0 = lower[:, 1] % n_grid
        i1 = (i0 + 1) % n_grid
        j1 = (j0 + 1) % n_grid
        filter_mode = "wrap"
    else:
        i0 = np.clip(lower[:, 0], 0, n_grid - 1)
        j0 = np.clip(lower[:, 1], 0, n_grid - 1)
        i1 = np.clip(lower[:, 0] + 1, 0, n_grid - 1)
        j1 = np.clip(lower[:, 1] + 1, 0, n_grid - 1)
        filter_mode = "reflect"

    wx, wy = fraction[:, 0], fraction[:, 1]
    np.add.at(counts, (i0, j0), (1.0 - wx) * (1.0 - wy))
    np.add.at(counts, (i1, j0), wx * (1.0 - wy))
    np.add.at(counts, (i0, j1), (1.0 - wx) * wy)
    np.add.at(counts, (i1, j1), wx * wy)

    density = gaussian_filter(counts, sigma=bandwidth / dx, mode=filter_mode)
    density = np.maximum(density, 1e-12)
    return density / (density.sum() * cell_volume)


def kl_divergence(target_density: np.ndarray, approximation: np.ndarray, cell_volume: float) -> float:
    return float(np.sum(target_density * (np.log(target_density) - np.log(approximation))) * cell_volume)


def save_kl_plot(
    output: Path,
    flow_times: list[float],
    flow_kl: list[float],
    langevin_times: list[float],
    langevin_kl: list[float],
    target_kde_kl: float,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(flow_times, flow_kl, label="Weighted Laplacian Flow", color="C0", lw=2)
    ax.plot(langevin_times, langevin_kl, label="Langevin", color="C3", lw=2)
    ax.axhline(target_kde_kl, label="Target", color="0.25", lw=1.6, ls="--")
    ax.set_xlabel("t")
    ax.set_ylabel(r"KL$(p\,\Vert\,q_t)$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_snapshot_plot(
    output: Path,
    snapshots: dict[int, np.ndarray],
    dt: float,
    mesh_grid: tuple[np.ndarray, np.ndarray],
    target_density: np.ndarray,
    target: str,
    comparison_snapshots: dict[int, np.ndarray] | None = None,
) -> None:
    ordered_steps = sorted(snapshots)
    rows = 2 if comparison_snapshots is not None else 1
    if target == "bimodal":
        contour_levels = np.linspace(target_density.min(), target_density.max(), 8)
    elif target == "cauchy":
        contour_levels = target_density.max() * np.geomspace(1e-3, 1.0, 8)
    else:
        raise ValueError(f"unknown target distribution {target!r}")
    lower, upper = domain_bounds(target)

    fig, axes = plt.subplots(
        rows,
        len(ordered_steps),
        figsize=(3.0 * len(ordered_steps), 3.0 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for column, step in enumerate(ordered_steps):
        row_data = [(snapshots[step], "C0", f"WLF  t={step * dt:.2f}")]
        if comparison_snapshots is not None:
            row_data.append((comparison_snapshots[step], "C3", f"Langevin  t={step * dt:.2f}"))
        for row, (particles, color, title) in enumerate(row_data):
            ax = axes[row, column]
            ax.contour(*mesh_grid, target_density, levels=contour_levels, colors="k", alpha=0.5, linewidths=0.7)
            ax.scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.35, color=color)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
