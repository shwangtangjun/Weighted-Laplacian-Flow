"""Grid-based weighted Laplacian flow experiments in two dimensions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized

from utils import (
    apply_boundary,
    domain_bounds,
    grad_log_target_numpy,
    initial_density_on_grid,
    kde_on_grid,
    kl_divergence,
    log_target_numpy,
    sample_initial,
    sample_target_projection,
    save_kl_plot,
    save_snapshot_plot,
    signed_power,
    target_density_on_grid,
)

DIMENSION = 2
N_PARTICLES = 16000
N_GRID = 128
T_FINAL = {
    ("unimodal", "bimodal"): 1.0,
    ("uniform", "bimodal"): 10.0,
    ("uniform", "cauchy"): 10.0,
    ("corner", "cauchy"): 10.0,
}
DT = {
    ("unimodal", "bimodal"): 0.001,
    ("uniform", "bimodal"): 0.01,
    ("uniform", "cauchy"): 0.01,
    ("corner", "cauchy"): 0.01,
}
ALPHA = {
    ("unimodal", "bimodal"): 1.0,
    ("uniform", "bimodal"): 0.0,
    ("uniform", "cauchy"): 1.0,
    ("corner", "cauchy"): 1.0,
}
R = 1.0
SNAPSHOT_COUNT = 5
SEED = 0
INTERPOLATION_ORDER = {"bimodal": 3, "cauchy": 1}
KDE_BANDWIDTH = {"bimodal": 0.15, "cauchy": 0.5}
RESULTS_DIR = Path("results")


class GridOperators:
    """Finite-difference and interpolation operators for one fixed grid."""

    def __init__(self, target: str, n_grid: int, interpolation_order: int):
        self.lower, self.upper = domain_bounds(target)
        self.width = self.upper - self.lower
        self.periodic = target == "bimodal"
        self.n_grid = n_grid
        self.dx = self.width / n_grid
        self.shape = (n_grid, n_grid)
        self.cell_volume = self.dx**2
        self.interpolation_order = interpolation_order

        if self.periodic:
            axis = np.linspace(self.lower, self.upper, n_grid, endpoint=False)
        else:
            axis = self.lower + (np.arange(n_grid) + 0.5) * self.dx
        self.mesh = np.meshgrid(axis, axis, indexing="ij")
        self.coords = np.stack(self.mesh, axis=-1)

    def gradient(self, field: np.ndarray, zero_normal: bool = False) -> np.ndarray:
        if self.periodic:
            grad_x = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * self.dx)
            grad_y = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * self.dx)
        else:
            grad_x = np.gradient(field, self.dx, axis=0, edge_order=2)
            grad_y = np.gradient(field, self.dx, axis=1, edge_order=2)
            if zero_normal:
                grad_x[[0, -1], :] = 0.0
                grad_y[:, [0, -1]] = 0.0
        return np.stack([grad_x, grad_y], axis=-1)

    def interpolate(self, field: np.ndarray, points: np.ndarray) -> np.ndarray:
        offset = 0.0 if self.periodic else 0.5
        coordinates = ((points - self.lower) / self.dx - offset).T
        mode = "grid-wrap" if self.periodic else "reflect"
        if field.ndim == 2:
            return map_coordinates(field, coordinates, order=self.interpolation_order, mode=mode)
        return np.stack(
            [map_coordinates(field[..., component], coordinates, order=self.interpolation_order, mode=mode) for component in range(field.shape[-1])],
            axis=-1,
        )

    def build_weighted_laplacian(self, log_weight: np.ndarray):
        inv_dx2 = 1.0 / self.dx**2
        plus_weights = [0.5 * (1.0 + np.exp(np.roll(log_weight, -1, axis=axis) - log_weight)) for axis in range(2)]
        minus_weights = [0.5 * (1.0 + np.exp(np.roll(log_weight, 1, axis=axis) - log_weight)) for axis in range(2)]
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []

        for index in np.ndindex(self.shape):
            row = np.ravel_multi_index(index, self.shape)
            center = 0.0
            for axis in range(2):
                plus_index = list(index)
                if self.periodic or plus_index[axis] + 1 < self.n_grid:
                    plus_index[axis] = (plus_index[axis] + 1) % self.n_grid
                    coefficient = plus_weights[axis][index] * inv_dx2
                    center -= coefficient
                    rows.append(row)
                    columns.append(np.ravel_multi_index(tuple(plus_index), self.shape))
                    values.append(coefficient)

                minus_index = list(index)
                if self.periodic or minus_index[axis] - 1 >= 0:
                    minus_index[axis] = (minus_index[axis] - 1) % self.n_grid
                    coefficient = minus_weights[axis][index] * inv_dx2
                    center -= coefficient
                    rows.append(row)
                    columns.append(np.ravel_multi_index(tuple(minus_index), self.shape))
                    values.append(coefficient)

            rows.append(row)
            columns.append(row)
            values.append(center)

        size = self.n_grid**2
        return coo_matrix((values, (rows, columns)), shape=(size, size)).tocsc()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a two-dimensional grid-based weighted Laplacian flow experiment.",
        epilog="Valid pairs: unimodal bimodal, uniform bimodal, uniform cauchy, corner cauchy.",
    )
    parser.add_argument("--initial", choices=("unimodal", "uniform", "corner"), default="unimodal")
    parser.add_argument("--target", choices=("bimodal", "cauchy"), default="bimodal")
    args = parser.parse_args()

    experiment_key = (args.initial, args.target)
    if experiment_key not in ALPHA:
        raise ValueError(
            f"unsupported experiment {args.initial!r} -> {args.target!r}; "
            "valid pairs: unimodal bimodal, uniform bimodal, uniform cauchy, corner cauchy"
        )
    tag = f"{DIMENSION}d_{args.initial}2{args.target}"
    alpha = ALPHA[experiment_key]
    t_final = T_FINAL[experiment_key]
    dt = DT[experiment_key]
    n_steps = int(round(t_final / dt))
    grid = GridOperators(args.target, N_GRID, INTERPOLATION_ORDER[args.target])
    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_rho_grid = log_target_numpy(grid.coords, args.target)
    rho_grid = np.exp(log_rho_grid)
    grad_log_rho_grid = grad_log_target_numpy(grid.coords, args.target)
    target_grid = target_density_on_grid(grid.coords, grid.cell_volume, args.target)
    initial_grid = initial_density_on_grid(grid.coords, args.initial, args.target)
    omega_grid = log_rho_grid - np.log(initial_grid)

    weighted_laplacian = grid.build_weighted_laplacian(log_rho_grid)
    reduced_solver = factorized(weighted_laplacian[1:, 1:].tocsc())

    def solve_weighted_poisson(rhs: np.ndarray) -> np.ndarray:
        phi = np.empty(N_GRID**2)
        phi[0] = 0.0
        phi[1:] = reduced_solver(rhs.ravel()[1:])
        phi -= phi.mean()
        return phi.reshape(grid.shape)

    def density(particles: np.ndarray) -> np.ndarray:
        return kde_on_grid(particles, N_GRID, grid.dx, grid.cell_volume, args.target, KDE_BANDWIDTH[args.target])

    def kl(approximation: np.ndarray) -> float:
        return kl_divergence(target_grid, approximation, grid.cell_volume)

    initial_particles = sample_initial(rng, N_PARTICLES, DIMENSION, args.initial, args.target)
    flow_particles = initial_particles.copy()
    langevin_particles = initial_particles.copy()
    initial_kl = kl(density(initial_particles))
    flow_kl = [initial_kl]
    langevin_kl = [initial_kl]
    flow_times = [0.0]
    langevin_times = [0.0]

    snapshot_steps = sorted(set(int(round(time / dt)) for time in np.linspace(0.0, t_final, SNAPSHOT_COUNT)))
    snapshots = {0: initial_particles.copy()}
    langevin_snapshots = {0: initial_particles.copy()}
    target_particles = sample_target_projection(rng, N_PARTICLES, args.target)
    target_kde_kl = kl(density(target_particles))

    print(f"experiment = {tag}, steps = {n_steps}, dt = {dt:g}, particles = {N_PARTICLES}")
    print(f"Target KDE = {target_kde_kl:.4e}")

    for step in range(1, n_steps + 1):
        # ------------------------------------------------------------------
        # 1. Pre-processing
        # ------------------------------------------------------------------
        omega_bar = np.sum(rho_grid * omega_grid) / np.sum(rho_grid)
        nonlinear_omega = signed_power(omega_grid - omega_bar, R)
        integral_correction = np.sum(rho_grid * nonlinear_omega) / np.sum(rho_grid)
        source_grid = -nonlinear_omega + integral_correction

        # ------------------------------------------------------------------
        # 2. Poisson step
        # ------------------------------------------------------------------
        phi_grid = solve_weighted_poisson(source_grid)
        velocity_grid = grid.gradient(phi_grid, zero_normal=not grid.periodic)

        # ------------------------------------------------------------------
        # 3. Transport step
        # ------------------------------------------------------------------
        previous_omega_grid = omega_grid
        back_points = (grid.coords - dt * velocity_grid).reshape(-1, 2)
        apply_boundary(back_points, args.target)
        omega_back = grid.interpolate(previous_omega_grid, back_points).reshape(grid.shape)
        source_back = grid.interpolate(source_grid, back_points).reshape(grid.shape)
        omega_grid = omega_back + dt * source_back

        # ------------------------------------------------------------------
        # 4. Update step
        # ------------------------------------------------------------------
        velocity_particles = grid.interpolate(velocity_grid, flow_particles)
        drift_grid = grad_log_rho_grid - grid.gradient(previous_omega_grid)
        drift_particles = grid.interpolate(drift_grid, flow_particles)
        noise = rng.standard_normal(size=flow_particles.shape)
        flow_particles += dt * velocity_particles + alpha * dt * drift_particles + np.sqrt(2.0 * alpha * dt) * noise
        apply_boundary(flow_particles, args.target)

        # ------------------------------------------------------------------
        # Langevin baseline
        # ------------------------------------------------------------------
        noise = rng.standard_normal(size=langevin_particles.shape)
        langevin_drift = grid.interpolate(grad_log_rho_grid, langevin_particles)
        langevin_particles += dt * langevin_drift + np.sqrt(2.0 * dt) * noise
        apply_boundary(langevin_particles, args.target)

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        flow_kl.append(kl(density(flow_particles)))
        langevin_kl.append(kl(density(langevin_particles)))
        flow_times.append(step * dt)
        langevin_times.append(step * dt)

        if step in snapshot_steps:
            snapshots[step] = flow_particles.copy()
            langevin_snapshots[step] = langevin_particles.copy()
            print(
                f"Step {step:4d}  flow_t={step * dt:5.2f}  langevin_t={step * dt:5.2f}  "
                f"KL(p||q_wlf)={flow_kl[-1]:.4f}  KL(p||q_langevin)={langevin_kl[-1]:.4f}"
            )

    kl_path = RESULTS_DIR / f"kl_{tag}.png"
    snapshots_path = RESULTS_DIR / f"snapshots_{tag}.png"
    save_kl_plot(kl_path, flow_times, flow_kl, langevin_times, langevin_kl, target_kde_kl)
    save_snapshot_plot(snapshots_path, snapshots, dt, grid.mesh, target_grid, args.target, comparison_snapshots=langevin_snapshots)

    print(f"Final KL(p||q_wlf) = {flow_kl[-1]}")
    print(f"Final KL(p||q_langevin) = {langevin_kl[-1]}")
    print(f"Saved {kl_path} and {snapshots_path}")


if __name__ == "__main__":
    main()
