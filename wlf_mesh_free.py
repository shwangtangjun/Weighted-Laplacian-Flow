"""Mesh-free weighted Laplacian flow experiments in ten dimensions."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils import (
    apply_boundary,
    domain_bounds,
    grad_log_target_numpy,
    kde_on_grid,
    kl_divergence,
    log_initial_torch,
    log_target_torch,
    sample_initial,
    sample_target_projection,
    save_kl_plot,
    save_snapshot_plot,
    signed_power,
    target_density_on_grid,
)

DIMENSION = 10
N_PARTICLES = 16000
T_FINAL = 4.0
DT = 0.01
ALPHA = 1.0
R = 1.0
SNAPSHOT_COUNT = 5
SEED = 0

BATCH_SIZE = 2**15
ESTIMATE_SIZE = 100000
INITIAL_ITERATIONS = 200
POTENTIAL_ITERATIONS = 50
OMEGA_ITERATIONS = 50
HIDDEN_DIM = 128
HIDDEN_LAYERS = 2
FOURIER_MODES = 4
DEVICE = "auto"

N_GRID = 128
KDE_BANDWIDTH = 0.15
RESULTS_DIR = Path("results")


class PeriodicScalarNet(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, hidden_layers: int, fourier_modes: int, period: float):
        super().__init__()
        self.period = period
        self.register_buffer("modes", torch.arange(1, fourier_modes + 1, dtype=torch.float32))
        input_dim = 2 * dimension * fourier_modes
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * np.pi * points.unsqueeze(-1) * self.modes / self.period
        features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).reshape(points.shape[0], -1)
        return self.network(features).squeeze(-1)


def scalar_gradient(network: nn.Module, points: np.ndarray | torch.Tensor, device: torch.device, return_numpy: bool = True):
    if isinstance(points, np.ndarray):
        differentiable_points = torch.as_tensor(points, device=device, dtype=torch.float32).detach().clone().requires_grad_()
    else:
        differentiable_points = points.detach().clone().requires_grad_()
    gradient = torch.autograd.grad(network(differentiable_points).sum(), differentiable_points)[0]
    if return_numpy:
        return gradient.detach().cpu().numpy()
    return gradient.detach()


def uniform_points(count: int, lower: float, width: float, device: torch.device) -> torch.Tensor:
    return lower + width * torch.rand((count, DIMENSION), device=device)


@torch.no_grad()
def estimate_source_constants(
    omega_net: nn.Module,
    target: str,
    lower: float,
    width: float,
    device: torch.device,
    estimate_size: int,
    batch_size: int,
    power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_rho_batches: list[torch.Tensor] = []
    omega_batches: list[torch.Tensor] = []
    for start in range(0, estimate_size, batch_size):
        points = uniform_points(min(batch_size, estimate_size - start), lower, width, device)
        log_rho_batches.append(log_target_torch(points, target))
        omega_batches.append(omega_net(points))

    log_rho = torch.cat(log_rho_batches)
    omega = torch.cat(omega_batches)
    weights = torch.softmax(log_rho, dim=0)
    omega_bar = torch.sum(weights * omega)
    nonlinear_omega = signed_power(omega - omega_bar, power)
    integral_correction = torch.sum(weights * nonlinear_omega)
    return omega_bar, integral_correction


def source_term(omega: torch.Tensor, omega_bar: torch.Tensor, integral_correction: torch.Tensor, power: float) -> torch.Tensor:
    return -signed_power(omega - omega_bar, power) + integral_correction


def train_potential(
    phi_net: nn.Module,
    omega_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    omega_bar: torch.Tensor,
    integral_correction: torch.Tensor,
    target: str,
    lower: float,
    width: float,
    device: torch.device,
    batch_size: int,
    iterations: int,
    power: float,
) -> None:
    for _ in range(iterations):
        points = uniform_points(batch_size, lower, width, device).requires_grad_()
        with torch.no_grad():
            log_rho = log_target_torch(points, target)
            weights = torch.softmax(log_rho, dim=0)
            source = source_term(omega_net(points), omega_bar, integral_correction, power)

        phi = phi_net(points)
        velocity = torch.autograd.grad(phi.sum(), points, create_graph=True)[0]
        loss = torch.sum(weights * (0.5 * torch.sum(velocity**2, dim=1) + source * phi))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_omega(
    omega_net: nn.Module,
    phi_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    omega_bar: torch.Tensor,
    integral_correction: torch.Tensor,
    lower: float,
    width: float,
    device: torch.device,
    batch_size: int,
    iterations: int,
    dt: float,
    power: float,
) -> nn.Module:
    old_omega_net = copy.deepcopy(omega_net).eval()
    old_omega_net.requires_grad_(False)

    for _ in range(iterations):
        points = uniform_points(batch_size, lower, width, device)
        velocity = scalar_gradient(phi_net, points, device, return_numpy=False)
        with torch.no_grad():
            back_points = torch.remainder(points - dt * velocity - lower, width) + lower
            omega_back = old_omega_net(back_points)
            target = omega_back + dt * source_term(omega_back, omega_bar, integral_correction, power)

        loss = torch.mean((omega_net(points) - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return old_omega_net


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ten-dimensional mesh-free weighted Laplacian flow experiment.")
    parser.add_argument("--initial", choices=("unimodal", "uniform"), default="unimodal")
    parser.add_argument("--target", choices=("bimodal",), default="bimodal")
    args = parser.parse_args()

    lower, upper = domain_bounds(args.target)
    width = upper - lower
    tag = f"{DIMENSION}d_{args.initial}2{args.target}"
    n_steps = int(round(T_FINAL / DT))
    device = choose_device(DEVICE)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dx = width / N_GRID
    cell_volume = dx**2
    axis = np.linspace(lower, upper, N_GRID, endpoint=False)
    mesh_grid = np.meshgrid(axis, axis, indexing="ij")
    coords_grid = np.stack(mesh_grid, axis=-1)
    target_grid = target_density_on_grid(coords_grid, cell_volume, args.target)

    def density(particles: np.ndarray) -> np.ndarray:
        return kde_on_grid(particles, N_GRID, dx, cell_volume, args.target, bandwidth=KDE_BANDWIDTH)

    def kl(approximation: np.ndarray) -> float:
        return kl_divergence(target_grid, approximation, cell_volume)

    initial_particles = sample_initial(rng, N_PARTICLES, DIMENSION, args.initial, args.target)
    flow_particles = initial_particles.copy()
    langevin_particles = initial_particles.copy()
    initial_kl = kl(density(initial_particles[:, :2]))
    flow_kl = [initial_kl]
    langevin_kl = [initial_kl]
    flow_times = [0.0]
    langevin_times = [0.0]

    snapshot_steps = sorted(set(int(round(time / DT)) for time in np.linspace(0.0, T_FINAL, SNAPSHOT_COUNT)))
    snapshots = {0: initial_particles[:, :2].copy()}
    target_particles = sample_target_projection(rng, N_PARTICLES, args.target)
    target_kde_kl = kl(density(target_particles))

    phi_net = PeriodicScalarNet(DIMENSION, HIDDEN_DIM, HIDDEN_LAYERS, FOURIER_MODES, width).to(device)
    omega_net = PeriodicScalarNet(DIMENSION, HIDDEN_DIM, HIDDEN_LAYERS, FOURIER_MODES, width).to(device)
    phi_optimizer = torch.optim.Adam(phi_net.parameters(), lr=1e-3)
    omega_optimizer = torch.optim.Adam(omega_net.parameters(), lr=1e-3)

    # Warmup phase
    for _ in range(INITIAL_ITERATIONS):
        points = uniform_points(BATCH_SIZE, lower, width, device)
        target_omega = log_target_torch(points, args.target) - log_initial_torch(points, args.initial)
        loss = torch.mean((omega_net(points) - target_omega) ** 2)
        omega_optimizer.zero_grad()
        loss.backward()
        omega_optimizer.step()

    print(f"experiment = {tag}, device = {device}, steps = {n_steps}, " f"dt = {DT:g}, particles = {N_PARTICLES}")
    print(f"Target KDE = {target_kde_kl:.4e}")

    for step in range(1, n_steps + 1):
        # ------------------------------------------------------------------
        # 1. Pre-processing
        # ------------------------------------------------------------------
        omega_bar, integral_correction = estimate_source_constants(omega_net, args.target, lower, width, device, ESTIMATE_SIZE, BATCH_SIZE, R)

        # ------------------------------------------------------------------
        # 2. Poisson step
        # ------------------------------------------------------------------
        train_potential(
            phi_net, omega_net, phi_optimizer, omega_bar, integral_correction, args.target, lower, width, device, BATCH_SIZE, POTENTIAL_ITERATIONS, R
        )

        # ------------------------------------------------------------------
        # 3. Transport step
        # ------------------------------------------------------------------
        previous_omega_net = train_omega(
            omega_net, phi_net, omega_optimizer, omega_bar, integral_correction, lower, width, device, BATCH_SIZE, OMEGA_ITERATIONS, DT, R
        )

        # ------------------------------------------------------------------
        # 4. Update step
        # ------------------------------------------------------------------
        velocity_particles = scalar_gradient(phi_net, flow_particles, device)
        grad_omega_particles = scalar_gradient(previous_omega_net, flow_particles, device)
        drift_particles = grad_log_target_numpy(flow_particles, args.target) - grad_omega_particles
        noise = rng.standard_normal(size=flow_particles.shape)
        flow_particles += DT * velocity_particles + ALPHA * DT * drift_particles + np.sqrt(2.0 * ALPHA * DT) * noise
        apply_boundary(flow_particles, args.target)

        # ------------------------------------------------------------------
        # Langevin baseline
        # ------------------------------------------------------------------
        noise = rng.standard_normal(size=langevin_particles.shape)
        langevin_particles += DT * grad_log_target_numpy(langevin_particles, args.target) + np.sqrt(2.0 * DT) * noise
        apply_boundary(langevin_particles, args.target)

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        flow_kl.append(kl(density(flow_particles[:, :2])))
        langevin_kl.append(kl(density(langevin_particles[:, :2])))
        flow_times.append(step * DT)
        langevin_times.append(step * DT)

        if step in snapshot_steps:
            snapshots[step] = flow_particles[:, :2].copy()
            print(
                f"Step {step:4d}  flow_t={step * DT:5.2f}  langevin_t={step * DT:5.2f}  "
                f"KL(p||q_wlf)={flow_kl[-1]:.4f}  KL(p||q_langevin)={langevin_kl[-1]:.4f}"
            )

    kl_path = RESULTS_DIR / f"kl_{tag}.png"
    snapshots_path = RESULTS_DIR / f"snapshots_{tag}.png"
    save_kl_plot(kl_path, flow_times, flow_kl, langevin_times, langevin_kl, target_kde_kl, title=r"Convergence of $q_t$ to $p$")
    save_snapshot_plot(snapshots_path, snapshots, DT, mesh_grid, target_grid, args.target)

    print(f"Final KL(p||q_wlf) = {flow_kl[-1]}")
    print(f"Final KL(p||q_langevin) = {langevin_kl[-1]}")
    print(f"Saved {kl_path} and {snapshots_path}")


if __name__ == "__main__":
    main()
