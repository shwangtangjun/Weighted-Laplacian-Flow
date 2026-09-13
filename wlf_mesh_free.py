"""Mesh-free weighted Laplacian flow experiments in ten dimensions."""

from __future__ import annotations

import argparse
import copy
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils import (
    choose_device,
    grad_log_target_torch,
    scalar_gradient,
    uniform_points,
    domain_bounds,
    kde_on_grid,
    kl_divergence,
    log_initial_torch,
    log_target_torch,
    sample_initial,
    sample_target_projection,
    save_kl_plot,
    save_snapshot_plot,
    make_snapshot_steps,
    target_density_on_grid,
)

DIMENSION = 10
N_PARTICLES = 16000
T_FINAL = 10.0
DT = 0.01
ALPHA = 1.0
# Integer: evenly spaced snapshots including endpoints; list: selected times.
SNAPSHOT_COUNT: int | list[float] = [0.0, 1.0, 2.0, 5.0, 10.0]
SEED = 0

BATCH_SIZE = 2**16
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


def train_potential(
    phi_net: nn.Module,
    omega_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    omega_bar: nn.Parameter,
    omega_bar_optimizer: torch.optim.Optimizer,
    target: str,
    lower: float,
    width: float,
    device: torch.device,
    batch_size: int,
    iterations: int,
) -> None:
    for _ in range(iterations):
        points = uniform_points(batch_size, DIMENSION, lower, width, device).requires_grad_()
        with torch.no_grad():
            log_rho = log_target_torch(points, target)
            weights = torch.softmax(log_rho, dim=0)
            omega = omega_net(points)

        phi = phi_net(points)
        velocity = torch.autograd.grad(phi.sum(), points, create_graph=True)[0]
        # Minimize in phi, maximize in omega_bar; enforce E_rho[phi] = 0.
        loss = torch.sum(weights * (0.5 * torch.sum(velocity**2, dim=1) - (omega - omega_bar) * phi))
        optimizer.zero_grad()
        omega_bar_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        omega_bar_optimizer.step()


def train_omega(
    omega_net: nn.Module,
    phi_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    omega_bar: torch.Tensor,
    lower: float,
    width: float,
    device: torch.device,
    batch_size: int,
    iterations: int,
    dt: float,
) -> nn.Module:
    # Freeze omega^n for transport targets and the subsequent particle drift.
    old_omega_net = copy.deepcopy(omega_net).eval()
    old_omega_net.requires_grad_(False)

    for _ in range(iterations):
        points = uniform_points(batch_size, DIMENSION, lower, width, device)
        velocity = scalar_gradient(phi_net, points, device, return_numpy=False)
        with torch.no_grad():
            back_points = torch.remainder(points - dt * velocity - lower, width) + lower
            omega_back = old_omega_net(back_points)
            target = omega_back - dt * (omega_back - omega_bar)

        loss = torch.mean((omega_net(points) - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return old_omega_net


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ten-dimensional mesh-free weighted Laplacian flow experiment.")
    parser.add_argument("--initial", choices=("unimodal",), default="unimodal")
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

    density = partial(
        kde_on_grid, n_grid=N_GRID, dx=dx, cell_volume=cell_volume,
        target=args.target, bandwidth=KDE_BANDWIDTH,
    )
    kl = partial(kl_divergence, target_grid, cell_volume=cell_volume)

    initial_particles = sample_initial(rng, N_PARTICLES, DIMENSION, args.initial, args.target)
    flow_particles = torch.as_tensor(initial_particles, dtype=torch.float32, device=device).clone()
    initial_kl = kl(density(initial_particles[:, :2]))
    flow_kl = [initial_kl]
    flow_times = [0.0]

    snapshot_steps = make_snapshot_steps(SNAPSHOT_COUNT, T_FINAL, DT)
    snapshots = {0: initial_particles[:, :2].copy()} if 0 in snapshot_steps else {}
    target_particles = sample_target_projection(rng, N_PARTICLES, args.target)
    target_kde_kl = kl(density(target_particles))

    phi_net = PeriodicScalarNet(DIMENSION, HIDDEN_DIM, HIDDEN_LAYERS, FOURIER_MODES, width).to(device)
    omega_net = PeriodicScalarNet(DIMENSION, HIDDEN_DIM, HIDDEN_LAYERS, FOURIER_MODES, width).to(device)
    omega_bar = nn.Parameter(torch.zeros((), device=device))
    phi_optimizer = torch.optim.Adam(phi_net.parameters(), lr=1e-3)
    omega_bar_optimizer = torch.optim.Adam([omega_bar], lr=1e-3, maximize=True)
    omega_optimizer = torch.optim.Adam(omega_net.parameters(), lr=1e-3)

    # Initialization: fit omega^0 to log rho - log q_0.
    for _ in range(INITIAL_ITERATIONS):
        points = uniform_points(BATCH_SIZE, DIMENSION, lower, width, device)
        target_omega = log_target_torch(points, args.target) - log_initial_torch(points, args.initial)
        loss = torch.mean((omega_net(points) - target_omega) ** 2)
        omega_optimizer.zero_grad()
        loss.backward()
        omega_optimizer.step()

    print(f"experiment = {tag}, device = {device}, steps = {n_steps}, " f"dt = {DT:g}, particles = {N_PARTICLES}")
    print(f"Target KDE = {target_kde_kl:.4e}")

    for step in range(1, n_steps + 1):
        # ------------------------------------------------------------------
        # 1. Poisson step
        # ------------------------------------------------------------------
        train_potential(
            phi_net, omega_net, phi_optimizer, omega_bar, omega_bar_optimizer, args.target, lower, width, device, BATCH_SIZE, POTENTIAL_ITERATIONS
        )

        # ------------------------------------------------------------------
        # 2. Transport step
        # ------------------------------------------------------------------
        previous_omega_net = train_omega(
            omega_net, phi_net, omega_optimizer, omega_bar.detach(), lower, width, device, BATCH_SIZE, OMEGA_ITERATIONS, DT
        )

        # ------------------------------------------------------------------
        # 3. Particle step
        # ------------------------------------------------------------------
        velocity_particles = scalar_gradient(phi_net, flow_particles, device, return_numpy=False)
        grad_omega_particles = scalar_gradient(previous_omega_net, flow_particles, device, return_numpy=False)
        with torch.no_grad():
            drift_particles = grad_log_target_torch(flow_particles, args.target) - grad_omega_particles
            noise = torch.randn_like(flow_particles)
            flow_particles.add_(DT * velocity_particles + ALPHA * DT * drift_particles + np.sqrt(2.0 * ALPHA * DT) * noise)
            flow_particles.sub_(lower).remainder_(width).add_(lower)

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        projection = flow_particles[:, :2].detach().cpu().numpy()
        flow_kl.append(kl(density(projection)))
        flow_times.append(step * DT)

        if step in snapshot_steps:
            snapshots[step] = projection.copy()
            print(f"Step {step:4d}  flow_t={step * DT:5.2f}  " f"KL(p||q_wlf)={flow_kl[-1]:.4f}")

    kl_path = RESULTS_DIR / f"kl_{tag}.png"
    snapshots_path = RESULTS_DIR / f"snapshots_{tag}.png"
    save_kl_plot(kl_path, flow_times, flow_kl, target_kde_kl, title=r"Convergence of $q_t$ to $p$")
    save_snapshot_plot(snapshots_path, snapshots, DT, mesh_grid, target_grid, args.target, target_particles)

    print(f"Final KL(p||q_wlf) = {flow_kl[-1]}")
    print(f"Saved {kl_path} and {snapshots_path}")


if __name__ == "__main__":
    main()
