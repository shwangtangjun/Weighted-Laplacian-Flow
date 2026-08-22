# Weighted-Laplacian-Flow
This repository contains the Python implementation of weighted Laplacian flow. It provides a grid-based method for two-dimensional experiments and a mesh-free neural-network method for ten-dimensional experiments.

## Dependencies
NumPy, SciPy, Matplotlib, and PyTorch (only for mesh-free method). The code is tested on the following versions:
```
python 3.13.13
numpy 2.4.4
scipy 1.17.1
matplotlib 3.10.8
torch 2.11.0+cu128
```

## Files
- [wlf_grid_based.py](wlf_grid_based.py) implements the grid-based method in two dimensions using finite differences, sparse factorization, and semi-Lagrangian transport.
- [wlf_mesh_free.py](wlf_mesh_free.py) implements the mesh-free method in ten dimensions using periodic neural networks, the Deep Ritz objective, and semi-Lagrangian regression.
- [utils.py](utils.py) contains shared code for:
  - initial and target distributions
  - particle sampling and boundary conditions
  - kernel density estimation and KL divergence
  - plotting KL histories and particle snapshots

All numerical and training parameters are fixed global variables near the top of the two main scripts. Edit these variables directly to change the number of particles, time step, grid resolution, network size, or training iterations.

## Usage
Run the two-dimensional grid-based experiments with:
```
python wlf_grid_based.py --initial unimodal --target bimodal
python wlf_grid_based.py --initial uniform --target bimodal
python wlf_grid_based.py --initial uniform --target cauchy
python wlf_grid_based.py --initial corner --target cauchy
```

Run the ten-dimensional mesh-free experiments with:
```
python wlf_mesh_free.py --initial unimodal --target bimodal
python wlf_mesh_free.py --initial uniform --target bimodal
```
