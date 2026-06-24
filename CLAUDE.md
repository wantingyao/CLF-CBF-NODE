# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

All responses must be in **Chinese or English only**. Do not use Japanese or Korean under any circumstances.

## Project Overview

Research codebase for ICRA 2024 paper: "Learning Complex Motion Plans using Neural ODEs with Safety and Stability Guarantees." The system learns robot motion from demonstrations using Neural ODEs (NODEs), then generates safe, reactive plans online at 1 KHz using a CLF-CBF Quadratic Program (QP).

## Environment

Local conda environment: **`node`**. Always run Python scripts with this environment:

```bash
conda run -n node python <script.py>
# or activate first:
conda activate node
```

## Setup

```bash
conda run -n node pip install -r requirements.txt
```

Dependencies: `jax`, `equinox`, `diffrax`, `optax`, `jaxopt`, `cvxpy`, `scipy`, `bagpy`

For robot deployment, also requires ROS Noetic, `libfranka`, `franka_ros`, and the `franka_interactive_controllers` catkin package (external repo). Trained NODE models (`.eqx`) and reference trajectories (`.npy`) live in `franka_interactive_controllers/config/`.

## Architecture

### Training (Notebooks/)

Jupyter notebooks handle NODE training and simulation:
- `LASA_2D_NODE.ipynb` — train NODE on 2D drawing dataset
- `LASA_2D_NODE_CLF_CBF_obstacles.ipynb` — 2D NODE + CLF-CBF QP with obstacles
- `LASA_2D_NODE_CLF_disturbance.ipynb` — 2D NODE + CLF disturbance rejection
- `Franka_demos_3D_NODE.ipynb` — 3D position-only NODE for Franka
- `Franka_demos_Full_pose_NODE_CLF.ipynb` — full pose (position + SO(3)) NODE
- `clfd_data_Full_pose_NODE_CLF.ipynb` — full pose NODE on clfd dataset
- `GMR_GP_training.ipynb`, `ImFlow_training.ipynb` — baseline comparisons

### ROS Nodes (Gazebo_scripts/ and Lab_PC_scripts/)

Two parallel script sets — `Gazebo_scripts/` for Gazebo simulation, `Lab_PC_scripts/` for real robot. Both contain identical file names; deploy by copying to `franka_interactive_controllers/scripts/`.

**NODE model servers** (`NODE_model_vel_*.py`) — ROS nodes that run the trained NODE forward in real time and publish velocity predictions:
- `NODE_model_vel_jit.py` — 3D position NODE; defines `Func` (nominal) and `Funcd` (disturbed) as `eqx.Module` MLP vector fields solved via `diffrax`
- `NODE_model_vel_SO3.py` — full pose NODE; adds `Func_rot` operating on SO(3) with quaternion arithmetic

**CLF-CBF QP controllers** (`cmd_vel_ustar_split_OSQP_*.py`) — ROS nodes implementing the `ustar_m` class that subscribes to Franka state + NODE velocities, solves a JAX-based OSQP QP, and publishes desired twist to the impedance controller:
- `cmd_vel_ustar_split_OSQP_jit.py` — 3D CLF-NODE (disturbance rejection)
- `cmd_vel_ustar_split_OSQP_jit_SO3.py` — full pose CLF-NODE
- `cmd_vel_ustar_split_OSQP_obstacle.py` — 3D CLF-CBF-NODE (adds obstacle avoidance via `handPos` subscriber)

**Visualization** (`my_visuals.py`) — publishes RViz markers for reference trajectory and robot path. `Visualizer_sphere.py` (Gazebo only) renders obstacle spheres.

### Key design patterns

- NODE vector fields are `eqx.Module` subclasses with orthogonal weight initialization; call `eqx.tree_at` to patch weights after construction.
- QP controllers call `jax.config.update("jax_enable_x64", True)` at module top — required for numerical precision in the CLF-CBF formulation.
- The `ustar_m` controller tracks a look-ahead window on the reference trajectory (`ref_range_start`, `ref_range`) to compute the virtual CLF input.
- Models are loaded from `.eqx` files and trajectories from `.npy` files; paths are hardcoded in each script and must be updated per task.

## Running the Experiment (ROS)

```bash
# Terminal 1: launch robot/simulator + impedance controller
roslaunch franka_interactive_controllers <launch-file> controller:=passiveDS_impedance

# Terminal 2: start NODE model server
roslaunch franka_interactive_controllers NODE_model.launch

# Terminal 3: start CLF-CBF QP controller
rosrun franka_interactive_controllers cmd_vel_ustar_split_OSQP_<variant>.py
```

Where `<launch-file>` is `simulate_panda_gazebo.launch` (Gazebo) or `franka_interactive_bringup.launch` (real robot), and `<variant>` is `jit`, `jit_SO3`, or `obstacle`.

## Standalone Scripts (scripts/)

Python scripts converted from notebooks, runnable without Jupyter:

- `lasa_2d_node_clf_cbf_obstacles.py` — 2D NODE + CLF-CBF QP with square superellipse, ring (annulus), and ellipse obstacles on the LASA Spoon dataset. Loads `Notebooks/LASA_models/Spoon_checkpoint.eqx`. Run with:
  ```bash
  conda run -n node python scripts/lasa_2d_node_clf_cbf_obstacles.py
  ```

All generated figures are saved to `scripts/figs/`.

## Datasets

- `Dataset/2D_drawing/` — hand-drawn 2D trajectories (`.txt`)
- `Dataset/IROS_dataset/` — 2D letter-shape trajectories from iflow (`.npy`)
- `Dataset/Franka_demos_Full_pose/` — full pose Franka demonstrations (`.npy`)
- `Dataset/clfd_data/` — full pose demonstrations from the clfd benchmark (`.npy`)
