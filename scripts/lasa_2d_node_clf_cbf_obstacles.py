#!/usr/bin/env python3
"""
LASA 2D NODE with CLF-CBF obstacle avoidance.
Converted from Notebooks/LASA_2D_NODE_CLF_CBF_obstacles.ipynb

Run with:
  conda run -n node python scripts/lasa_2d_node_clf_cbf_obstacles.py

Loads Notebooks/LASA_models/Spoon_checkpoint.eqx (pre-trained).
All figures are saved to scripts/figs/.
"""

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
import numpy as np
from scipy import interpolate
import cvxpy as cp

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.special import logsumexp as _logsumexp
import equinox as eqx
import diffrax
import optax
import pyLasaDataset as lasa

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent
FIGS_DIR     = SCRIPT_DIR / "figs"
MODEL_PATH   = PROJECT_DIR / "Notebooks" / "LASA_models" / "Spoon_checkpoint.eqx"
FIGS_DIR.mkdir(exist_ok=True)

font = {'size': 12}
matplotlib.rc('font', **font)


# ── Model definitions ─────────────────────────────────────────────────────────

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initializer = jnn.initializers.orthogonal()
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        key_weights = jrandom.split(key, depth + 1)
        for i in range(depth + 1):
            where = lambda m, i=i: m.layers[i].weight
            shape = self.mlp.layers[i].weight.shape
            self.mlp = eqx.tree_at(
                where, self.mlp,
                replace=initializer(key_weights[i], shape, dtype=jnp.float32),
            )

    @eqx.filter_jit
    def __call__(self, t, y, args):
        return self.mlp(y)


class Funcd(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initializer = jnn.initializers.orthogonal()
        self.mlp = eqx.nn.MLP(
            in_size=2 * data_size,
            out_size=2 * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        key_weights = jrandom.split(key, depth + 1)
        for i in range(depth + 1):
            where = lambda m, i=i: m.layers[i].weight
            shape = self.mlp.layers[i].weight.shape
            self.mlp = eqx.tree_at(
                where, self.mlp,
                replace=initializer(key_weights[i], shape, dtype=jnp.float32),
            )

    @eqx.filter_jit
    def __call__(self, t, yd, args):
        return self.mlp(yd)


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


class NeuralODEd(eqx.Module):
    func: Funcd

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Funcd(data_size, width_size, depth, key=key)

    @eqx.filter_jit
    def __call__(self, ts, yd0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=yd0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


# ── Dataloader ────────────────────────────────────────────────────────────────

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


# ── Obstacle geometry helpers ─────────────────────────────────────────────────

def square_xy(xc, yc, a, b, angle, p=6, n=300):
    """Boundary points of a smooth square (superellipse |·|^p + |·|^p = 1)."""
    th = np.linspace(0, 2 * np.pi, n)
    cp_ = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2.0 / p)
    sp_ = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2.0 / p)
    ta, tb = a * cp_, b * sp_
    xs = xc + np.cos(angle) * ta + np.sin(angle) * tb
    ys = yc - np.sin(angle) * ta + np.cos(angle) * tb
    return xs, ys


def ring_sector_xy(xc, yc, ri, ro, n=180):
    """Boundary polygon for a full ring (annulus)."""
    th = np.linspace(0, 2 * np.pi, n)
    xo = xc + ro * np.cos(th);  yo = yc + ro * np.sin(th)
    xi = xc + ri * np.cos(th[::-1]); yi = yc + ri * np.sin(th[::-1])
    return (np.concatenate([xo, xi, xo[:1]]),
            np.concatenate([yo, yi, yo[:1]]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load LASA Spoon dataset ───────────────────────────────────────────────
    data  = lasa.DataSet.Spoon
    demos = data.demos
    demo_0 = demos[1]
    t = demo_0.t          # shape (1, T)

    lasa.utilities.plot_model(data)
    plt.savefig(FIGS_DIR / "lasa_spoon_dataset.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: {FIGS_DIR / 'lasa_spoon_dataset.png'}")

    # ── Build position / velocity arrays ─────────────────────────────────────
    ndemos = len(demos)
    T = demos[0].t.shape[-1]
    pos_all, vel_all = [], []
    for i in range(ndemos):
        pos_all.append(demos[i].pos.T)
        vel_all.append(demos[i].vel.T)
    posn = jnp.array(pos_all)        # (ndemos, T, 2)
    veln = jnp.array(vel_all)        # (ndemos, T, 2)
    tn   = jnp.array(t.T).reshape(T) # (T,) original time axis

    # ── Resample to uniform grid ──────────────────────────────────────────────
    nsamples = 1000
    ts_norm  = t[0] / t[0, -1]
    ts_new   = jnp.linspace(0, 1, nsamples)
    dim = posn.shape[2]

    traj_proc = np.zeros((ndemos, nsamples, dim))
    vel_proc  = np.zeros((ndemos, nsamples, dim))
    for i in range(ndemos):
        for j in range(dim):
            traj_proc[i, :, j] = interpolate.interp1d(
                ts_norm, np.array(posn[i, :, j]))(ts_new)
            vel_proc[i, :, j]  = interpolate.interp1d(
                ts_norm, np.array(veln[i, :, j]))(ts_new)
    traj_proc = jnp.array(traj_proc)

    nTD = 4
    traj_train = traj_proc[1:nTD]

    # ── Model setup ───────────────────────────────────────────────────────────
    _, _, data_size = posn.shape
    width_size, depth, seed = 128, 3, 1000
    key = jrandom.PRNGKey(seed)
    _, model_key, _ = jrandom.split(key, 3)
    model_template = NeuralODE(data_size, width_size, depth, key=model_key)

    print(f"Loading model from {MODEL_PATH}")
    model = eqx.tree_deserialise_leaves(str(MODEL_PATH), model_template)

    ts = tn   # use original time axis for rollout (matches notebook)

    # ── Obstacle parameters ───────────────────────────────────────────────────
    train_indx = 5
    p_sq = 6   # superellipse exponent

    # Square obstacles
    center1          = (-27, -7)
    semi_major_axis1 = 5
    semi_minor_axis1 = 7
    angle1           = -np.pi / 4

    center2          = (-25, -14)
    semi_major_axis2 = 12
    semi_minor_axis2 = 3
    angle2           = np.pi / 4

    # Full ring (annulus)
    c      = jnp.array([-4.0, -14.0])
    r_in   = 3.5
    r_out  = 5.5
    k_ring = 1.5   # soft-min sharpness

    # Ellipse obstacle: thin & long, below target (0,0)
    ae, be = 8.0, 1.5
    xe     = 6.5
    ye     = -be * np.sqrt(1.0 - (6.5 / ae) ** 2) - 2.0

    # ── Fig 1: dataset overview + obstacle shapes ─────────────────────────────
    model_y_prev = model(ts, posn[train_indx, 0])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(posn[train_indx, :, 0], posn[train_indx, :, 1],
            c="dodgerblue", label="Real")
    ax.plot(posn[train_indx, 0, 0],  posn[train_indx, 0, 1],
            c="saddlebrown", marker='o', markersize=12, label="Start")
    ax.plot(posn[train_indx, -1, 0], posn[train_indx, -1, 1],
            c="black", marker='x', markersize=12, label="Target")
    ax.plot(model_y_prev[:, 0], model_y_prev[:, 1], c="crimson", label="Model")

    x1sq, y1sq = square_xy(center1[0], center1[1],
                            semi_major_axis1, semi_minor_axis1, angle1, p=p_sq)
    x2sq, y2sq = square_xy(center2[0], center2[1],
                            semi_major_axis2, semi_minor_axis2, angle2, p=p_sq)
    xr, yr = ring_sector_xy(float(np.asarray(c)[0]), float(np.asarray(c)[1]),
                              r_in, r_out)
    _te = np.linspace(0, 2 * np.pi, 200)

    ax.plot(x1sq, y1sq, label="Square 1")
    ax.plot(x2sq, y2sq, label="Square 2")
    ax.plot(xr, yr,     label="Ring")
    ax.plot(xe + ae * np.cos(_te), ye + be * np.sin(_te), label="Ellipse")
    ax.set_title("Obstacles and predicted trajectory")
    ax.set_xlabel("X-axis"); ax.set_ylabel("Y-axis")
    ax.grid(True); ax.axis("equal"); ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "obstacles_preview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'obstacles_preview.png'}")

    # ── JAX ring barrier (smooth soft-min via logsumexp) ─────────────────────
    _cc = jnp.asarray(c, dtype=float)

    def _ring_barrier(xy):
        dvec  = xy - _cc
        dd    = jnp.sqrt(dvec @ dvec + 1e-9)
        g_out = r_out - dd
        g_in  = dd - r_in
        gs    = jnp.stack([g_out, g_in])
        return (1.0 / k_ring) * _logsumexp(-k_ring * gs)

    _ring_vg = jax.jit(jax.value_and_grad(_ring_barrier))

    # ── CLF-CBF QP loop ───────────────────────────────────────────────────────
    ys = posn
    model_load = model
    f = lambda t_val, z: model_load.func(t_val, z, None)

    xref = model_load(ts, ys[train_indx, 0, :])   # (T, 2) reference rollout
    x    = posn[train_indx, 0]
    xall = jnp.expand_dims(x, axis=0)
    dti  = ts[1] - ts[0]

    # QP obstacle params
    xc1, yc1  = center1
    xc2, yc2  = center2
    a1, b1    = semi_major_axis1, semi_minor_axis1
    a2, b2    = semi_major_axis2, semi_minor_axis2

    alpha_L   = 4.0
    alpha_B   = 2.3
    alpha_B_c = 10.0
    alpha_B_e = 2.3
    lambda_v  = 0.1
    eps_sq    = 1e-9

    max_steps = 5000
    reach_tol = 0.5

    print("\nRunning CLF-CBF QP loop...")
    for i in range(max_steps):
        idx   = min(i, len(ts) - 1)
        x_t   = np.asarray(x)
        xref_t = np.asarray(xref[idx, :])

        fx_t    = np.asarray(f(ts[idx], jnp.array(x_t)))
        fxref_t = np.asarray(f(ts[idx], jnp.array(xref_t)))

        # CLF
        G_L = 2.0 * (x_t - xref_t)
        h_L = (-2.0 * (x_t - xref_t) @ (fx_t - fxref_t)
               - alpha_L * (x_t - xref_t) @ (x_t - xref_t))

        # Square product CBF (superellipse, p-norm form)
        pos_x, pos_y = float(x[0]), float(x[1])
        ta1 = (pos_x - xc1) * np.cos(angle1) - (pos_y - yc1) * np.sin(angle1)
        tb1 = (pos_x - xc1) * np.sin(angle1) + (pos_y - yc1) * np.cos(angle1)
        ta2 = (pos_x - xc2) * np.cos(angle2) - (pos_y - yc2) * np.sin(angle2)
        tb2 = (pos_x - xc2) * np.sin(angle2) + (pos_y - yc2) * np.cos(angle2)

        S1 = (ta1 / a1) ** p_sq + (tb1 / b1) ** p_sq + eps_sq
        S2 = (ta2 / a2) ** p_sq + (tb2 / b2) ** p_sq + eps_sq
        sq1 = S1 ** (1.0 / p_sq) - 1
        sq2 = S2 ** (1.0 / p_sq) - 1

        g1x = S1 ** (1.0/p_sq - 1) * ((ta1/a1)**(p_sq-1)*np.cos(angle1)/a1
                                        + (tb1/b1)**(p_sq-1)*np.sin(angle1)/b1)
        g1y = S1 ** (1.0/p_sq - 1) * (-(ta1/a1)**(p_sq-1)*np.sin(angle1)/a1
                                        + (tb1/b1)**(p_sq-1)*np.cos(angle1)/b1)
        g2x = S2 ** (1.0/p_sq - 1) * ((ta2/a2)**(p_sq-1)*np.cos(angle2)/a2
                                        + (tb2/b2)**(p_sq-1)*np.sin(angle2)/b2)
        g2y = S2 ** (1.0/p_sq - 1) * (-(ta2/a2)**(p_sq-1)*np.sin(angle2)/a2
                                        + (tb2/b2)**(p_sq-1)*np.cos(angle2)/b2)

        B      = sq1 * sq2
        grad_B = jnp.array([[g1x * sq2 + sq1 * g2x,
                              g1y * sq2 + sq1 * g2y]])

        # Ring CBF via JAX autodiff
        _Bv, _gv = _ring_vg(jnp.asarray(x_t, dtype=float))
        B_c      = np.asarray(_Bv)
        grad_B_c = np.asarray(_gv)

        # Ellipse CBF (quadratic)
        B_e      = ((x_t[0] - xe) / ae) ** 2 + ((x_t[1] - ye) / be) ** 2 - 1.0
        grad_B_e = np.array([[2 * (x_t[0] - xe) / ae ** 2,
                               2 * (x_t[1] - ye) / be ** 2]])

        # Solve QP
        Q       = np.eye(x_t.shape[0])
        vopt    = cp.Variable(x_t.shape[0])
        epsilon = cp.Variable((1, 1))
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(vopt, Q)
                        + lambda_v * cp.quad_form(epsilon, np.eye(1))),
            [G_L @ vopt - epsilon <= h_L,
             grad_B   @ (fx_t + vopt) >= -alpha_B   * B,
             grad_B_c @ (fx_t + vopt) >= -alpha_B_c * B_c,
             grad_B_e @ (fx_t + vopt) >= -alpha_B_e * B_e],
        )
        prob.solve(verbose=False)

        v_val = vopt.value if vopt.value is not None else np.zeros(x_t.shape[0])

        # Euler step
        f1_x  = jnp.array(fx_t + v_val)
        xnext = f1_x * dti + x
        xall  = jnp.append(xall, jnp.expand_dims(xnext, axis=0), axis=0)
        x     = xnext

        if i % 10 == 0:
            print(f"  step {i:4d}  x={np.array(x)}")

        dist_goal = float(np.linalg.norm(np.asarray(x) - np.asarray(xref[-1])))
        if dist_goal < reach_tol:
            print(f"Reached target at step {i} (dist {dist_goal:.4f}); stopping early")
            break

    # ── Fig 2: path vs target (simple) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(xref[:, 0],  xref[:, 1],  label="Target trajectory", color="green")
    ax.plot(xall[:, 0],  xall[:, 1],  label="Motion plan",       color="red")

    x1sq, y1sq = square_xy(center1[0], center1[1],
                            semi_major_axis1, semi_minor_axis1, angle1)
    x2sq, y2sq = square_xy(center2[0], center2[1],
                            semi_major_axis2, semi_minor_axis2, angle2)
    xr, yr = ring_sector_xy(float(np.asarray(c)[0]), float(np.asarray(c)[1]),
                              r_in, r_out)
    _te = np.linspace(0, 2 * np.pi, 200)

    ax.plot(x1sq, y1sq, label="Square 1")
    ax.plot(x2sq, y2sq, label="Square 2")
    ax.plot(xr, yr,     label="Ring")
    ax.plot(xe + ae * np.cos(_te), ye + be * np.sin(_te), label="Ellipse")
    ax.set_xlabel("X-axis", fontsize=14, labelpad=6)
    ax.set_ylabel("Y-axis", fontsize=14, labelpad=6)
    ax.tick_params(labelsize=12)
    ax.legend(); ax.axis("equal")
    fig.tight_layout(pad=1.5)
    plt.savefig(FIGS_DIR / "path_vs_target.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'path_vs_target.png'}")

    # ── Fig 3: vector field + motion plan (publication style) ─────────────────
    xmin, xmax = -52, 10
    ymin, ymax = -30, 8
    indx = train_indx

    f_field = lambda z: model.func(jnp.array(0.0), jnp.array(z), None)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 14))
    xg, yg = np.meshgrid(np.linspace(xmin, xmax, 50),
                          np.linspace(ymin, ymax, 50))
    xy_vec = np.hstack((xg.reshape(-1, 1), yg.reshape(-1, 1)))
    uv_vec = np.array(jax.vmap(f_field)(xy_vec))
    u = uv_vec[:, 0].reshape(xg.shape)
    v = uv_vec[:, 1].reshape(yg.shape)
    sp = ax.streamplot(xg, yg, u, v, arrowsize=3, density=1.4, color="plum")

    model_y = model(ts, ys[indx, 0])
    ax.plot(posn[indx, :, 0], posn[indx, :, 1],
            c="black", linestyle="--", linewidth=8, label="Demonstration")
    ax.plot(model_y[:, 0], model_y[:, 1],
            c="green", linewidth=12, label="Target trajectory")
    ax.plot(xall[:, 0], xall[:, 1],
            c="red", linewidth=8, label="Motion plan")
    ax.plot(model_y[0, 0],  model_y[0, 1],  marker="o", markersize=36, c="saddlebrown")
    ax.plot(model_y[-1, 0], model_y[-1, 1], marker="o", markersize=36, c="darkblue")

    # Draw and fill obstacles
    x1sq, y1sq = square_xy(center1[0], center1[1],
                            semi_major_axis1, semi_minor_axis1, angle1)
    x2sq, y2sq = square_xy(center2[0], center2[1],
                            semi_major_axis2, semi_minor_axis2, angle2)
    _cx, _cy = float(np.asarray(c)[0]), float(np.asarray(c)[1])
    _th = np.linspace(0, 2 * np.pi, 180)
    _xo = _cx + r_out * np.cos(_th); _yo = _cy + r_out * np.sin(_th)
    _xi = _cx + r_in  * np.cos(_th[::-1]); _yi = _cy + r_in  * np.sin(_th[::-1])
    _te = np.linspace(0, 2 * np.pi, 200)

    ax.fill(np.concatenate([_xo, _xi]), np.concatenate([_yo, _yi]),
            color="lightblue", alpha=1.0)
    ax.fill(xe + ae * np.cos(_te), ye + be * np.sin(_te),
            color="lightblue", alpha=1.0)
    ax.fill(x1sq, y1sq, color="lightblue", alpha=1.0)
    ax.fill(x2sq, y2sq, color="lightblue", alpha=1.0)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim([float(xmin), float(xmax)])
    ax.set_ylim([float(ymin), float(ymax)])
    plt.savefig(FIGS_DIR / "vector_field_clf_cbf.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'vector_field_clf_cbf.png'}")

    # ── Fig 4: vector field + motion plan colored by rollout step ─────────────
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 14))
    sp = ax.streamplot(xg, yg, u, v, arrowsize=3, density=1.4, color="plum")

    ax.plot(posn[indx, :, 0], posn[indx, :, 1],
            c="black", linestyle="--", linewidth=8, label="Demonstration")
    ax.plot(model_y[:, 0], model_y[:, 1],
            c="green", linewidth=12, label="Target trajectory")

    _path  = np.asarray(xall)
    _steps = np.arange(len(_path))
    _pts   = _path.reshape(-1, 1, 2)
    _segs  = np.concatenate([_pts[:-1], _pts[1:]], axis=1)
    _lc    = LineCollection(_segs, cmap="plasma_r", linewidth=8,
                             label="Motion plan", zorder=3)
    _lc.set_array(_steps[:-1])
    ax.add_collection(_lc)
    cbar = fig.colorbar(_lc, ax=ax)
    cbar.set_label("Rollout step (darker = later)")

    ax.plot(model_y[0, 0],  model_y[0, 1],  marker="o", markersize=36, c="saddlebrown")
    ax.plot(model_y[-1, 0], model_y[-1, 1], marker="o", markersize=36, c="darkblue")

    ax.fill(np.concatenate([_xo, _xi]), np.concatenate([_yo, _yi]),
            color="lightblue", alpha=1.0)
    ax.fill(xe + ae * np.cos(_te), ye + be * np.sin(_te),
            color="lightblue", alpha=1.0)
    ax.fill(x1sq, y1sq, color="lightblue", alpha=1.0)
    ax.fill(x2sq, y2sq, color="lightblue", alpha=1.0)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim([float(xmin), float(xmax)])
    ax.set_ylim([float(ymin), float(ymax)])
    plt.savefig(FIGS_DIR / "vector_field_clf_cbf_colored.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'vector_field_clf_cbf_colored.png'}")

    print(f"\nDone. All figures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
