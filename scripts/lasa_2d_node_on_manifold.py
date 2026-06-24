#!/usr/bin/env python3
"""
LASA 2D NODE with CLF + On-Manifold Modulation obstacle avoidance.

Hybrid approach:
  1. CLF (inner layer, closed-form) — tracks reference NODE trajectory
  2. On-manifold modulation M(x) (outer layer) — ensures obstacle impenetrability

Based on: Fourie et al., "On-Manifold Strategies for Reactive Dynamical System
Modulation With Nonconvex Obstacles," IEEE TRO 2024.

Run with:
  conda run -n node python scripts/lasa_2d_node_on_manifold.py

Loads Notebooks/LASA_models/Spoon_checkpoint.eqx (pre-trained).
All figures are saved to scripts/figs/ with '_manifold' suffix.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
import numpy as np
from scipy import interpolate

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import diffrax
import optax
import pyLasaDataset as lasa

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
FIGS_DIR    = SCRIPT_DIR / "figs"
MODEL_PATH  = PROJECT_DIR / "Notebooks" / "LASA_models" / "Spoon_checkpoint.eqx"
FIGS_DIR.mkdir(exist_ok=True)

font = {'size': 12}
matplotlib.rc('font', **font)


# ── Model definitions (identical to lasa_2d_node_clf_cbf_obstacles.py) ────────

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


# ── Obstacle geometry helpers ─────────────────────────────────────────────────

def square_xy(xc, yc, a, b, angle, p=6, n=300):
    th = np.linspace(0, 2 * np.pi, n)
    cp_ = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2.0 / p)
    sp_ = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2.0 / p)
    ta, tb = a * cp_, b * sp_
    xs = xc + np.cos(angle) * ta + np.sin(angle) * tb
    ys = yc - np.sin(angle) * ta + np.cos(angle) * tb
    return xs, ys


def ring_sector_xy(xc, yc, ri, ro, n=180,
                   open_start=-np.pi/4, open_end=np.pi/4):
    """Draw a 3/4 ring (C-shape). Opening spans [open_start, open_end] (rad).
    Closed arc goes counterclockwise from open_end to open_start + 2π."""
    th = np.linspace(open_end, open_start + 2 * np.pi, n)
    xo = xc + ro * np.cos(th)
    yo = yc + ro * np.sin(th)
    xi = xc + ri * np.cos(th[::-1])
    yi = yc + ri * np.sin(th[::-1])
    return (np.concatenate([xo, xi, xo[:1]]),
            np.concatenate([yo, yi, yo[:1]]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load LASA Spoon dataset ───────────────────────────────────────────────
    data  = lasa.DataSet.Spoon
    demos = data.demos
    demo_0 = demos[1]
    t = demo_0.t

    ndemos = len(demos)
    T = demos[0].t.shape[-1]
    pos_all, vel_all = [], []
    for i in range(ndemos):
        pos_all.append(demos[i].pos.T)
        vel_all.append(demos[i].vel.T)
    posn = jnp.array(pos_all, dtype=jnp.float32)
    veln = jnp.array(vel_all, dtype=jnp.float32)
    tn   = jnp.array(t.T, dtype=jnp.float32).reshape(T)

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
    traj_proc = jnp.array(traj_proc, dtype=jnp.float32)

    nTD = 4
    posn = traj_proc
    traj_train = traj_proc[1:nTD]

    _, _, data_size = posn.shape
    width_size, depth, seed = 128, 3, 1000
    key = jrandom.PRNGKey(seed)
    _, model_key, _ = jrandom.split(key, 3)
    model_template = NeuralODE(data_size, width_size, depth, key=model_key)

    print(f"Loading model from {MODEL_PATH}")
    model = eqx.tree_deserialise_leaves(str(MODEL_PATH), model_template)

    ts = tn

    # ── Obstacle parameters (same as CLF-CBF script) ──────────────────────────
    train_indx = 5
    p_sq = 6

    center1          = (-27 - 1.25 * np.sqrt(2), -7 - 1.25 * np.sqrt(2))
    semi_major_axis1 = 2.5
    semi_minor_axis1 = 7
    angle1           = -np.pi / 4

    # C-shape obstacle via smooth implicit function:
    #   h_c(x,y) = smax_β(r - r_max, r_min - r, θ₀ - |θ|)
    #   Γ(x) = h_c(x) + 1   (= 1 on boundary, > 1 outside, < 1 inside)
    # smax_β is log-sum-exp smooth max: (1/β) log Σ exp(β·zᵢ)
    # Each term > 0 ↔ in free space (outside outer wall / inside hole / in opening).
    cx_c      = -4.0
    cy_c      = -14.0
    r_min_c   = 3.5
    r_max_c   = 5.5
    theta_0_c = float(0.45 * np.pi)   # opening half-angle (opening faces right)
    beta_c    = 10.0                   # smooth-max sharpness

    ae, be = 8.0, 1.5
    xe     = 1.5                                          # shifted 5 units left
    ye     = -be * np.sqrt(1.0 - (xe / ae) ** 2) - 2.0

    # ── Gamma functions (Γ ≥ 1 outside obstacle, = 1 on boundary) ────────────

    _c1   = jnp.array([center1[0], center1[1]], dtype=jnp.float32)
    _eps  = 1e-9

    def _gamma_sq(x, center, a, b, angle):
        ta = (x[0] - center[0]) * jnp.cos(angle) - (x[1] - center[1]) * jnp.sin(angle)
        tb = (x[0] - center[0]) * jnp.sin(angle) + (x[1] - center[1]) * jnp.cos(angle)
        S = (ta / a) ** p_sq + (tb / b) ** p_sq + _eps
        return S ** (1.0 / p_sq)

    def _gamma_ellipse(x):
        return ((x[0] - xe) / ae) ** 2 + ((x[1] - ye) / be) ** 2

    def _gamma_cshape(x):
        dx    = x[0] - cx_c
        dy    = x[1] - cy_c
        r     = jnp.sqrt(dx * dx + dy * dy + _eps)
        theta = jnp.arctan2(-dy, -dx)  # 180° rotation around center
        t1 = r - r_max_c                    # > 0: outside outer wall
        t2 = r_min_c - r                    # > 0: inside inner hole
        t3 = jnp.abs(theta) - (jnp.pi - theta_0_c)  # > 0: in opening sector (left side)
        # smooth max via log-sum-exp (numerically stable, C¹ everywhere)
        h_c = jax.nn.logsumexp(jnp.array([beta_c * t1, beta_c * t2, beta_c * t3])) / beta_c
        return h_c + 1.0

    # min-Γ over all obstacles — only used for safety check and logging
    def gamma_combined(x):
        g1 = _gamma_sq(x, _c1, semi_major_axis1, semi_minor_axis1, angle1)
        g3 = _gamma_cshape(x)
        g4 = _gamma_ellipse(x)
        return jnp.min(jnp.stack([g1, g3, g4]))

    _gamma_jit = jax.jit(gamma_combined)

    # Per-obstacle value-and-grad JITs (for computing individual M_i)
    _vg_sq1    = jax.jit(jax.value_and_grad(
        lambda x: _gamma_sq(x, _c1, semi_major_axis1, semi_minor_axis1, angle1)))
    _vg_cshape = jax.jit(jax.value_and_grad(_gamma_cshape))
    _vg_ell    = jax.jit(jax.value_and_grad(_gamma_ellipse))

    # Per-obstacle scalar Γ JITs (for sorting)
    _g_sq1    = jax.jit(lambda x: _gamma_sq(x, _c1, semi_major_axis1, semi_minor_axis1, angle1))
    _g_cshape = jax.jit(_gamma_cshape)
    _g_ell    = jax.jit(_gamma_ellipse)

    _ALL_VGS = [_vg_sq1, _vg_cshape, _vg_ell]
    _ALL_GS  = [_g_sq1,  _g_cshape,  _g_ell]

    # ── Per-obstacle modulation matrix M_i(x) ────────────────────────────────
    def compute_M_i(vg_fn, x):
        """Diagonal modulation matrix for a single obstacle's Γ_i.
        Effective boundary at Γ=mod_gamma: λ₁=0 there, λ₁<0 inside."""
        Gamma_i, grad_i = vg_fn(x)
        norm_g = jnp.sqrt(grad_i @ grad_i + 1e-12)
        n_hat  = grad_i / norm_g
        e1     = jnp.array([-n_hat[1], n_hat[0]])
        lam1   = 1.0 - 1.1 / Gamma_i
        lam2   = 1.0 + 1.1 / Gamma_i
        H      = jnp.column_stack([n_hat, e1])
        Lam    = jnp.diag(jnp.array([lam1, lam2]))
        return H @ Lam @ H.T, Gamma_i

    # ── Reference rollout & simulation setup ─────────────────────────────────
    ys    = posn
    f     = lambda t_val, z: model.func(t_val, z, None)
    xref  = model(ts, ys[train_indx, 0, :])  # (T, 2) reference rollout
    x     = jnp.array(posn[train_indx, 0], dtype=jnp.float32)
    xall  = jnp.expand_dims(x, axis=0)
    dti   = ts[1] - ts[0]

    alpha_L    = 4.0
    clf_margin = 0.3
    mod_gamma  = 3.0    # modulation active when Gamma < mod_gamma
    max_steps  = 2000
    reach_tol  = 0.5

    print("\nRunning CLF + On-Manifold Modulation loop...")
    for i in range(max_steps):
        idx    = min(i, len(ts) - 1)
        x_t    = jnp.array(x, dtype=jnp.float32)
        xref_t = jnp.array(xref[idx], dtype=jnp.float32)

        v_nom  = f(ts[idx], x_t)
        v_ref  = f(ts[idx], xref_t)

        # ── Step 1: CLF correction (inner layer) ──────────────────────────────
        Gamma_val = float(_gamma_jit(x_t))
        clf_scale = float(jnp.clip((Gamma_val - 1.0) / clf_margin, 0.0, 1.0))

        G_L   = 2.0 * (x_t - xref_t)
        h_L   = (-2.0 * (x_t - xref_t) @ (v_nom - v_ref)
                 - alpha_L * (x_t - xref_t) @ (x_t - xref_t))
        slack  = G_L @ v_nom - h_L
        G_norm = G_L @ G_L + 1e-12
        u_clf_raw = jnp.where(slack > 0.0, -slack / G_norm * G_L, jnp.zeros(2))
        u_clf  = clf_scale * u_clf_raw
        v_clf  = v_nom + u_clf

        # ── Cascade modulation: M_1 @ M_2 @ M_3 @ M_4 @ v_clf ───────────────
        # Sort obstacles nearest-first (smallest Γ_i first), apply each M_i
        # only when Γ_i < mod_gamma so far-away obstacles don't interfere.
        obs_pairs = sorted(
            zip([float(g(x_t)) for g in _ALL_GS], _ALL_VGS),
            key=lambda p: p[0]
        )
        Gamma = obs_pairs[0][0]  # min Γ for logging
        v_final = v_clf
        for g_i, vg_fn in obs_pairs:
            if g_i < mod_gamma:
                M_i, _ = compute_M_i(vg_fn, x_t)
                v_final = M_i @ v_final

        # ── Euler step ────────────────────────────────────────────────────────
        xnext = x_t + v_final * dti
        xall  = jnp.append(xall, jnp.expand_dims(xnext, axis=0), axis=0)
        x     = xnext

        if i % 10 == 0:
            print(f"  step {i:4d}  x={np.array(x)}  Γ={float(Gamma):.3f}")

        dist_goal = float(jnp.linalg.norm(x - xref[-1]))
        if dist_goal < reach_tol:
            print(f"Reached target at step {i} (dist {dist_goal:.4f}); stopping early")
            break

    # ── Fig 1: path vs target ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(xref[:, 0],  xref[:, 1],  label="Target trajectory", color="green")
    ax.plot(xall[:, 0],  xall[:, 1],  label="Motion plan",       color="red")

    x1sq, y1sq = square_xy(center1[0], center1[1],
                            semi_major_axis1, semi_minor_axis1, angle1)
    _te = np.linspace(0, 2 * np.pi, 200)

    # True h_c = 0 level set for the C-shape (smooth, no straight end-caps).
    _mg   = 1.0
    _xca  = np.linspace(cx_c - r_max_c - _mg, cx_c + r_max_c + _mg, 400)
    _yca  = np.linspace(cy_c - r_max_c - _mg, cy_c + r_max_c + _mg, 400)
    _XC, _YC = np.meshgrid(_xca, _yca)
    _rc   = np.sqrt((_XC - cx_c)**2 + (_YC - cy_c)**2 + 1e-9)
    _thc  = np.arctan2(cy_c - _YC, cx_c - _XC)  # 180° rotation around center
    _stk  = np.array([beta_c * (_rc - r_max_c),
                      beta_c * (r_min_c - _rc),
                      beta_c * (np.abs(_thc) - (np.pi - theta_0_c))])
    _mv   = np.max(_stk, axis=0)
    _HC   = np.log(np.sum(np.exp(_stk - _mv), axis=0)) / beta_c + _mv / beta_c

    ax.plot(x1sq, y1sq, label="Square 1")
    ax.contour(_XC, _YC, _HC, levels=[0], colors=['tab:green'])
    ax.contourf(_XC, _YC, _HC, levels=[-1e6, 0], colors=['tab:green'], alpha=0.15)
    ax.plot(xe + ae * np.cos(_te), ye + be * np.sin(_te), label="Ellipse")
    ax.set_xlabel("X-axis", fontsize=14, labelpad=6)
    ax.set_ylabel("Y-axis", fontsize=14, labelpad=6)
    ax.tick_params(labelsize=12)
    ax.legend(); ax.axis("equal")
    fig.tight_layout(pad=1.5)
    plt.savefig(FIGS_DIR / "path_vs_target_manifold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'path_vs_target_manifold.png'}")

    # ── Fig 2: nominal vs modulated vector field comparison ───────────────────
    xmin, xmax = -52, 10
    ymin, ymax = -30, 8
    indx = train_indx

    f_field = lambda z: model.func(jnp.array(0.0), jnp.array(z, dtype=jnp.float32), None)

    xg, yg = np.meshgrid(np.linspace(xmin, xmax, 55),
                          np.linspace(ymin, ymax, 55))
    xy_vec = np.hstack((xg.reshape(-1, 1), yg.reshape(-1, 1)))
    xy_jax = jnp.array(xy_vec, dtype=jnp.float32)
    uv_nom = np.array(jax.vmap(f_field)(xy_jax))
    u_nom  = uv_nom[:, 0].reshape(xg.shape)
    v_nom_grid = uv_nom[:, 1].reshape(yg.shape)

    def _modulated_field(z):
        # Cascade all M_i in fixed order (sq1→sq2→cshape→ell);
        # vmap cannot sort dynamically, fixed order is still correct per-obstacle.
        v = model.func(jnp.array(0.0), z, None)
        for vg_fn in _ALL_VGS:
            M_i, g_i = compute_M_i(vg_fn, z)
            v = jnp.where(g_i < mod_gamma, M_i @ v, v)
        return v

    uv_mod = np.array(jax.vmap(_modulated_field)(xy_jax))
    u_mod  = uv_mod[:, 0].reshape(xg.shape)
    v_mod  = uv_mod[:, 1].reshape(yg.shape)

    model_y = model(ts, ys[indx, 0])

    # Seed streamlines only from outside all obstacles (Γ ≥ 1).
    # Modulation guarantees the boundary is impenetrable, so exterior seeds
    # stay exterior — no interior streamlines arise naturally.
    gamma_flat = np.array(jax.vmap(_gamma_jit)(xy_jax))
    exterior   = gamma_flat >= 1.0
    _seed_xy   = xy_vec[exterior][::4]  # subsample for reasonable density

    def _fill_obstacles(ax_):
        ax_.contourf(_XC, _YC, _HC, levels=[-1e6, 0], colors=['lightblue'], alpha=1.0)
        ax_.contour(_XC, _YC, _HC, levels=[0], colors=['steelblue'], linewidths=1.5)
        ax_.fill(xe + ae * np.cos(_te), ye + be * np.sin(_te), color="lightblue", alpha=1.0)
        ax_.fill(x1sq, y1sq, color="lightblue", alpha=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(30, 14))

    ax0 = axes[0]
    ax0.streamplot(xg, yg, u_nom, v_nom_grid, arrowsize=3, color="plum",
                   start_points=_seed_xy, integration_direction="both", maxlength=100)
    _fill_obstacles(ax0)
    ax0.plot(posn[indx, :, 0], posn[indx, :, 1],
             c="black", linestyle="--", linewidth=6, label="Demonstration")
    ax0.plot(model_y[:, 0], model_y[:, 1], c="green", linewidth=8, label="Target trajectory")
    ax0.plot(model_y[0, 0],  model_y[0, 1],  marker="o", markersize=24, c="saddlebrown")
    ax0.plot(model_y[-1, 0], model_y[-1, 1], marker="o", markersize=24, c="darkblue")
    ax0.set_title("Nominal NODE vector field $f(x)$", fontsize=16)
    ax0.set_xlabel(r"$x_1$"); ax0.set_ylabel(r"$x_2$")
    ax0.set_xlim([xmin, xmax]); ax0.set_ylim([ymin, ymax])
    ax0.legend(fontsize=12)

    ax1 = axes[1]
    ax1.streamplot(xg, yg, u_mod, v_mod, arrowsize=3, color="plum",
                   start_points=_seed_xy, integration_direction="forward", maxlength=100)
    _fill_obstacles(ax1)
    ax1.plot(posn[indx, :, 0], posn[indx, :, 1],
             c="black", linestyle="--", linewidth=6, label="Demonstration")
    ax1.plot(model_y[:, 0], model_y[:, 1], c="green", linewidth=8, label="Target trajectory")
    ax1.plot(xall[:, 0], xall[:, 1], c="red", linewidth=8, label="Motion plan (CLF+M)")
    ax1.plot(model_y[0, 0],  model_y[0, 1],  marker="o", markersize=24, c="saddlebrown")
    ax1.plot(model_y[-1, 0], model_y[-1, 1], marker="o", markersize=24, c="darkblue")
    ax1.set_title(r"Modulated vector field $M(x)f(x)$", fontsize=16)
    ax1.set_xlabel(r"$x_1$"); ax1.set_ylabel(r"$x_2$")
    ax1.set_xlim([xmin, xmax]); ax1.set_ylim([ymin, ymax])
    ax1.legend(fontsize=12)

    fig.tight_layout()
    plt.savefig(FIGS_DIR / "vector_field_comparison_manifold.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'vector_field_comparison_manifold.png'}")

    # ── Fig 3: nominal vector field + motion plan ─────────────────────────────
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 14))
    ax.streamplot(xg, yg, u_nom, v_nom_grid, arrowsize=3, color="plum",
                  start_points=_seed_xy, integration_direction="both", maxlength=100)
    _fill_obstacles(ax)
    ax.plot(posn[indx, :, 0], posn[indx, :, 1],
            c="black", linestyle="--", linewidth=8, label="Demonstration")
    ax.plot(model_y[:, 0], model_y[:, 1],
            c="green", linewidth=12, label="Target trajectory")
    ax.plot(xall[:, 0], xall[:, 1],
            c="red", linewidth=8, label="Motion plan (CLF+M)")
    ax.plot(model_y[0, 0],  model_y[0, 1],  marker="o", markersize=36, c="saddlebrown")
    ax.plot(model_y[-1, 0], model_y[-1, 1], marker="o", markersize=36, c="darkblue")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_xlim([float(xmin), float(xmax)]); ax.set_ylim([float(ymin), float(ymax)])
    plt.savefig(FIGS_DIR / "vector_field_manifold.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'vector_field_manifold.png'}")

    # ── Fig 4: vector field colored by rollout step ───────────────────────────
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 14))
    ax.streamplot(xg, yg, u_nom, v_nom_grid, arrowsize=3, color="plum",
                  start_points=_seed_xy, integration_direction="both", maxlength=100)
    _fill_obstacles(ax)
    ax.plot(posn[indx, :, 0], posn[indx, :, 1],
            c="black", linestyle="--", linewidth=8, label="Demonstration")
    ax.plot(model_y[:, 0], model_y[:, 1],
            c="green", linewidth=12, label="Target trajectory")

    _path  = np.asarray(xall)
    _steps = np.arange(len(_path))
    _pts   = _path.reshape(-1, 1, 2)
    _segs  = np.concatenate([_pts[:-1], _pts[1:]], axis=1)
    _lc    = LineCollection(_segs, cmap="plasma_r", linewidth=8, label="Motion plan")
    _lc.set_array(_steps[:-1])
    ax.add_collection(_lc)
    cbar = fig.colorbar(_lc, ax=ax)
    cbar.set_label("Rollout step (darker = later)")

    ax.plot(model_y[0, 0],  model_y[0, 1],  marker="o", markersize=36, c="saddlebrown")
    ax.plot(model_y[-1, 0], model_y[-1, 1], marker="o", markersize=36, c="darkblue")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_xlim([float(xmin), float(xmax)]); ax.set_ylim([float(ymin), float(ymax)])
    plt.savefig(FIGS_DIR / "vector_field_manifold_colored.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'vector_field_manifold_colored.png'}")

    # ── Safety check: verify Γ_combined ≥ 1 for all trajectory points ────────
    gammas = [float(_gamma_jit(jnp.array(pt, dtype=jnp.float32)))
              for pt in np.asarray(xall)]
    min_gamma = min(gammas)
    print(f"\nSafety check — min Γ(= min individual Γᵢ) along trajectory: {min_gamma:.4f}")
    if min_gamma < 1.0:
        print(f"  WARNING: obstacle penetration detected (Γ < 1 at {sum(g < 1 for g in gammas)} steps)")
    else:
        print("  OK: no obstacle penetration (Γ ≥ 1 everywhere)")

    print(f"\nDone. All figures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
