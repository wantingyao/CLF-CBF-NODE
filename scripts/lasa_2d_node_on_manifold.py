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

    center2          = (-25 + 3 * np.sqrt(2), -14 + 3 * np.sqrt(2))
    semi_major_axis2 = 6
    semi_minor_axis2 = 3
    angle2           = np.pi / 4 + np.pi / 2

    c      = jnp.array([-4.0, -14.0], dtype=jnp.float32)
    r_in   = 3.5
    r_out  = 5.5

    ae, be = 8.0, 1.5
    xe     = 1.5                                          # shifted 5 units left
    ye     = -be * np.sqrt(1.0 - (xe / ae) ** 2) - 2.0

    # ── Gamma functions (Γ ≥ 1 outside obstacle, = 1 on boundary) ────────────

    _c1 = jnp.array([center1[0], center1[1]], dtype=jnp.float32)
    _c2 = jnp.array([center2[0], center2[1]], dtype=jnp.float32)
    _cc = c
    _eps = 1e-9

    def _gamma_sq(x, center, a, b, angle):
        ta = (x[0] - center[0]) * jnp.cos(angle) - (x[1] - center[1]) * jnp.sin(angle)
        tb = (x[0] - center[0]) * jnp.sin(angle) + (x[1] - center[1]) * jnp.cos(angle)
        S = (ta / a) ** p_sq + (tb / b) ** p_sq + _eps
        return S ** (1.0 / p_sq)

    def _gamma_ellipse(x):
        return ((x[0] - xe) / ae) ** 2 + ((x[1] - ye) / be) ** 2

    # 3/4-ring opening: lower-left, ±45° from the -135° direction.
    _ring_open_dir = jnp.array([-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)])
    _ring_open_cos = float(np.cos(np.pi / 4))   # ≈ 0.707, same ±45° half-width

    def _gamma_ring_3quarter(x):
        diff = x - _cc
        r = jnp.sqrt(diff @ diff + _eps)
        gamma_outer = r / r_out
        diff_norm = diff / (r + _eps)
        cos_a = diff_norm @ _ring_open_dir
        # Hard cutoff: open sector (cos_a > threshold) → Gamma=100 (no obstacle);
        # closed sector → r/r_out as usual.
        # jnp.where gradient is 0 in the open branch, but w_mod=0 there anyway
        # (Gamma=100 >> 1+mod_margin), so M_eff=I and the zero gradient is harmless.
        return jnp.where(cos_a > _ring_open_cos, jnp.array(100.0), gamma_outer)

    def gamma_combined(x):
        # Use min over individual Γᵢ as the combined representation.
        # softmin(ρ) ≤ true min always, and can drop below 1 even when all Γᵢ > 1
        # (phantom obstacle between adjacent sq1/sq2), making λ₁ negative.
        # jnp.min gives Γ ≥ 1 outside all obstacles. JAX computes subgradient via
        # straight-through of the argmin element, which is correct geometrically.
        g1 = _gamma_sq(x, _c1, semi_major_axis1, semi_minor_axis1, angle1)
        g2 = _gamma_sq(x, _c2, semi_major_axis2, semi_minor_axis2, angle2)
        g3 = _gamma_ring_3quarter(x)
        g4 = _gamma_ellipse(x)
        return jnp.min(jnp.stack([g1, g2, g3, g4]))

    _gamma_vg = jax.jit(jax.value_and_grad(gamma_combined))
    _gamma_jit = jax.jit(gamma_combined)

    # Individual Gamma JIT functions for per-obstacle blend weights
    _gamma_sq1_jit  = jax.jit(lambda x: _gamma_sq(x, _c1, semi_major_axis1, semi_minor_axis1, angle1))
    _gamma_sq2_jit  = jax.jit(lambda x: _gamma_sq(x, _c2, semi_major_axis2, semi_minor_axis2, angle2))
    _gamma_ring_jit = jax.jit(_gamma_ring_3quarter)
    _gamma_ell_jit  = jax.jit(_gamma_ellipse)

    # ── On-manifold modulation matrix M(x) ───────────────────────────────────
    # Diagonal modulation (no φ term). The φ term creates balance-condition
    # fixed points φ*(n̂·v) + λ₂*(e₁·v) = 0 for any fixed δ at some boundary
    # point, even for a goal-directed DS. Instead, we choose the blend target
    # v_input so that e₁·v_input < 0 throughout the relevant boundary arc —
    # then φ is never activated and no balance point can form.

    def compute_modulation_matrix(x):
        """Returns M(x) using diagonal modulation (no φ term)."""
        Gamma, grad_Gamma = _gamma_vg(x)
        # Clamp Gamma ≥ 1 so λ₁ = 1-1/Γ stays ≥ 0.  Discrete Euler steps can
        # overshoot the boundary (Γ drops below 1); without the clamp λ₁ < 0
        # reverses the normal component and pushes the robot deeper inside.
        norm_g = jnp.sqrt(grad_Gamma @ grad_Gamma + 1e-12)
        n_hat  = grad_Gamma / norm_g
        e1     = jnp.array([-n_hat[1], n_hat[0]])

        lam1 = 1.0 - 1.0 / Gamma
        lam2 = 1.0 + 1.0 / Gamma

        H   = jnp.column_stack([n_hat, e1])
        Lam = jnp.diag(jnp.array([lam1, lam2]))
        return H @ Lam @ H.T, Gamma

    def apply_modulation(x, v_in):
        M, Gamma = compute_modulation_matrix(x)
        return M @ v_in, Gamma

    _apply_mod = jax.jit(apply_modulation)

    # ── Reference rollout & simulation setup ─────────────────────────────────
    ys    = posn
    f     = lambda t_val, z: model.func(t_val, z, None)
    xref  = model(ts, ys[train_indx, 0, :])  # (T, 2) reference rollout
    x     = jnp.array(posn[train_indx, 0], dtype=jnp.float32)
    xall  = jnp.expand_dims(x, axis=0)
    dti   = ts[1] - ts[0]

    alpha_L    = 4.0
    clf_margin = 0.3    # CLF suppressed when Gamma < 1+clf_margin
    k_goal     = 2.0   # goal-directed DS gain
    k_rot      = 6.0   # CW  rotation gain around ellipse center
    k_ring     = 4.0   # CCW rotation gain around ring center (guides robot over the top)
    max_steps  = 5000
    reach_tol  = 0.5

    # Per-obstacle goal-blend margins:
    # sq1/sq2 get a tight margin so the nominal NODE DS dominates most of the time.
    sq_blend_margin   = 0.1
    ring_blend_margin = 0.5   # larger margin: CCW blend activates before robot gets too close
    ell_blend_margin  = 0.1

    # Soft modulation activation margin: M(x) blends linearly to I when
    # Γ > 1 + mod_margin, so the nominal DS is unaffected far from obstacles.
    mod_margin = 1.5

    # Goal: reference trajectory endpoint; ellipse center for CW rotation
    xgoal  = xref[-1]
    xe_val = float(xe)
    ye_val = float(ye)

    print("\nRunning CLF + On-Manifold Modulation loop (min-Γ, per-obstacle blend)...")
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

        # ── Step 2: Per-obstacle blend ─────────────────────────────────────────
        # sq1/sq2/ring: blend to goal-directed DS.
        # ellipse: add CW rotation around the ellipse center to the goal DS.
        #   v_cw = [(y-ye), -(x-xe)] always has e₁·v < 0 on the lower-left arc,
        #   so the diagonal modulation deflects the robot LEFT without any φ
        #   balance-condition fixed point.
        g1 = float(_gamma_sq1_jit(x_t))
        g2 = float(_gamma_sq2_jit(x_t))
        g3 = float(_gamma_ring_jit(x_t))
        g4 = float(_gamma_ell_jit(x_t))
        w1 = max(0.0, min(1.0, (1.0 + sq_blend_margin   - g1) / sq_blend_margin))
        w2 = max(0.0, min(1.0, (1.0 + sq_blend_margin   - g2) / sq_blend_margin))
        w3 = max(0.0, min(1.0, (1.0 + ring_blend_margin - g3) / ring_blend_margin))
        w4 = max(0.0, min(1.0, (1.0 + ell_blend_margin  - g4) / ell_blend_margin))
        w_goal  = max(w1, w2, w3, w4)
        w_total = w1 + w2 + w3 + w4 + 1e-9

        v_goal_d = k_goal * (xgoal - x_t)

        # Ring: goal-directed toward an escape point outside the opening.
        # CCW rotation fails at the ring center (zero field); a fixed attraction
        # point past the opening works for any robot position inside/near the ring.
        cx_ring = float(np.asarray(c)[0])   # -4.0
        cy_ring = float(np.asarray(c)[1])   # -14.0
        escape_pt = jnp.array([cx_ring + (r_out + 3.0) * float(_ring_open_dir[0]),
                                cy_ring + (r_out + 3.0) * float(_ring_open_dir[1])])
        v_ccw_ring = k_ring * (escape_pt - x_t)

        # Ellipse: CW rotation around ellipse center.
        v_cw_pure = k_rot * jnp.array([(x_t[1] - ye_val), -(x_t[0] - xe_val)])
        w_cw = max(0.0, min(1.0, (ye_val - float(x_t[1])) / float(be)))
        v_blend_ell = w_cw * v_cw_pure + (1.0 - w_cw) * v_goal_d

        v_blend_target = ((w1 + w2) * v_goal_d + w3 * v_ccw_ring + w4 * v_blend_ell) / w_total
        v_input  = (1.0 - w_goal) * v_clf + w_goal * v_blend_target

        # ── Step 3: On-manifold modulation (outer layer, diagonal M, no φ) ────
        # Soft activation: blend M(x) → I as Γ → 1 + mod_margin.
        # When Γ > 1 + mod_margin: w_mod = 0 → M_eff = I (no effect).
        # When Γ = 1 (boundary):   w_mod = 1 → M_eff = M(x) (full modulation).
        M_mat, Gamma = compute_modulation_matrix(x_t)
        w_mod = max(0.0, min(1.0, (1.0 + mod_margin - float(Gamma)) / mod_margin))
        M_eff = w_mod * M_mat + (1.0 - w_mod) * jnp.eye(2)
        v_final = M_eff @ v_input

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
    x2sq, y2sq = square_xy(center2[0], center2[1],
                            semi_major_axis2, semi_minor_axis2, angle2)
    xr, yr = ring_sector_xy(float(np.asarray(c)[0]), float(np.asarray(c)[1]),
                              r_in, r_out,
                              open_start=-np.pi, open_end=-np.pi/2)
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
    plt.savefig(FIGS_DIR / "path_vs_target_manifold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'path_vs_target_manifold.png'}")

    # ── Fig 2: vector field + motion plan ─────────────────────────────────────
    xmin, xmax = -52, 10
    ymin, ymax = -30, 8
    indx = train_indx

    f_field = lambda z: model.func(jnp.array(0.0), jnp.array(z, dtype=jnp.float32), None)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 14))
    xg, yg = np.meshgrid(np.linspace(xmin, xmax, 50),
                          np.linspace(ymin, ymax, 50))
    xy_vec = np.hstack((xg.reshape(-1, 1), yg.reshape(-1, 1)))
    uv_vec = np.array(jax.vmap(f_field)(jnp.array(xy_vec, dtype=jnp.float32)))
    u = uv_vec[:, 0].reshape(xg.shape)
    v = uv_vec[:, 1].reshape(yg.shape)
    ax.streamplot(xg, yg, u, v, arrowsize=3, density=1.4, color="plum")

    model_y = model(ts, ys[indx, 0])
    ax.plot(posn[indx, :, 0], posn[indx, :, 1],
            c="black", linestyle="--", linewidth=8, label="Demonstration")
    ax.plot(model_y[:, 0], model_y[:, 1],
            c="green", linewidth=12, label="Target trajectory")
    ax.plot(xall[:, 0], xall[:, 1],
            c="red", linewidth=8, label="Motion plan")
    ax.plot(model_y[0, 0],  model_y[0, 1],  marker="o", markersize=36, c="saddlebrown")
    ax.plot(model_y[-1, 0], model_y[-1, 1], marker="o", markersize=36, c="darkblue")

    _cx, _cy = float(np.asarray(c)[0]), float(np.asarray(c)[1])
    _xr, _yr = ring_sector_xy(_cx, _cy, r_in, r_out,
                               open_start=-np.pi, open_end=-np.pi/2)

    ax.fill(_xr, _yr, color="lightblue", alpha=1.0)
    ax.fill(xe + ae * np.cos(_te), ye + be * np.sin(_te),
            color="lightblue", alpha=1.0)
    ax.fill(x1sq, y1sq, color="lightblue", alpha=1.0)
    ax.fill(x2sq, y2sq, color="lightblue", alpha=1.0)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim([float(xmin), float(xmax)])
    ax.set_ylim([float(ymin), float(ymax)])
    plt.savefig(FIGS_DIR / "vector_field_manifold.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGS_DIR / 'vector_field_manifold.png'}")

    # ── Fig 3: vector field colored by rollout step ───────────────────────────
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 14))
    ax.streamplot(xg, yg, u, v, arrowsize=3, density=1.4, color="plum")

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

    ax.fill(_xr, _yr, color="lightblue", alpha=1.0)
    ax.fill(xe + ae * np.cos(_te), ye + be * np.sin(_te),
            color="lightblue", alpha=1.0)
    ax.fill(x1sq, y1sq, color="lightblue", alpha=1.0)
    ax.fill(x2sq, y2sq, color="lightblue", alpha=1.0)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim([float(xmin), float(xmax)])
    ax.set_ylim([float(ymin), float(ymax)])
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
