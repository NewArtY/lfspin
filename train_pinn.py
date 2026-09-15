#!/usr/bin/env python3
"""
Train a PINN and evaluate it against the DOP853 reference (and, for linear
polarization, the exact closed-form solution).  Case: a0 = 0.42, N = 2,
phi_CEP = 0.7 rad, electron initially at rest, spin along z.

    python train_pinn.py --config main        # linear pol., Fourier features, 5x128, Adam 32000 + L-BFGS 3000
    python train_pinn.py --config ablation    # same as main but plain input eta/T (k_max = 0)
    python train_pinn.py --config elliptical  # as main, elliptical polarization, delta = 0.5 (no closed form)
    python train_pinn.py --config baseline    # plain input, smaller 4x64, Adam 24000, no L-BFGS

Outputs: results/pinn_<config>.json, results/pinn_<config>_model.pt
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import scipy
import torch

from physics import LaserPulse, integrate_lightfront, closed_form_spin_linear
from pinn import train_adam, polish_lbfgs, loss_terms, n_params

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

MAIN = dict(k_max=8, n_hidden=128, n_layers=5, n_steps=32000, lbfgs_iters=3000, pol="linear", delta=None)
CONFIGS = {
    "main": MAIN,
    "ablation": dict(MAIN, k_max=0),
    "elliptical": dict(MAIN, pol="elliptical", delta=0.5),
    "baseline": dict(k_max=0, n_hidden=64, n_layers=4, n_steps=24000, lbfgs_iters=0, pol="linear", delta=None),
}
A0, N, PHI, OMEGA, SEED = 0.42, 2, 0.7, 1.0, 0


def evaluate(model, cfg):
    """Accuracy on a 2000-point grid and loss terms on 1200 uniform points."""
    pulse = LaserPulse(A0, N, PHI, OMEGA, pol=cfg["pol"], delta=cfg["delta"] or 1.0)
    u0, S0 = (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
    sol, orb, eta = integrate_lightfront(pulse, u0, S0, (0, pulse.T), n_dense=2000)
    with torch.no_grad():
        out = model(torch.tensor(eta.reshape(-1, 1)))
    g, ux, uy, uz, *S = [o.numpy().ravel() for o in out]
    S = np.vstack(S)
    uu = g ** 2 - ux ** 2 - uy ** 2 - uz ** 2 - 1
    SS = S[0] ** 2 - S[1] ** 2 - S[2] ** 2 - S[3] ** 2 + 1
    u_ref = np.vstack([orb["ux"], orb["uy"], orb["uz"]])
    eta_fixed = torch.linspace(0, pulse.T, 1200, dtype=torch.float64).reshape(-1, 1)
    l_orb, l_spin, l_inv = [x.item() for x in loss_terms(model, eta_fixed, A0, N, PHI, OMEGA, cfg["delta"])]
    ev = dict(
        max_err_gamma_vs_ref=float(np.max(np.abs(g - orb["gamma"]))),
        max_err_u_vs_ref=float(np.max(np.abs(np.vstack([ux, uy, uz]) - u_ref))),
        max_err_S_vs_ref=float(np.max(np.abs(S - sol.y))),
        max_err_S_per_component_vs_ref=np.max(np.abs(S - sol.y), axis=1).tolist(),
        max_abs_uu_minus_1=float(np.max(np.abs(uu))), rms_uu_minus_1=float(np.sqrt(np.mean(uu ** 2))),
        max_abs_SS_plus_1=float(np.max(np.abs(SS))), rms_SS_plus_1=float(np.sqrt(np.mean(SS ** 2))),
        grid1200_orb_mse=l_orb, grid1200_spin_mse=l_spin, grid1200_inv_mse=l_inv,
        grid1200_total_final_weights=8 * l_orb + 3 * l_spin + 20 * l_inv,
        eta=eta.tolist(), gamma_pinn=g.tolist(), u_pinn=[ux.tolist(), uy.tolist(), uz.tolist()],
        S_pinn=S.tolist(),
    )
    if cfg["pol"] == "linear":
        ax, _ = pulse.a(eta)
        S_exact = closed_form_spin_linear(ax)
        ev["max_err_S_vs_exact"] = float(np.max(np.abs(S - S_exact)))
        ev["ref_vs_exact_max"] = float(np.max(np.abs(sol.y - S_exact)))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, default="main")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = CONFIGS[args.config]
    os.makedirs(RES, exist_ok=True)

    t0 = time.time()
    model, history, best = train_adam(A0, N, PHI, OMEGA, n_steps=cfg["n_steps"], n_colloc=150,
                                      lr=1e-3, seed=SEED, k_max=cfg["k_max"],
                                      n_hidden=cfg["n_hidden"], n_layers=cfg["n_layers"],
                                      ramp_steps=15000, delta=cfg["delta"],
                                      checkpoint=os.path.join(RES, f"pinn_{args.config}_adam.ckpt"))
    t_adam = time.time() - t0
    ev_adam = evaluate(model, cfg)
    for k in ("eta", "gamma_pinn", "u_pinn", "S_pinn"):
        ev_adam.pop(k)
    print("after Adam:", {k: v for k, v in ev_adam.items() if "max_err" in k}, flush=True)

    t1 = time.time()
    if cfg["lbfgs_iters"]:
        model = polish_lbfgs(model, A0, N, PHI, OMEGA, n_colloc=1200, max_iter=cfg["lbfgs_iters"],
                             delta=cfg["delta"])
    t_lbfgs = time.time() - t1
    ev = evaluate(model, cfg)

    torch.save(model.state_dict(), os.path.join(RES, f"pinn_{args.config}_model.pt"))
    out = dict(config=args.config, case=dict(a0=A0, N=N, phi=PHI, omega=OMEGA, seed=SEED,
                                             pol=cfg["pol"], delta=cfg["delta"]), **cfg,
               n_params=n_params(model), history=history, best_adam_loss=best,
               adam_time_s=t_adam, lbfgs_time_s=t_lbfgs, torch_threads=torch.get_num_threads(),
               cpu=platform.processor(), after_adam=ev_adam, final=ev,
               versions=dict(python=platform.python_version(), numpy=np.__version__,
                             scipy=scipy.__version__, torch=torch.__version__))
    with open(os.path.join(RES, f"pinn_{args.config}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("final:", {k: v for k, v in ev.items() if k not in ("eta", "gamma_pinn", "u_pinn", "S_pinn")})
    print(f"Adam {t_adam:.0f} s, L-BFGS {t_lbfgs:.0f} s")


if __name__ == "__main__":
    main()
