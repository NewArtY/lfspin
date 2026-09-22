#!/usr/bin/env python3
"""Orthogonality S.u of the trained networks, evaluated from the saved weights.

The BMT equation preserves S.u = 0 exactly, but the loss never asks for it: the
invariant penalties cover only u.u - 1 and S.S + 1.  The orthogonality of the
trained solution is therefore an independent check, and it is not recorded
during training, so it is computed here from the checkpoints.

Each network is also re-evaluated against the reference integrator; the
resulting max errors must reproduce the values stored in results/pinn_*.json,
which validates that the architecture is reconstructed correctly.

Output: results/pinn_invariants.json
"""
import json
import os

import numpy as np
import torch

from physics import LaserPulse, integrate_lightfront
from pinn import PINN

HERE = os.path.dirname(os.path.abspath(__file__))
A0, N, PHI, OMEGA = 0.42, 2, 0.7, 1.0
T = 2 * np.pi * N

CONFIGS = {
    "main": dict(k_max=8, n_hidden=128, n_layers=5, pol="linear", delta=None),
    "ablation": dict(k_max=0, n_hidden=128, n_layers=5, pol="linear", delta=None),
    "baseline": dict(k_max=0, n_hidden=64, n_layers=4, pol="linear", delta=None),
    "elliptical": dict(k_max=8, n_hidden=128, n_layers=5, pol="elliptical", delta=0.5),
}

out = {}
for name, c in CONFIGS.items():
    model = PINN(n_hidden=c["n_hidden"], n_layers=c["n_layers"], k_max=c["k_max"],
                 omega=OMEGA, T=T)
    state = torch.load(os.path.join(HERE, "results", f"pinn_{name}_model.pt"),
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()

    pulse = LaserPulse(A0, N, PHI, OMEGA, pol=c["pol"], delta=c["delta"] or 1.0)
    sol, orb, eta = integrate_lightfront(pulse, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                                         (0, pulse.T), n_dense=2000)
    with torch.no_grad():
        o = model(torch.tensor(eta.reshape(-1, 1)))
    g, ux, uy, uz, *S = [x.numpy().ravel() for x in o]
    S = np.vstack(S)

    Su = S[0] * g - S[1] * ux - S[2] * uy - S[3] * uz
    gr, uxr, uyr, uzr = orb["gamma"], orb["ux"], orb["uy"], orb["uz"]
    Su_ref = sol.y[0] * gr - sol.y[1] * uxr - sol.y[2] * uyr - sol.y[3] * uzr

    with open(os.path.join(HERE, "results", f"pinn_{name}.json")) as f:
        stored = json.load(f)["final"]

    rec = dict(
        max_abs_S_dot_u=float(np.max(np.abs(Su))),
        rms_S_dot_u=float(np.sqrt(np.mean(Su ** 2))),
        reference_max_abs_S_dot_u=float(np.max(np.abs(Su_ref))),
        recomputed_max_err_gamma=float(np.max(np.abs(g - gr))),
        stored_max_err_gamma=stored["max_err_gamma_vs_ref"],
        recomputed_max_err_S=float(np.max(np.abs(S - sol.y))),
        stored_max_err_S=stored["max_err_S_vs_ref"],
    )
    rec["reconstruction_ok"] = bool(
        abs(rec["recomputed_max_err_gamma"] - rec["stored_max_err_gamma"]) < 1e-12
        and abs(rec["recomputed_max_err_S"] - rec["stored_max_err_S"]) < 1e-12)
    out[name] = rec
    print(f"{name:11s} max|S.u| = {rec['max_abs_S_dot_u']:.3e}  "
          f"(reference {rec['reference_max_abs_S_dot_u']:.1e}), "
          f"reconstruction_ok={rec['reconstruction_ok']}", flush=True)

out["note"] = ("S.u is never imposed: the loss penalizes only u.u - 1 and S.S + 1. "
               "Values are maxima over a 2000-point grid across the pulse.")
with open(os.path.join(HERE, "results", "pinn_invariants.json"), "w", newline="\n") as f:
    json.dump(out, f, indent=1)
print("wrote results/pinn_invariants.json")
