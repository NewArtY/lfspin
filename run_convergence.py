#!/usr/bin/env python3
"""Fixed-step-budget convergence study for a co-propagating electron.
Output: results/convergence.json (Fig. 2(c), Table S1)."""
import json
import os
import time

import numpy as np

from physics import LaserPulse, integrate_lightfront, orbital_solution
from convergence import rhs_labtime, rk4_fixed, boris_bmt, lightfront_rk4_fixed

HERE = os.path.dirname(os.path.abspath(__file__))
A0, N, PHI = 0.42, 2, 0.7
GAMMA0 = 10.0
UZ0 = np.sqrt(GAMMA0 ** 2 - 1)
U0 = (GAMMA0, 0.0, 0.0, UZ0)
S0 = (UZ0, 0.0, 0.0, GAMMA0)  # longitudinal spin: S.u = 0, S.S = -1
KAPPA = GAMMA0 - UZ0
STEPS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def main():
    pulse = LaserPulse(A0, N, PHI)
    sol, orb, _ = integrate_lightfront(pulse, U0, S0, (0, pulse.T), n_dense=2000, rtol=1e-13, atol=1e-15)
    g_ref, S_ref = float(orb["gamma"][-1]), sol.y[:, -1]
    S_scale = np.max(np.abs(S_ref))
    # lab-time crossing duration t1 = int_0^T gamma/(omega kappa) d eta; the integrand is
    # smooth with vanishing derivatives at the pulse edges, so the trapezoid rule is exact
    # to roundoff on a fine grid
    eta_f = np.linspace(0, pulse.T, 400001)
    t1 = float(np.trapezoid(orbital_solution(pulse, eta_f, U0)["gamma"] / KAPPA, eta_f))

    def err(g, S):
        eg = abs(g - g_ref) / g_ref
        es = float(np.max(np.abs(np.asarray(S) - S_ref)) / S_scale)
        return eg, es, max(eg, es)

    keys = ("rk4", "boris", "hc", "lightfront")
    res = {k: [] for k in keys + tuple(k + "_split" for k in keys)}
    y0 = [0.0, 0.0, GAMMA0, 0.0, 0.0, UZ0, *S0]
    for n in STEPS:
        t0 = time.time()
        yr = rk4_fixed(rhs_labtime, 0.0, t1, y0, n, pulse)
        yb = boris_bmt(0.0, t1, y0, n, pulse, scheme="boris")
        yh = boris_bmt(0.0, t1, y0, n, pulse, scheme="hc")
        gl, Sl = lightfront_rk4_fixed(pulse, U0, S0, 0.0, pulse.T, n)
        for key, (g, S) in (("rk4", (yr[2], yr[6:10])), ("boris", (yb[2], yb[6:10])),
                            ("hc", (yh[2], yh[6:10])), ("lightfront", (gl, Sl))):
            eg, es, e = err(g, S)
            res[key].append(e)
            res[key + "_split"].append((eg, es))
        print(f"n={n:5d} RK4={res['rk4'][-1]:.3e} Boris={res['boris'][-1]:.3e} HC={res['hc'][-1]:.3e} "
              f"LF={res['lightfront'][-1]:.3e} [{time.time() - t0:.1f}s]", flush=True)

    out = dict(a0=A0, N=N, phi=PHI, gamma0=GAMMA0, uz0=UZ0, kappa=KAPPA, S0=S0, t1=t1,
               gamma_ref_end=g_ref, S_ref_end=S_ref.tolist(), n_steps=STEPS,
               deta_dt_range=[float(KAPPA / orb["gamma"].max()), float(KAPPA / orb["gamma"].min())],
               error_metric="max(|dgamma|/gamma_ref, max_mu|dS^mu|/max_mu|S_ref^mu|) at the end of the pulse",
               **res)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "convergence.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("t1 =", t1, " saved results/convergence.json")


if __name__ == "__main__":
    main()
