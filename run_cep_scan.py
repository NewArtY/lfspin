#!/usr/bin/env python3
"""CEP scan of the intra-pulse spin excursion and peak energy (Fig. 1, Table S2).

Observable: the rest-frame polarization angle Sigma(eta), zeta = (-sin Sigma, 0, cos Sigma),
zeta = S_vec - S^0 u_vec/(gamma+1), extracted from the DOP853 reference; its peak
max_eta |Sigma| is compared with the closed form 2 arctan(a_max/2) + a_e a_max.
The laboratory angle theta_S of the spatial part of S^mu is kept for reference.
Output: results/cep_scan.json"""
import json
import os

import numpy as np

from physics import A_E, LaserPulse, integrate_lightfront, theta_S_lab, rest_frame_spin

HERE = os.path.dirname(os.path.abspath(__file__))
A0 = 0.42
U0 = (1.0, 0.0, 0.0, 0.0)
SZ = (0.0, 0.0, 0.0, 1.0)
N_PHI = 49          # uniformly spaced over [0, 2 pi] (end points equivalent)
N_DENSE = 2500


def run(N, phi, a_e, n_dense=N_DENSE):
    p = LaserPulse(A0, N, phi)
    sol, orb, eta = integrate_lightfront(p, U0, SZ, (0, p.T), n_dense=n_dense, a_e=a_e)
    ax, _ = p.a(eta)
    zeta = rest_frame_spin(sol.y, orb["gamma"], orb["ux"], orb["uy"], orb["uz"])
    sigma = np.degrees(np.arctan2(-zeta[0], zeta[2]))
    return eta, ax, sigma, np.degrees(theta_S_lab(sol.y)), orb["gamma"]


out = dict(a0=A0, n_phi=N_PHI, n_dense=N_DENSE, panel_a={}, scan={})
for phi, key in ((0.0, "phi0"), (np.pi / 2, "phi_pi2")):
    eta, ax, sig, th, _ = run(2, phi, A_E, 2000)
    out["panel_a"][key] = dict(eta=eta.tolist(), a_x=ax.tolist(), sigma_deg=sig.tolist(),
                               theta_S_deg=th.tolist())

phis = np.linspace(0, 2 * np.pi, N_PHI)
for N in (2, 8):
    rows = {k: [] for k in ("sigma_max", "sigma_max_g2", "sigma_max_formula", "theta_max",
                            "a_max", "gamma_max")}
    for phi in phis:
        _, ax, sig, th, g = run(N, phi, A_E)
        _, _, sig2, _, _ = run(N, phi, 0.0)
        am = float(np.max(np.abs(ax)))
        rows["sigma_max"].append(float(np.max(np.abs(sig))))
        rows["sigma_max_g2"].append(float(np.max(np.abs(sig2))))
        rows["sigma_max_formula"].append(float(np.degrees(2 * np.arctan(am / 2) + A_E * am)))
        rows["theta_max"].append(float(th.max()))
        rows["a_max"].append(am)
        rows["gamma_max"].append(float(g.max()))
    s, s2, th = (np.array(rows[k]) for k in ("sigma_max", "sigma_max_g2", "theta_max"))
    ke = np.array(rows["gamma_max"]) - 1
    summary = dict(
        sigma_max_range=[float(s.min()), float(s.max())], sigma_spread_deg=float(np.ptp(s)),
        sigma_spread_g2_deg=float(np.ptp(s2)),
        max_abs_sigma_minus_formula_deg=float(np.max(np.abs(s - np.array(rows["sigma_max_formula"])))),
        anomalous_part_of_peak_deg=float(np.max(s - s2)),
        theta_max_range=[float(th.min()), float(th.max())], theta_spread_deg=float(np.ptp(th)),
        a_max_range=[min(rows["a_max"]), max(rows["a_max"])],
        gamma_max_range=[min(rows["gamma_max"]), max(rows["gamma_max"])],
        ke_spread_percent=float(np.ptp(ke) / ke.mean() * 100))
    out["scan"][f"N{N}"] = dict(phi=phis.tolist(), **rows, summary=summary)
    print(f"N={N}:", summary, flush=True)

S2, S8 = out["scan"]["N2"]["summary"], out["scan"]["N8"]["summary"]
out["suppression_factor"] = S2["sigma_spread_deg"] / S8["sigma_spread_deg"]
out["suppression_factor_theta"] = S2["theta_spread_deg"] / S8["theta_spread_deg"]
print("suppression factor (Sigma):", out["suppression_factor"])
with open(os.path.join(HERE, "results", "cep_scan.json"), "w", newline="\n") as f:
    json.dump(out, f, indent=1)
print("saved results/cep_scan.json")
