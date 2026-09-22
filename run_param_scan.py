#!/usr/bin/env python3
"""Parameter map of the CEP sensitivity of the intra-pulse polarization angle.

For linear polarization the peak rest-frame angle is fixed by the peak vector
potential alone,

    Sigma_max = 2 arctan(a_max/2) + a_e a_max,
    a_max(a0, N, phi) = a0 * m(N, phi),   m = max_eta |Env(eta) cos(eta + phi)|,

so a0 factors out of m.  The map therefore costs no ODE solves: m is tabulated
once per (N, phi) and everything follows in closed form.  Two consequences are
recorded explicitly:

  * the RELATIVE CEP sensitivity of a_max, and hence of the peak energy
    gamma_max - 1 = a_max^2/2, depends on N alone;
  * the absolute spread of Sigma_max grows with a0 and saturates through the
    arctan, so the relative spread of Sigma_max is only weakly a0 dependent.

A DOP853 spot check validates the closed form away from the a0 = 0.42 point
used in the main text.

Output: results/param_scan.json
"""
import json
import os

import numpy as np

from physics import A_E, LaserPulse, integrate_lightfront, rest_frame_spin

HERE = os.path.dirname(os.path.abspath(__file__))
U0 = (1.0, 0.0, 0.0, 0.0)
SZ = (0.0, 0.0, 0.0, 1.0)
N_PHI = 49            # uniform over [0, 2 pi], end points equivalent
N_ETA = 200001        # eta grid for the peak of |Env cos|
A0S = sorted(set(np.round(np.linspace(0.1, 2.0, 20), 3).tolist() + [0.42]))
NS = list(range(1, 11))


def peak_factor(N, phi):
    """m(N, phi) = max_eta |Env(eta) cos(eta + phi)| at unit a0."""
    eta = np.linspace(0.0, 2 * np.pi * N, N_ETA)
    ax, _ = LaserPulse(1.0, N, phi).a(eta)
    return float(np.max(np.abs(ax)))


def sigma_deg(a):
    return np.degrees(2 * np.arctan(np.asarray(a) / 2) + A_E * np.asarray(a))


phis = np.linspace(0.0, 2 * np.pi, N_PHI)
m = np.array([[peak_factor(N, phi) for phi in phis] for N in NS])   # (N, phi)
print(f"tabulated m(N, phi) on {m.shape[0]} x {m.shape[1]} grid", flush=True)

# --- universal, a0-independent quantities (functions of N only) --------------
rel_amax = np.ptp(m, axis=1) / m.mean(axis=1) * 100.0
rel_ke = np.ptp(m ** 2, axis=1) / (m ** 2).mean(axis=1) * 100.0

# --- the map ----------------------------------------------------------------
dsigma = np.zeros((len(A0S), len(NS)))
relsigma = np.zeros_like(dsigma)
for i, a0 in enumerate(A0S):
    s = sigma_deg(a0 * m)                      # (N, phi)
    dsigma[i] = np.ptp(s, axis=1)
    relsigma[i] = np.ptp(s, axis=1) / s.mean(axis=1) * 100.0

# --- consistency with the main-text numbers at a0 = 0.42 --------------------
i42, j2, j8 = A0S.index(0.42), NS.index(2), NS.index(8)
main_text = dict(a0=0.42,
                 sigma_spread_N2_deg=float(dsigma[i42, j2]),
                 sigma_spread_N8_deg=float(dsigma[i42, j8]),
                 suppression=float(dsigma[i42, j2] / dsigma[i42, j8]),
                 ke_spread_N2_percent=float(rel_ke[j2]),
                 ke_spread_N8_percent=float(rel_ke[j8]))
print("at a0 = 0.42:", main_text, flush=True)

# --- DOP853 validation away from the main-text point ------------------------
# The closed form is checked POINTWISE against the reference solution on the
# integrator's own grid; the peak values are compared separately, where the
# residual is set by how finely the grid resolves the maximum, not by the
# closed form itself.
checks = []
for a0, N in ((0.42, 2), (0.42, 8), (1.0, 3), (2.0, 5)):
    for phi in (0.0, 0.7, 2.5):
        p = LaserPulse(a0, N, phi)
        sol, orb, eta = integrate_lightfront(p, U0, SZ, (0, p.T), n_dense=20000)
        zeta = rest_frame_spin(sol.y, orb["gamma"], orb["ux"], orb["uy"], orb["uz"])
        sig_num = np.degrees(np.arctan2(-zeta[0], zeta[2]))
        ax, _ = p.a(eta)
        sig_form = sigma_deg(ax)
        checks.append(dict(a0=a0, N=N, phi=phi,
                           max_abs_pointwise_diff_deg=float(np.max(np.abs(sig_num - sig_form))),
                           peak_dop853_deg=float(np.max(np.abs(sig_num))),
                           peak_formula_deg=float(sigma_deg(a0 * peak_factor(N, phi)))))
        checks[-1]["peak_abs_diff_deg"] = abs(checks[-1]["peak_dop853_deg"]
                                              - checks[-1]["peak_formula_deg"])
worst_pt = max(c["max_abs_pointwise_diff_deg"] for c in checks)
worst_pk = max(c["peak_abs_diff_deg"] for c in checks)
print(f"DOP853 validation: max pointwise |formula - numerical| = {worst_pt:.2e} deg; "
      f"max peak difference (grid resolution) = {worst_pk:.2e} deg", flush=True)

out = dict(
    a0_grid=[float(a) for a in A0S], N_grid=NS, n_phi=N_PHI, n_eta=N_ETA,
    peak_factor_m=m.tolist(),
    sigma_spread_deg=dsigma.tolist(),
    sigma_spread_percent=relsigma.tolist(),
    a_max_spread_percent=rel_amax.tolist(),
    peak_energy_spread_percent=rel_ke.tolist(),
    main_text_point=main_text,
    dop853_validation=checks,
    dop853_max_pointwise_diff_deg=float(worst_pt),
    dop853_max_peak_diff_deg=float(worst_pk),
)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "param_scan.json"), "w", newline="\n") as f:
    json.dump(out, f, indent=1)
print("wrote results/param_scan.json")
