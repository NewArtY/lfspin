#!/usr/bin/env python3
"""
Exact results and cross-checks quoted in the Letter and Supplement.
Output: results/exact_checks.json

 1. Closed-form spin (linear polarization, electron at rest) vs DOP853, physical
    and inflated anomaly.
 2. Rest-frame angle Sigma = 2 arctan(a/2) + a_e a vs numerics.
 3. Net rotation after the pulse over 16 CEP values, N = 2 and 8 (light-front DOP853).
 4. Independent proper-time integration of the full covariant system.
 5. gamma - 1 = a^2/2 identity.
 6. Circular polarization: net rotation of a transverse spin vs (1/2) a_e^2 |A|,
    and the change of a longitudinal spin.
"""
import json
import os

import numpy as np

from physics import (A_E, LaserPulse, integrate_lightfront, integrate_reference_tau,
                     closed_form_spin_linear, sigma_closed_form, rest_frame_spin, theta_S_lab)

HERE = os.path.dirname(os.path.abspath(__file__))
A0 = 0.42
U0 = (1.0, 0.0, 0.0, 0.0)
SZ = (0.0, 0.0, 0.0, 1.0)
out = {}

# 1-2. closed form
cf = []
for a_e in (A_E, 0.05, 0.3):
    for N, phi in ((2, 0.0), (2, np.pi / 2), (2, 0.7), (8, 0.7), (8, 2.0)):
        p = LaserPulse(A0, N, phi)
        sol, orb, eta = integrate_lightfront(p, U0, SZ, (0, p.T), n_dense=4000, a_e=a_e)
        ax, _ = p.a(eta)
        dS = np.max(np.abs(sol.y - closed_form_spin_linear(ax, a_e)))
        zeta = rest_frame_spin(sol.y, orb["gamma"], orb["ux"], orb["uy"], orb["uz"])
        sig_num = np.arctan2(-zeta[0], zeta[2])
        dsig = np.max(np.abs(sig_num - sigma_closed_form(ax, a_e)))
        cf.append(dict(a_e=a_e, N=N, phi=phi, max_abs_dS=float(dS), max_abs_dSigma_rad=float(dsig)))
out["closed_form_vs_DOP853"] = cf
out["closed_form_max_dS_overall"] = max(c["max_abs_dS"] for c in cf)
out["closed_form_max_dSigma_overall"] = max(c["max_abs_dSigma_rad"] for c in cf)
print("closed form: max|dS| =", out["closed_form_max_dS_overall"],
      " max|dSigma| =", out["closed_form_max_dSigma_overall"])

# peak rest-frame angle and anomalous content at a = a0
out["Sigma_peak_deg"] = float(np.degrees(sigma_closed_form(A0)))
out["Sigma_peak_g2_deg"] = float(np.degrees(sigma_closed_form(A0, 0.0)))
out["anomalous_part_at_peak_deg"] = float(np.degrees(A_E * A0))

# 3. net rotation over 16 CEPs
phis16 = np.arange(16) * 2 * np.pi / 16
for N in (2, 8):
    net = []
    for phi in phis16:
        p = LaserPulse(A0, N, phi)
        sol, _, _ = integrate_lightfront(p, U0, SZ, (0, p.T), n_dense=50)
        net.append(abs(np.degrees(theta_S_lab(sol.y[:, -1]))))
    out[f"net_rotation_max_deg_N{N}"] = float(max(net))
    print(f"N={N}: max net rotation over 16 CEPs = {max(net):.2e} deg")

# 4. proper-time cross-check (kappa = 1, so tau = eta)
tau_net, tau_g = [], []
for phi in phis16:
    p = LaserPulse(A0, 2, phi)
    sol = integrate_reference_tau(p, U0, SZ, (0, p.T), n_dense=50)
    tau_net.append(abs(np.degrees(theta_S_lab(sol.y[8:12, -1]))))
    tau_g.append(abs(sol.y[4, -1] - 1.0))
out["tau_crosscheck_net_rotation_max_deg_N2"] = float(max(tau_net))
out["tau_crosscheck_gamma_minus_1_max_N2"] = float(max(tau_g))
print("tau cross-check:", max(tau_net), "deg; |gamma-1| =", max(tau_g))

# 5. gamma identity
p = LaserPulse(A0, 2, 0.7)
_, orb, eta = integrate_lightfront(p, U0, SZ, (0, p.T), n_dense=4000)
ax, _ = p.a(eta)
out["max_abs_gamma_minus_1_minus_a2_over_2"] = float(np.max(np.abs(orb["gamma"] - 1 - ax ** 2 / 2)))

# 6. circular polarization: holonomy
circ = []
for N in (2, 8):
    for phi in (0.0, 0.7, 2.0):
        p = LaserPulse(A0, N, phi, pol="circular")
        area = p.signed_area()
        sol, _, _ = integrate_lightfront(p, U0, (0.0, 1.0, 0.0, 0.0), (0, p.T), n_dense=50,
                                         rtol=1e-12, atol=1e-14)
        Sx, Sy = sol.y[1, -1], sol.y[2, -1]
        theta = np.arctan2(Sy, Sx)
        solz, _, _ = integrate_lightfront(p, U0, SZ, (0, p.T), n_dense=50, rtol=1e-12, atol=1e-14)
        circ.append(dict(N=N, phi=phi, signed_area=float(area), net_rotation_rad=float(theta),
                         half_ae2_area_rad=float(0.5 * A_E ** 2 * area),
                         ratio=float(abs(theta) / (0.5 * A_E ** 2 * abs(area))),
                         longitudinal_spin_final_angle_deg=float(np.degrees(theta_S_lab(solz.y[:, -1])))))
        print(f"circular N={N} phi={phi}: |Theta|={abs(theta):.4e} rad, "
              f"a_e^2|A|/2={0.5 * A_E ** 2 * abs(area):.4e}")
out["circular_holonomy"] = circ

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "exact_checks.json"), "w", newline="\n") as f:
    json.dump(out, f, indent=1)
print("saved results/exact_checks.json")
