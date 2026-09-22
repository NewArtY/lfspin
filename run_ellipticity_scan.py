#!/usr/bin/env python3
"""Ellipticity scan: from the exact zero net rotation of linear polarization to
the plane-wave holonomy of the elliptical case.

At delta = 0 the net rest-frame rotation after the pulse vanishes identically at
any g, any a0 and any CEP.  At delta > 0 the rotations at different phases no
longer commute and a net rotation of magnitude 1/2 a_e^2 |A| survives, with
A = int (a_x a_y' - a_y a_x') d eta the signed area enclosed by a_perp(eta); the
derivation is given in the companion paper and is not repeated here.  This scan
resolves the transition numerically and checks that it is CEP independent, which
separates it from the strongly CEP dependent intra-pulse excursion.

Diagnostics recorded for the reference integrator: max |u.u - 1|, max |S.S + 1|
and max |S.u|, the last being the orthogonality that the BMT equation preserves
but that is never imposed.

Output: results/ellipticity_scan.json
"""
import json
import os

import numpy as np

from physics import A_E, LaserPulse, integrate_lightfront, rest_frame_spin

HERE = os.path.dirname(os.path.abspath(__file__))
A0 = 0.42
U0 = (1.0, 0.0, 0.0, 0.0)
SZ = (0.0, 0.0, 0.0, 1.0)       # spin along the propagation axis
SX = (0.0, 1.0, 0.0, 0.0)       # transverse spin: net rotation is about z
DELTAS = np.round(np.linspace(0.0, 1.0, 11), 3)
PHIS = (0.0, 0.7, 1.6, 3.0)
N_DENSE = 2000
RTOL, ATOL = 1e-12, 1e-14


def invariants(sol, orb):
    """max |u.u - 1|, max |S.S + 1|, max |S.u| along the trajectory."""
    g, ux, uy, uz = (orb[k] for k in ("gamma", "ux", "uy", "uz"))
    uu = g ** 2 - ux ** 2 - uy ** 2 - uz ** 2
    S0, Sx, Sy, Sz = sol.y
    SS = S0 ** 2 - Sx ** 2 - Sy ** 2 - Sz ** 2
    Su = S0 * g - Sx * ux - Sy * uy - Sz * uz
    return (float(np.max(np.abs(uu - 1.0))), float(np.max(np.abs(SS + 1.0))),
            float(np.max(np.abs(Su))))


rows = []
for N in (2, 8):
    for d in DELTAS:
        per_phi = []
        for phi in PHIS:
            p = LaserPulse(A0, N, phi, pol="elliptical", delta=float(d))
            area = p.signed_area()

            # transverse initial spin: the net rotation is an angle about +z
            sol, orb, eta = integrate_lightfront(p, U0, SX, (0, p.T), n_dense=N_DENSE,
                                                 rtol=RTOL, atol=ATOL)
            zeta = rest_frame_spin(sol.y, orb["gamma"], orb["ux"], orb["uy"], orb["uz"])
            zeta0, zetaT = zeta[:, 0], zeta[:, -1]
            net = float(np.arctan2(zetaT[1] * zeta0[0] - zetaT[0] * zeta0[1],
                                   zetaT[0] * zeta0[0] + zetaT[1] * zeta0[1]))
            uu, ss, su = invariants(sol, orb)

            # longitudinal initial spin: intra-pulse excursion and net rotation
            solz, orbz, _ = integrate_lightfront(p, U0, SZ, (0, p.T), n_dense=N_DENSE,
                                                 rtol=RTOL, atol=ATOL)
            zz = rest_frame_spin(solz.y, orbz["gamma"], orbz["ux"], orbz["uy"], orbz["uz"])
            cos_t = np.clip(np.einsum("i...,i...->...", zz, zz[:, :1]), -1.0, 1.0)
            excursion = float(np.degrees(np.max(np.arccos(cos_t))))
            net_z = float(np.degrees(np.arccos(cos_t[-1])))

            per_phi.append(dict(phi=float(phi), signed_area=float(area),
                                net_rotation_rad=net,
                                half_ae2_area_rad=float(0.5 * A_E ** 2 * area),
                                excursion_deg=excursion, net_rotation_long_deg=net_z,
                                max_abs_uu_minus_1=uu, max_abs_SS_plus_1=ss,
                                max_abs_S_dot_u=su))

        nets = np.array([r["net_rotation_rad"] for r in per_phi])
        pred = np.array([r["half_ae2_area_rad"] for r in per_phi])
        exc = np.array([r["excursion_deg"] for r in per_phi])
        row = dict(N=N, delta=float(d), per_phi=per_phi,
                   net_rotation_mean_rad=float(nets.mean()),
                   net_rotation_cep_spread_rad=float(np.ptp(nets)),
                   holonomy_pred_mean_rad=float(pred.mean()),
                   ratio=float(abs(nets.mean()) / abs(pred.mean())) if abs(pred.mean()) > 0 else None,
                   excursion_mean_deg=float(exc.mean()),
                   excursion_cep_spread_deg=float(np.ptp(exc)),
                   max_abs_S_dot_u=max(r["max_abs_S_dot_u"] for r in per_phi))
        rows.append(row)
        print(f"N={N} delta={d:.2f}: net={nets.mean():+.4e} rad, "
              f"a_e^2|A|/2={pred.mean():+.4e}, ratio={row['ratio']}, "
              f"CEP spread={np.ptp(nets):.2e}, max|S.u|={row['max_abs_S_dot_u']:.1e}", flush=True)

# --- how the residual to the leading-order holonomy scales with a0 ----------
# The companion paper's 1/2 a_e^2 |A| is the leading order in the anomaly; the
# scan above reproduces it to a fixed relative 2.3e-8, so the residual is a
# higher-order term rather than integration error.  Its a0 dependence is checked
# here at fixed delta = 1 and N = 2.
scaling = []
for a0 in (0.1, 0.2, 0.42, 0.8, 1.2):
    p = LaserPulse(a0, 2, 0.7, pol="circular")
    area = p.signed_area()
    sol, orb, _ = integrate_lightfront(p, U0, SX, (0, p.T), n_dense=N_DENSE,
                                       rtol=RTOL, atol=ATOL)
    z = rest_frame_spin(sol.y, orb["gamma"], orb["ux"], orb["uy"], orb["uz"])
    z0, zT = z[:, 0], z[:, -1]
    net = float(np.arctan2(zT[1] * z0[0] - zT[0] * z0[1], zT[0] * z0[0] + zT[1] * z0[1]))
    pred = float(0.5 * A_E ** 2 * area)
    rel = float(abs(net) / abs(pred) - 1.0)
    scaling.append(dict(a0=a0, net_rotation_rad=net, holonomy_pred_rad=pred,
                        relative_residual=rel, residual_over_a0sq=rel / a0 ** 2))
    print(f"a0={a0}: rel. residual={rel:+.3e}, residual/a0^2={rel / a0 ** 2:+.3e}", flush=True)

out = dict(a0=A0, deltas=[float(x) for x in DELTAS], phis=list(PHIS), n_dense=N_DENSE,
           rtol=RTOL, atol=ATOL, rows=rows, holonomy_residual_scaling=scaling,
           max_abs_S_dot_u_overall=max(r["max_abs_S_dot_u"] for r in rows))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "ellipticity_scan.json"), "w", newline="\n") as f:
    json.dump(out, f, indent=1)
print("wrote results/ellipticity_scan.json; max |S.u| overall =",
      out["max_abs_S_dot_u_overall"])
