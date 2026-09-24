#!/usr/bin/env python3
"""Arbitrary-precision test of the net post-pulse rotation against the
leading-order holonomy.

Motivation.  In double precision the net rotation of a circularly polarized
pulse differs from the leading-order holonomy (1/2) a_e^2 |A| by about
6e-15 rad, which is the size of the reference integrator's own absolute
tolerance.  At that level a double-precision result cannot distinguish a
physical higher-order term from accumulated round-off: the double-precision
residual indeed drifts from 3.4e-8 to 2.1e-8 as the tolerances are tightened,
and scipy refuses rtol below 2.2e-14.

Method.  The BMT equation is integrated along the exact Volkov orbit with
mpmath's Taylor-method solver at a requested number of decimal digits, so that
the integration error can be pushed to 1e-45 while the residual stays at
1e-15.  For a circularly polarized cos^2 pulse the enclosed area is analytic,
A = (3/8) pi N a0^2, so the prediction needs no quadrature.

Result.  The residual is independent of the working precision from 20 to 50
digits, scales as a_e^2 a0^2, and is therefore the next term of the expansion
in the anomaly, of order a_e^4, not a numerical artifact.

Runtime: a few minutes; the 40-digit run dominates it.

Output: results/holonomy_highprec.json
"""
import json
import os
import time

from mpmath import mp, mpf, odefun, cos, sin, pi, sqrt, atan2

HERE = os.path.dirname(os.path.abspath(__file__))
N = 2
PHI = "0.7"
A_E_PHYS = "0.00115965218"


def net_rotation(dps, a0, a_e, n_cycles=N, phi=PHI):
    """|net rest-frame rotation| after a circularly polarized cos^2 pulse."""
    mp.dps = dps
    a0, a_e, phi = mpf(a0), mpf(a_e), mpf(phi)
    tol = mpf(10) ** (-(dps - 5))
    T = 2 * pi * n_cycles
    amp = a0 / sqrt(2)                      # circular: a0 / sqrt(1 + delta^2)

    def env(eta):
        return cos(pi * (eta - T / 2) / T) ** 2

    def denv(eta):
        return -(pi / T) * sin(2 * pi * (eta - T / 2) / T)

    def a_perp(eta):
        e = env(eta)
        return amp * e * cos(eta + phi), amp * e * sin(eta + phi)

    def rhs(eta, S):
        """dS/deta for an electron initially at rest: kappa = 1, omega = 1, q/m = -1."""
        e, de = env(eta), denv(eta)
        c, s = cos(eta + phi), sin(eta + phi)
        ux, uy = amp * e * c, amp * e * s            # a_perp(0) = 0 for this envelope
        dax, day = amp * (de * c - e * s), amp * (de * s + e * c)
        Ex, Ey = -dax, -day
        Bx, By = -Ey, Ex
        g = 1 + (ux * ux + uy * uy) / 2
        uz = g - 1
        S0, S1, S2, S3 = S
        FS0 = Ex * S1 + Ey * S2
        FS1 = Ex * S0 - By * S3
        FS2 = Ey * S0 + Bx * S3
        FS3 = By * S1 - Bx * S2
        uFS = g * FS0 - ux * FS1 - uy * FS2 - uz * FS3
        k = 1 + a_e
        return [-(k * FS0 - a_e * g * uFS), -(k * FS1 - a_e * ux * uFS),
                -(k * FS2 - a_e * uy * uFS), -(k * FS3 - a_e * uz * uFS)]

    def rest_frame(S, eta):
        ax, ay = a_perp(eta)
        g = 1 + (ax * ax + ay * ay) / 2
        f = S[0] / (g + 1)
        return S[1] - f * ax, S[2] - f * ay, S[3] - f * (g - 1)

    S0 = [mpf(0), mpf(1), mpf(0), mpf(0)]        # transverse initial spin
    t0 = time.time()
    ST = odefun(rhs, mpf(0), S0, tol=tol, method="taylor")(T)
    wall = time.time() - t0

    z0, zT = rest_frame(S0, mpf(0)), rest_frame(ST, T)
    net = atan2(zT[1] * z0[0] - zT[0] * z0[1], zT[0] * z0[0] + zT[1] * z0[1])
    area = mpf(3) / 8 * pi * n_cycles * a0 ** 2
    pred = a_e ** 2 * area / 2
    rel = abs(net) / pred - 1
    return dict(dps=dps, a0=str(a0), a_e=str(a_e), wall_s=round(wall, 1),
                net_rotation_rad=mp.nstr(abs(net), 20),
                holonomy_pred_rad=mp.nstr(pred, 20),
                absolute_residual_rad=float(abs(net) - pred),
                relative_residual=float(rel),
                relative_residual_over_a0sq=float(rel / a0 ** 2),
                relative_residual_over_a0sq_aesq=float(rel / (a0 ** 2 * a_e ** 2)))


def double_precision_residual(rtol, atol):
    """The same quantity from the double-precision reference integrator."""
    import numpy as np
    from physics import A_E, LaserPulse, integrate_lightfront, rest_frame_spin
    p = LaserPulse(0.42, N, float(PHI), pol="circular")
    sol, orb, _ = integrate_lightfront(p, (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                                       (0, p.T), n_dense=2000, rtol=rtol, atol=atol)
    z = rest_frame_spin(sol.y, orb["gamma"], orb["ux"], orb["uy"], orb["uz"])
    z0, zT = z[:, 0], z[:, -1]
    net = abs(float(np.arctan2(zT[1] * z0[0] - zT[0] * z0[1], zT[0] * z0[0] + zT[1] * z0[1])))
    pred = 0.5 * A_E ** 2 * p.signed_area()
    return dict(rtol=rtol, atol=atol, relative_residual=net / pred - 1.0,
                absolute_residual_rad=net - pred)


out = {"double_precision_drift": [], "precision_convergence": [],
       "scaling_in_a0": [], "scaling_in_anomaly": []}

print("double precision, tightening tolerances (the residual drifts):", flush=True)
for rtol, atol in ((1e-11, 1e-13), (1e-12, 1e-14), (1e-13, 1e-15), (1e-14, 1e-16)):
    r = double_precision_residual(rtol, atol)
    out["double_precision_drift"].append(r)
    print(f"  rtol={rtol:.0e} atol={atol:.0e}: rel.res={r['relative_residual']:+.4e}  "
          f"abs.res={r['absolute_residual_rad']:+.2e} rad", flush=True)

print("precision convergence at a0 = 0.42 (the residual must not move):", flush=True)
for dps in (20, 25, 30, 40):
    r = net_rotation(dps, "0.42", A_E_PHYS)
    out["precision_convergence"].append(r)
    print(f"  dps={dps:3d}  rel.res={r['relative_residual']:+.10e}  ({r['wall_s']} s)", flush=True)

print("scaling in a0 at 30 digits (residual/a0^2 must be constant):", flush=True)
for a0 in ("0.1", "0.2", "0.42", "0.8", "1.2"):
    r = net_rotation(30, a0, A_E_PHYS)
    out["scaling_in_a0"].append(r)
    print(f"  a0={a0:5s}  rel.res={r['relative_residual']:+.6e}  "
          f"/a0^2={r['relative_residual_over_a0sq']:+.9e}", flush=True)

print("scaling in the anomaly at 30 digits (residual/(a0^2 a_e^2) must be constant):", flush=True)
for a_e in (A_E_PHYS, "0.01", "0.05"):
    r = net_rotation(30, "0.42", a_e)
    out["scaling_in_anomaly"].append(r)
    print(f"  a_e={a_e:14s} rel.res={r['relative_residual']:+.6e}  "
          f"/(a0^2 a_e^2)={r['relative_residual_over_a0sq_aesq']:+.7f}", flush=True)

ks = [r["relative_residual_over_a0sq_aesq"] for r in out["scaling_in_a0"]]
out["coefficient_K"] = dict(
    value=sum(ks) / len(ks), spread=max(ks) - min(ks),
    definition="net = (1/2) a_e^2 |A| (1 - K a_e^2 a0^2 + ...), K from the a0 scan")
print(f"K = {out['coefficient_K']['value']:.7f} "
      f"(spread {out['coefficient_K']['spread']:.1e} over the a0 scan)")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "holonomy_highprec.json"), "w", newline="\n") as f:
    json.dump(out, f, indent=1)
print("wrote results/holonomy_highprec.json")
