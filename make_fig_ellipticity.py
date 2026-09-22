#!/usr/bin/env python3
"""Net post-pulse rotation and intra-pulse excursion versus ellipticity
(reads results/ellipticity_scan.json).

(a) The net rest-frame rotation vanishes identically at delta = 0 and follows
    the leading-order holonomy 1/2 a_e^2 |A| for delta > 0; it is CEP
    independent to the last digit of the reference integrator.
(b) The intra-pulse peak angle is the opposite case: large, and strongly CEP
    dependent for a few-cycle pulse, with the CEP band closing only at delta = 1,
    where the field is circularly polarized and the CEP is unobservable."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.labelsize": 8.5, "axes.linewidth": 0.7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0, "lines.linewidth": 1.2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]

with open(os.path.join(HERE, "results", "ellipticity_scan.json")) as f:
    d = json.load(f)


def series(N):
    rows = [r for r in d["rows"] if r["N"] == N]
    delta = np.array([r["delta"] for r in rows])
    net = np.array([abs(r["net_rotation_mean_rad"]) for r in rows])
    pred = np.array([abs(r["holonomy_pred_mean_rad"]) for r in rows])
    exc = np.array([[p["excursion_deg"] for p in r["per_phi"]] for r in rows])
    return delta, net, pred, exc


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.6))
fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.10, hspace=0.42)

for N, c, m in ((2, C[0], "o"), (8, C[1], "s")):
    delta, net, pred, _ = series(N)
    ax1.plot(delta, pred * 1e6, color=c, lw=1.1)
    ax1.plot(delta, net * 1e6, m, color=c, ms=3.4, mfc="none", label=fr"$N={N}$")
ax1.set_xlabel(r"ellipticity $\delta$")
ax1.set_ylabel(r"net rotation $|\Theta|$  ($10^{-6}$ rad)")
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(0, None)
ax1.legend(frameon=False, loc="lower right", handlelength=1.6)
ax1.text(0.40, 0.10, r"lines: $\frac{1}{2}a_e^{2}|\mathcal{A}|$", transform=ax1.transAxes, fontsize=7)
ax1.text(0.03, 0.88, "(a)", transform=ax1.transAxes)

for N, c in ((2, C[0]), (8, C[1])):
    delta, _, _, exc = series(N)
    ax2.fill_between(delta, exc.min(axis=1), exc.max(axis=1), color=c, alpha=0.25, lw=0)
    ax2.plot(delta, exc.mean(axis=1), color=c, label=fr"$N={N}$")
ax2.set_xlabel(r"ellipticity $\delta$")
ax2.set_ylabel(r"peak angle $\Sigma^{\max}$ (deg)")
ax2.set_xlim(-0.02, 1.02)
ax2.legend(frameon=False, loc="upper right", handlelength=1.8)
ax2.text(0.03, 0.06, "(b)", transform=ax2.transAxes)

os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
out = os.path.join(HERE, "figs", "fig_ellipticity.pdf")
fig.savefig(out)
print("wrote", out)
