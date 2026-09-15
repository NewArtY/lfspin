#!/usr/bin/env python3
"""Fig. S1: PINN solution for elliptical polarization (no closed form), against the
DOP853 reference (reads results/pinn_elliptical.json)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from physics import LaserPulse, integrate_lightfront

HERE = os.path.dirname(os.path.abspath(__file__))
rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.labelsize": 8.5, "axes.linewidth": 0.7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.5, "lines.linewidth": 1.2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]

with open(os.path.join(HERE, "results", "pinn_elliptical.json")) as f:
    pj = json.load(f)
case, fin = pj["case"], pj["final"]
eta = np.array(fin["eta"])
g_p, u_p, S_p = np.array(fin["gamma_pinn"]), np.array(fin["u_pinn"]), np.array(fin["S_pinn"])
pulse = LaserPulse(case["a0"], case["N"], case["phi"], case["omega"], pol=case["pol"], delta=case["delta"])
sol, orb, eta_ref = integrate_lightfront(pulse, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                                         (0, pulse.T), n_dense=len(eta))
assert np.allclose(eta, eta_ref)
u_r = np.vstack([orb["ux"], orb["uy"], orb["uz"]])
x = eta / (2 * np.pi)

fig = plt.figure(figsize=(3.4, 6.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.8], hspace=0.5)

ax1 = fig.add_subplot(gs[0])
for i, lbl in enumerate([r"$u_x$", r"$u_y$", r"$u_z$"]):
    ax1.plot(x, u_r[i], color=C[i], lw=2.4, alpha=0.45)
    ax1.plot(x, u_p[i], color=C[i], lw=0.9, ls="--", dashes=(4, 1.6), label=lbl)
ax1.set_ylabel("four-velocity")
ax1.set_xlim(0, case["N"])
ax1.legend(loc="lower right", ncol=3, frameon=False, handlelength=1.6, columnspacing=0.8)
ax1.text(0.03, 0.93, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=9.5, va="top")

ax2 = fig.add_subplot(gs[1])
for i, lbl in enumerate([r"$S^0$", r"$S^1$", r"$S^2$", r"$S^3$"]):
    ax2.plot(x, sol.y[i], color=C[i], lw=2.4, alpha=0.45)
    ax2.plot(x, S_p[i], color=C[i], lw=0.9, ls="--", dashes=(4, 1.6), label=lbl)
ax2.set_ylabel(r"spin 4-vector $S^\mu$")
ax2.set_xlim(0, case["N"])
ax2.legend(loc="center left", ncol=2, frameon=False, handlelength=1.6, columnspacing=0.8)
ax2.text(0.03, 0.93, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9.5, va="top")

ax3 = fig.add_subplot(gs[2])
ax3.semilogy(x, np.abs(g_p - orb["gamma"]), color=C[4], lw=1.0, label=r"$|\Delta\gamma|$")
ax3.semilogy(x, np.abs(u_p - u_r).max(axis=0), color=C[2], lw=1.0, ls=":", label=r"$\max_i|\Delta u_i|$")
ax3.semilogy(x, np.abs(S_p - sol.y).max(axis=0), color=C[0], lw=1.0, ls="--", label=r"$\max_\mu|\Delta S^\mu|$")
ax3.set_xlabel(r"$\eta / 2\pi$  (optical cycles)")
ax3.set_ylabel("PINN abs. error")
ax3.set_xlim(0, case["N"])
ax3.set_ylim(1e-7, 1e-1)
ax3.legend(loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=3, frameon=False, handlelength=1.5,
           columnspacing=0.7, fontsize=6.0)
ax3.text(0.03, 0.9, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=9.5, va="top")

os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "figS1.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figs", "figS1.png"), dpi=300, bbox_inches="tight")
print(f"saved figs/figS1.pdf; gamma {fin['max_err_gamma_vs_ref']:.3e}, u {fin['max_err_u_vs_ref']:.3e}, "
      f"S {fin['max_err_S_vs_ref']:.3e}")
