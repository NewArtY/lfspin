#!/usr/bin/env python3
"""Fig. 2: PINN solution and convergence study
(reads results/pinn_main.json and results/convergence.json)."""
import json
import os

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

from physics import LaserPulse, integrate_lightfront

HERE = os.path.dirname(os.path.abspath(__file__))
rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.labelsize": 8.5, "axes.linewidth": 0.7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "lines.linewidth": 1.2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_BLUE, C_ORANGE, C_GREEN, C_VERMIL, C_PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"

with open(os.path.join(HERE, "results", "pinn_main.json")) as f:
    pj = json.load(f)
case, fin = pj["case"], pj["final"]
eta = np.array(fin["eta"])
g_p, S_p = np.array(fin["gamma_pinn"]), np.array(fin["S_pinn"])
pulse = LaserPulse(case["a0"], case["N"], case["phi"], case["omega"])
sol, orb, eta_ref = integrate_lightfront(pulse, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                                         (0, pulse.T), n_dense=len(eta))
assert np.allclose(eta, eta_ref)
g_r, S_r = orb["gamma"], sol.y
err_g, err_S = np.abs(g_p - g_r), np.abs(S_p - S_r)
N = case["N"]
x = eta / (2 * np.pi)

fig = plt.figure(figsize=(3.4, 9.4))
gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 0.55, 0.9], hspace=0.72)

ax1 = fig.add_subplot(gs[0])
ax1.plot(x, g_r, color="#BEBEBE", lw=2.6, label="DOP853 reference", zorder=2, solid_capstyle="round")
ax1.plot(x, g_p, color="#111111", lw=0.9, ls="--", dashes=(4, 1.6), label="PINN", zorder=3)
ax1.set_xlabel(r"$\eta / 2\pi$  (optical cycles)")
ax1.set_ylabel(r"$\gamma(\eta)$")
ax1.set_xlim(0, N)
ax1.legend(loc="upper right", frameon=False, handlelength=1.8, labelspacing=0.3)
ax1.text(0.03, 0.93, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=9.5, va="top")

ax2 = fig.add_subplot(gs[1])
for i, (lbl, c) in enumerate(zip([r"$S^0$", r"$S^1$", r"$S^2$", r"$S^3$"],
                                 [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE])):
    ax2.plot(x, S_r[i], color=c, lw=2.4, alpha=0.45, zorder=2, solid_capstyle="round")
    ax2.plot(x, S_p[i], color=c, lw=0.9, ls="--", dashes=(4, 1.6), zorder=3, label=lbl)
ax2.set_xlabel(r"$\eta / 2\pi$  (optical cycles)")
ax2.set_ylabel(r"spin 4-vector $S^\mu$")
ax2.set_xlim(0, N)
leg1 = ax2.legend(loc="lower right", frameon=False, handlelength=1.8, labelspacing=0.25, fontsize=6.5,
                  ncol=4, columnspacing=0.8)
ax2.add_artist(leg1)
ax2.legend([Line2D([0], [0], color="0.55", lw=2.4, alpha=0.45),
            Line2D([0], [0], color="0.1", lw=0.9, ls="--", dashes=(4, 1.6))],
           ["DOP853 (thick, light)", "PINN (thin, dashed)"], loc="center left",
           bbox_to_anchor=(0.02, 0.62), frameon=False, fontsize=6.2, handlelength=1.8, labelspacing=0.25)
ax2.text(0.03, 0.93, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9.5, va="top")

ax3 = fig.add_subplot(gs[2])
with open(os.path.join(HERE, "results", "convergence.json")) as f:
    conv = json.load(f)
n = np.array(conv["n_steps"])
ax3.loglog(n, conv["rk4"], color=C_BLUE, marker="o", ms=3, lw=1.1, label="lab-time RK4")
ax3.loglog(n, conv["boris"], color=C_VERMIL, marker="^", ms=3.3, lw=1.1, label="Boris–BMT")
ax3.loglog(n, conv["hc"], color=C_PURPLE, marker="v", ms=3.3, lw=1.1, ls="--", label="Higuera–Cary–BMT")
ax3.loglog(n, conv["lightfront"], color=C_GREEN, marker="s", ms=3, lw=1.1, label="light-front RK4")
ax3.set_xlabel(r"number of steps $N_{\rm steps}$")
ax3.set_ylabel("max rel. error (surfing e$^-$)", fontsize=7.0)
ax3.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.5,
           labelspacing=0.15, columnspacing=0.8, fontsize=5.4, borderaxespad=0.2)
ax3.set_ylim(1e-15, 1e3)
ax3.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
ax3.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax3.text(0.03, 0.05, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=9.5, va="bottom")

ax4 = fig.add_subplot(gs[3])
ax4.semilogy(x, err_g, color=C_VERMIL, lw=1.1, label=r"$|\Delta\gamma|$")
ax4.semilogy(x, err_S.max(axis=0), color=C_BLUE, lw=1.1, ls="--", label=r"$\max_\mu|\Delta S^\mu|$")
ax4.set_xlabel(r"$\eta / 2\pi$  (optical cycles)")
ax4.set_ylabel("PINN abs. error")
ax4.set_xlim(0, N)
ax4.set_ylim(1e-7, 5e-3)
ax4.legend(loc="upper left", bbox_to_anchor=(0.1, 1.0), ncol=2, frameon=False, handlelength=1.6,
           columnspacing=1.0, fontsize=6.5)
ax4.text(0.03, 0.9, "(d)", transform=ax4.transAxes, fontweight="bold", fontsize=9.5, va="top")

os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "fig2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figs", "fig2.png"), dpi=300, bbox_inches="tight")
print(f"saved figs/fig2.pdf; PINN max err gamma {err_g.max():.3e}, spin {err_S.max():.3e}")
