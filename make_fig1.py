#!/usr/bin/env python3
"""Fig. 1: CEP control of the intra-pulse rest-frame polarization angle
(reads results/cep_scan.json)."""
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
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "lines.linewidth": 1.2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_BLUE, C_ORANGE, C_VERMIL = "#0072B2", "#E69F00", "#D55E00"  # Okabe-Ito

with open(os.path.join(HERE, "results", "cep_scan.json")) as f:
    d = json.load(f)
N_A = 2

fig = plt.figure(figsize=(3.4, 7.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.85, 0.85], hspace=0.48)

# (a) potential and signed rest-frame polarization angle for two CEP values, N = 2
ax1 = fig.add_subplot(gs[0])
ax1b = ax1.twinx()
for key, ls, color, lbl in (("phi0", "-", C_BLUE, r"$\varphi_0=0$"),
                            ("phi_pi2", "--", C_VERMIL, r"$\varphi_0=\pi/2$")):
    p = d["panel_a"][key]
    x = np.array(p["eta"]) / (2 * np.pi)
    ax1.plot(x, p["a_x"], color="0.65", lw=1.0, ls=ls, zorder=1, label=r"$a_x(\eta)$, " + lbl)
    ax1b.plot(x, p["sigma_deg"], color=color, lw=1.3, ls=ls, zorder=3, label=r"$\Sigma(\eta)$, " + lbl)
ax1.set_xlabel(r"$\eta / 2\pi$  (optical cycles)")
ax1.set_ylabel(r"$a_x(\eta)$", color="0.4")
ax1.tick_params(axis="y", colors="0.4")
ax1.set_ylim(-0.46, 0.46)
ax1b.set_ylim(-27.5, 27.5)
ax1b.set_ylabel(r"polarization angle $\Sigma$ (deg)")
ax1.set_xlim(0, N_A)
ax1.axhline(0, color="0.85", lw=0.5, zorder=0)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1b.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.005), frameon=False,
           handlelength=1.8, labelspacing=0.25, fontsize=6.0, ncol=2, columnspacing=0.9,
           handletextpad=0.4)
ax1.text(0.03, 0.93, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=9.5, va="top")

# (b) peak |Sigma| vs CEP
ax2 = fig.add_subplot(gs[1])
s2, s8 = d["scan"]["N2"], d["scan"]["N8"]
phi = np.array(s2["phi"]) / np.pi
t2, t8 = np.array(s2["sigma_max"]), np.array(s8["sigma_max"])
ax2.plot(phi, t2, color=C_BLUE, marker="o", ms=2.3, lw=1.1, label=r"$N=2$ cycles")
ax2.plot(phi, t8, color=C_ORANGE, marker="s", ms=2.3, lw=1.1, label=r"$N=8$ cycles")
ax2.set_xlabel(r"$\varphi_0\ /\ \pi$")
ax2.set_ylabel(r"peak angle $\Sigma^{\rm max}$ (deg)")
ax2.set_xlim(0, 2)
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, handlelength=1.6,
           labelspacing=0.3, fontsize=6.3)
ax2.text(0.03, 0.93, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9.5, va="top")
ymin, ymax = t2.min() - 0.9, t8.max() + 2.1
ax2.set_ylim(ymin, ymax)
ax2.annotate(rf"$\Delta\Sigma^{{\rm max}}={np.ptp(t2):.2f}^\circ$", xy=(0.5, t2.min()),
             xytext=(0.42, ymin + 0.3), fontsize=6.3, color=C_BLUE, ha="center", va="bottom")
ax2.annotate(rf"$\Delta\Sigma^{{\rm max}}={np.ptp(t8):.2f}^\circ$", xy=(0.5, t8.max()),
             xytext=(0.5, ymax - 0.3), fontsize=6.3, color=C_ORANGE, ha="center", va="top")

# (c) peak kinetic energy vs CEP
ax3 = fig.add_subplot(gs[2])
for s, color, marker, n in ((s2, C_BLUE, "o", 2), (s8, C_ORANGE, "s", 8)):
    ke = np.array(s["gamma_max"]) - 1
    ax3.plot(phi, ke / ke.mean(), color=color, marker=marker, ms=2.3, lw=1.1,
             label=rf"$N={n}$ ($\Delta={np.ptp(ke) / ke.mean() * 100:.1f}\%$)")
ax3.set_xlabel(r"$\varphi_0\ /\ \pi$")
ax3.set_ylabel(r"$(\gamma_{\rm max}{-}1)\ /\ \langle\gamma_{\rm max}{-}1\rangle$")
ax3.set_xlim(0, 2)
ax3.axhline(1.0, color="0.85", lw=0.5, zorder=0)
ax3.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, handlelength=1.6,
           labelspacing=0.3, fontsize=6.3)
ax3.text(0.03, 0.05, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=9.5, va="bottom")

os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "fig1.pdf"), bbox_inches="tight", pad_inches=0.03)
fig.savefig(os.path.join(HERE, "figs", "fig1.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
print(f"saved figs/fig1.pdf; Sigma_max spreads {np.ptp(t2):.3f} / {np.ptp(t8):.3f} deg")
