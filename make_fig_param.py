#!/usr/bin/env python3
"""Parametric CEP sensitivity of the intra-pulse observables (reads
results/param_scan.json).

(a) The relative CEP spread of the peak vector potential and of the peak kinetic
    energy depends on the number of cycles alone: a0 factors out of
    a_max = a0 m(N, phi), so these curves hold at any intensity.
(b) The absolute spread of the peak rest-frame polarization angle does depend on
    a0, and saturates through the arctan of the closed form."""
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

with open(os.path.join(HERE, "results", "param_scan.json")) as f:
    d = json.load(f)
N = np.array(d["N_grid"])
a0 = np.array(d["a0_grid"])
spread = np.array(d["sigma_spread_deg"])          # (a0, N), degrees
rel_a = np.array(d["a_max_spread_percent"])
rel_ke = np.array(d["peak_energy_spread_percent"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.6))
fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.10, hspace=0.42)

ax1.semilogy(N, rel_ke, "s--", color=C[1], ms=3.6, label=r"$\gamma_{\max}-1$")
ax1.semilogy(N, rel_a, "o-", color=C[0], ms=3.6, label=r"$a_{\max}$")
ax1.set_xlabel(r"number of cycles $N$")
ax1.set_ylabel("relative CEP spread (%)")
ax1.set_xticks(N)
ax1.set_xlim(0.7, 10.3)
ax1.legend(frameon=False, loc="upper right", handlelength=1.8)
ax1.text(0.03, 0.06, "(a)  independent of $a_0$", transform=ax1.transAxes)

for k, (n, c) in enumerate(zip((2, 4, 8), (C[0], C[2], C[3]))):
    ax2.plot(a0, spread[:, list(N).index(n)], color=c, label=fr"$N={n}$")
ax2.axvline(0.42, color="0.6", lw=0.7, ls=":")
ax2.text(0.46, 0.35, r"$a_0=0.42$", fontsize=7, color="0.35")
ax2.set_xlabel(r"$a_0$")
ax2.set_ylabel(r"$\Delta\Sigma^{\max}$ over CEP (deg)")
ax2.set_xlim(0, a0.max())
ax2.set_ylim(0, None)
ax2.legend(frameon=False, loc="upper left", handlelength=1.8)
ax2.text(0.03, 0.06, "(b)", transform=ax2.transAxes)

os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
out = os.path.join(HERE, "figs", "fig_param.pdf")
fig.savefig(out)
print("wrote", out)
