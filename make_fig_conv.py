#!/usr/bin/env python3
"""Fixed-step convergence comparison as a standalone figure
(reads results/convergence.json).

In the Letter version this was one panel of a combined figure; for the article
it stands on its own, so the legend and the tick labels can be read."""
import json
import os

import numpy as np
import matplotlib
import matplotlib.ticker
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
C_BLUE, C_GREEN, C_VERMIL, C_PURPLE = "#0072B2", "#009E73", "#D55E00", "#CC79A7"

with open(os.path.join(HERE, "results", "convergence.json")) as f:
    conv = json.load(f)
n = np.array(conv["n_steps"])

fig, ax = plt.subplots(figsize=(3.4, 3.0))
fig.subplots_adjust(left=0.20, right=0.97, top=0.82, bottom=0.16)
ax.loglog(n, conv["boris"], color=C_VERMIL, marker="^", ms=3.4, lw=1.1, label="Boris--BMT")
ax.loglog(n, conv["hc"], color=C_PURPLE, marker="v", ms=3.4, lw=1.1, ls="--",
          label="Higuera--Cary--BMT")
ax.loglog(n, conv["rk4"], color=C_BLUE, marker="o", ms=3.2, lw=1.1, label="lab-time RK4")
ax.loglog(n, conv["lightfront"], color=C_GREEN, marker="s", ms=3.2, lw=1.1,
          label="light-front RK4")
ax.set_xlabel(r"number of steps $N_{\rm steps}$")
ax.set_ylabel(r"max relative error at $\eta=2\pi N$")
ax.set_ylim(1e-15, 1e1)
ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=7))
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
          handlelength=1.6, labelspacing=0.2, columnspacing=1.0, fontsize=6.6,
          borderaxespad=0.2)

os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
out = os.path.join(HERE, "figs", "fig_conv.pdf")
fig.savefig(out)
print("wrote", out)
