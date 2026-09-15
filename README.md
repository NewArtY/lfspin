# lfspin — light-front spin dynamics, exact benchmarks and a PINN solver

Code and data for the Letter

> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin,
> *Rapidity-Coupled Spin Dynamics in Pulsed Laser Fields from Physics-Informed Neural Networks*
> (submitted to Physical Review A).

**Release 1.0.0** · Zenodo version DOI: `10.5281/zenodo.XXXXXXXX` ·
Repository: <https://github.com/NewArtY/lfspin> ·
Release history: [`CHANGELOG.md`](CHANGELOG.md)

The package is self-contained. It needs only the Python packages listed in
`requirements.txt`, and a single command regenerates every number, table and
figure of the Letter and its Supplemental Material.

## Physics in one paragraph

An electron (charge −|e|, units c = m = |e| = 1) crosses a plane-wave laser
pulse `a_x(η) = a0 cos²-envelope(η) cos(η + φ0)`, with η = ω(t − z).

- **Orbit.** Because k·u is conserved for any pulse, the orbit is the
  algebraic Volkov solution.
- **Spin.** The spin obeys the covariant BMT equation. For linear polarization
  and an electron initially at rest, the spin four-vector is an explicit
  function of the instantaneous potential at any g (`physics.closed_form_spin_linear`).
- **Rest-frame polarization angle.** It is Σ = 2 arctan(a_x/2) + a_e a_x, so the
  net rotation after the pulse vanishes for every CEP, while the peak
  intra-pulse angle depends on the CEP through max|a_x|.
- **Elliptical polarization.** No closed form exists; the net rotation is the
  holonomy ½ a_e² |𝒜|.

The numerical solvers are:
- a light-front DOP853 reference,
- fixed-step lab-time RK4,
- Boris–BMT and Higuera–Cary–BMT pushers (Cayley spin update),
- light-front RK4,
- a physics-informed neural network (PINN) trained only on the equation
  residuals and invariants.

All of them are checked against these exact results. The PINN is also run for
an elliptically polarized pulse and in a controlled ablation of its
Fourier-feature input embedding.

## Contents

| File | Purpose |
|---|---|
| `physics.py` | Field (linear/elliptical/circular), Volkov orbit, BMT right-hand sides, DOP853 integrators (light-front and proper-time), closed forms |
| `convergence.py` | Fixed-step lab-time RK4, Boris–BMT and Higuera–Cary–BMT pushers, light-front RK4 |
| `pinn.py` | Fourier-feature PINN, residual loss, Adam (with resumable checkpoints) and L-BFGS training |
| `run_exact_checks.py` | Closed form vs DOP853, net-rotation CEP scan, proper-time cross-check, circular-polarization holonomy → `results/exact_checks.json` |
| `run_cep_scan.py` | 49-point CEP scan of the rest-frame angle Σ^max and γ_max for N = 2, 8 (physical g and g = 2) → `results/cep_scan.json` |
| `run_convergence.py` | Fixed-step-budget study, surfing electron γ0 = 10 → `results/convergence.json` |
| `train_pinn.py` | Four configurations → `results/pinn_<config>.json`, `results/pinn_<config>_model.pt` (see below) |
| `make_fig1.py`, `make_fig2.py`, `make_figS1.py` | Figures 1, 2 of the Letter and Fig. S1 → `figs/*.pdf`, `figs/*.png` |
| `reproduce_all.py` | Runs everything in order; checks the deposit against `MANIFEST.sha256` |
| `results/*.log` | Transcripts of the runs that produced the deposited JSON files |
| `MANIFEST.sha256` | SHA-256 digest and size of every deposited file |
| `CITATION.cff`, `.zenodo.json` | Citation and Zenodo metadata |
| `CHANGELOG.md` | Release history |

PINN configurations in `train_pinn.py`:

| Config | Setup |
|---|---|
| `main` | Linear polarization, Fourier features (k_max = 8), 5×128, Adam 32 000 + L-BFGS 3000 |
| `ablation` | As `main` but plain input η/T (k_max = 0) |
| `elliptical` | As `main`, elliptical polarization with ellipticity δ = 0.5 (no closed form) |
| `baseline` | Plain input, smaller 4×64 network, Adam 24 000, no L-BFGS |

## Reproduce

```bash
python -m pip install -r requirements.txt
python reproduce_all.py --verify         # check the files against MANIFEST.sha256 first
python reproduce_all.py                  # full run; the four PINN trainings take 17–87 min each on CPU
python reproduce_all.py --skip-training  # reuse the trained networks in results/
```

The ODE computations are deterministic. PINN training is seeded
(`torch.manual_seed(0)`, `numpy.random.seed(0)`) and can be resumed from
checkpoints. It is reproducible up to floating-point nondeterminism of
multithreaded BLAS, so a rerun on different hardware can change the PINN errors
in their last digits.

During training `train_pinn.py` writes a resumable checkpoint
`results/pinn_<config>_adam.ckpt`. It is not part of the deposit and is not
removed afterwards; while it exists, a new training run of that configuration
resumes from it, so delete it to train from scratch.

## Checking the deposit

```bash
python reproduce_all.py --verify     # check every deposited file against MANIFEST.sha256
python reproduce_all.py --manifest   # rewrite the manifest after an intentional change
```

`--verify` names every file that is changed, missing or unexpected and exits
nonzero if there is any. It leaves out what the package writes on the fly
(`__pycache__/`, `*.ckpt`) and what belongs to whoever runs it (a virtual
environment, an IDE directory, `.git/`). Git stores the files byte for byte
(`.gitattributes`), so a clone or a `git archive` of the release passes the
check on any operating system. Rerunning the computations rewrites `results/`
and `figs/`, after which `--verify` reports the differences; the figure PDFs
always differ, because they carry their creation date.

## Where the numbers of the paper come from

| Quantity in the paper | Source |
|---|---|
| Closed form vs DOP853 (≤ 10⁻¹²), Σ formula | `exact_checks.json`: `closed_form_max_dS_overall`, `closed_form_max_dSigma_overall` |
| Net rotation over 16 CEPs, N = 2 and 8 | `exact_checks.json`: `net_rotation_max_deg_N2`, `_N8` |
| Proper-time cross-check | `exact_checks.json`: `tau_crosscheck_*` |
| Circular-polarization holonomy | `exact_checks.json`: `circular_holonomy` |
| Σ^max spreads, suppression factor, a_max and γ_max ranges, energy spreads | `cep_scan.json`: `scan.N2.summary`, `scan.N8.summary`, `suppression_factor` |
| Table S1, Fig. 2(c) | `convergence.json` |
| PINN accuracy, invariants, Table S3 | `pinn_main.json` |
| Ablation, baseline, elliptical case (Fig. S1) | `pinn_ablation.json`, `pinn_baseline.json`, `pinn_elliptical.json` |

## Environment used for the published results

Python 3.14.3, numpy 2.4.2, scipy 1.18.1, matplotlib 3.11.1, torch 2.10.0+cpu (Windows 11, CPU only).

## License

MIT, see `LICENSE`.

## How to cite

Please cite the Letter and this deposit:

> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin,
> *lfspin: light-front spin dynamics, exact benchmarks and physics-informed
> neural network solver*, version 1.0.0, Zenodo (2026),
> doi:10.5281/zenodo.XXXXXXXX.

This is the version DOI of release 1.0.0: it points to exactly the code and
data that produced the numbers of the Letter. Zenodo also assigns a concept
DOI, shown on the record page, which always resolves to the latest release.
Machine-readable metadata are in `CITATION.cff`.
