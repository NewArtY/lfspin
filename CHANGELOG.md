# Changelog

All notable changes to this deposit. Versions follow the Zenodo releases: each
release has its own version DOI, and the concept DOI shown on the Zenodo record
always resolves to the latest one.

## 1.1.0 — 2026-09-23

Adds the computations of the expanded article. The Letter-version results are
unchanged except for the CEP scan, whose grid was refined (see below).

- `run_param_scan.py`: map of the CEP sensitivity over amplitude and pulse
  length. Because `a_max = a0 * m(N, phi)`, the amplitude factors out and the
  relative CEP spread of the peak potential and of the peak energy depends on
  the number of cycles alone; the map is built from the closed form and
  validated pointwise against the reference integrator.
- `run_ellipticity_scan.py`: net post-pulse rotation from linear to circular
  polarization, against the leading-order holonomy, with the CEP independence
  of that rotation and the amplitude scaling of the residual.
- `run_pinn_invariants.py`: `max |S.u|` of the trained networks, recomputed
  from the checkpoints. The loss never refers to this constraint, so it is an
  independent check; the script also re-derives the stored accuracy figures to
  confirm that the architecture is reconstructed correctly.
- New figures: `make_fig_param.py`, `make_fig_ellipticity.py`, and
  `make_fig_conv.py` / `make_fig_pinn.py`, which split the combined figure of
  the Letter into a convergence figure and a network figure.
- `run_cep_scan.py` now evaluates the trajectories on 20 000 points per pulse
  instead of 2500. The coarser grid did not resolve the maximum of |Sigma| for
  eight-cycle pulses, which biased the derived numbers in their third digit:
  the suppression factor is 13.74 rather than 13.82 and the peak-energy spread
  at N = 8 is 1.909% rather than 1.899%. The values now agree with the closed
  form to the digits quoted. The Letter version, archived in release 1.0.1 and
  on arXiv, quotes the earlier numbers.
- `reproduce_all.py` runs the new stages; `MANIFEST.sha256` regenerated.

## 1.0.1 — 2026-09-16

Fixes the integrity check of the deposit. No computed result changes.

- `results/*.json` and `results/*.log` are stored with LF line endings. In 1.0.0
  they were CRLF in the working tree but LF in git, so `MANIFEST.sha256`,
  computed from the working tree, did not match a fresh clone or the Zenodo
  archive and `reproduce_all.py --verify` reported 14 changed files.
- `MANIFEST.sha256` regenerated from the LF files; `--verify` now passes on the
  working tree, on a fresh clone and on the downloaded Zenodo archive.
- The scripts write their JSON results with LF on every platform, so a rerun
  reproduces the deposited bytes.
- The Zenodo DOI replaces the placeholders in `README.md` and `CITATION.cff`
  (concept DOI 10.5281/zenodo.22779264, which always resolves to the latest
  release).

## 1.0.0 — 2026-09-16

First release, deposited with the Letter

> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin,
> *Rapidity-Coupled Spin Dynamics in Pulsed Laser Fields from Physics-Informed
> Neural Networks*, submitted to Physical Review A.

Contains the exact checks, the carrier-envelope-phase scan, the fixed-step-budget
convergence study, the four PINN training configurations, the figure scripts,
and the computed results, run transcripts and trained networks behind every
number of the Letter and its Supplemental Material. `MANIFEST.sha256` lists
the checksums of all deposited files.
