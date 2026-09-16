# Changelog

All notable changes to this deposit. Versions follow the Zenodo releases: each
release has its own version DOI, and the concept DOI shown on the Zenodo record
always resolves to the latest one.

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
