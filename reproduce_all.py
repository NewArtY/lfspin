#!/usr/bin/env python3
"""Regenerate every number, table and figure of the Letter and its Supplemental Material.

    python reproduce_all.py                  # full run (each PINN training takes ~1-1.5 h on CPU)
    python reproduce_all.py --skip-training  # reuse results/pinn_*.json if present
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PINN_CONFIGS = ("main", "ablation", "elliptical", "baseline")


def run(*args):
    print(f"\n=== {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-training", action="store_true")
    a = ap.parse_args()
    run("run_exact_checks.py")
    run("run_cep_scan.py")
    run("run_convergence.py")
    for cfg in PINN_CONFIGS:
        if a.skip_training and os.path.exists(os.path.join(HERE, "results", f"pinn_{cfg}.json")):
            print(f"skipping training ({cfg}): results/pinn_{cfg}.json exists")
        else:
            run("train_pinn.py", "--config", cfg)
    run("make_fig1.py")
    run("make_fig2.py")
    run("make_figS1.py")
    print("\nAll done: results/*.json, figs/fig1.pdf, figs/fig2.pdf, figs/figS1.pdf")


if __name__ == "__main__":
    main()
