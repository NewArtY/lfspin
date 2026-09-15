#!/usr/bin/env python3
"""Regenerate every number, table and figure of the Letter and its Supplemental Material.

    python reproduce_all.py                  # full run (the four PINN trainings take ~4.4 h on CPU)
    python reproduce_all.py --skip-training  # reuse results/pinn_*.json if present
    python reproduce_all.py --verify         # check the files against MANIFEST.sha256
    python reproduce_all.py --manifest       # rewrite MANIFEST.sha256 after an intentional change
"""
import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PINN_CONFIGS = ("main", "ablation", "elliptical", "baseline")

MANIFEST = "MANIFEST.sha256"
# Not part of the deposit: what the package writes on the fly (bytecode, the resumable Adam
# checkpoints of train_pinn.py) and what belongs to whoever runs it (virtual environment,
# IDE, version control, OS files). Without the second group a venv inside lfspin/ fails --verify.
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".venv", "venv", "env", ".git", ".idea",
             ".vscode", ".vs", ".claude", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_FILES = {MANIFEST, "Thumbs.db", "desktop.ini", ".DS_Store"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".ckpt", ".ckpt.tmp")


def run(*args):
    print(f"\n=== {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def deposit_files():
    """Deposited files as sorted paths relative to HERE, with forward slashes."""
    out = []
    for dirpath, dirnames, filenames in os.walk(HERE):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn not in SKIP_FILES and not fn.endswith(SKIP_SUFFIXES):
                out.append(os.path.relpath(os.path.join(dirpath, fn), HERE).replace(os.sep, "/"))
    return sorted(out)


def sha256(rel):
    h = hashlib.sha256()
    with open(os.path.join(HERE, rel), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest():
    files = deposit_files()
    with open(os.path.join(HERE, MANIFEST), "w", encoding="utf-8", newline="\n") as f:
        f.write("# sha256  size  path -- check with `python reproduce_all.py --verify`\n")
        for rel in files:
            f.write(f"{sha256(rel)}  {os.path.getsize(os.path.join(HERE, rel))}  {rel}\n")
    print(f"wrote {MANIFEST}: {len(files)} files")


def verify_manifest():
    path = os.path.join(HERE, MANIFEST)
    if not os.path.exists(path):
        sys.exit(f"{MANIFEST} not found; run --manifest first")
    want = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                digest, size, rel = line.rstrip("\n").split("  ", 2)
                want[rel] = (digest, int(size))
    have = set(deposit_files())
    missing = sorted(set(want) - have)
    extra = sorted(have - set(want))
    changed = sorted(rel for rel in set(want) & have
                     if os.path.getsize(os.path.join(HERE, rel)) != want[rel][1]
                     or sha256(rel) != want[rel][0])
    for tag, rels in (("CHANGED", changed), ("MISSING", missing), ("EXTRA", extra)):
        for rel in rels:
            print(f"  {tag:8s} {rel}")
    ok = not (changed or missing or extra)
    print(f"verify: {len(want)} files, {len(changed)} changed, {len(missing)} missing, "
          f"{len(extra)} extra -> {'OK' if ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--verify", action="store_true", help="check the files against MANIFEST.sha256")
    ap.add_argument("--manifest", action="store_true", help="rewrite MANIFEST.sha256")
    a = ap.parse_args()
    if a.manifest:
        write_manifest()
        return
    if a.verify:
        sys.exit(0 if verify_manifest() else 1)
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
    print("The deposited files in results/ and figs/ have been rewritten; `--verify` now reports "
          "every byte that differs from the deposit (the figure PDFs always differ in their "
          "creation date).")


if __name__ == "__main__":
    main()
