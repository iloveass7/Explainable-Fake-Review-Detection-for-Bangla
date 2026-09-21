#!/usr/bin/env python3
"""Execute the six-stage Kaggle pipeline in order."""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "01-data-acquisition-audit.ipynb",
    "02-preprocessing-eda.ipynb", 
    "03-feature-engineering.ipynb",
    "04-gold-only-training.ipynb",
    "05-pseudo-labeling.ipynb",
    "06-final-evaluation.ipynb",
    "07-decision-support-sql-pyspark.ipynb"
]

def run_notebook(nb_path: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--execute", "--to", "notebook",
        "--output", nb_path.name,
        "--output-dir", str(output_dir),
        "--ExecutePreprocessor.timeout=-1",
        str(nb_path),
    ]
    print(f"Running {nb_path.name}", flush=True)
    subprocess.run(cmd, cwd=nb_path.parent, check=True)

def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "executed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path("/kaggle/working").exists():
        raise RuntimeError("This runner is configured for Kaggle only")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(base_dir / "requirements.txt")],
        cwd=base_dir,
        check=True,
    )

    notebooks_dir = base_dir / "notebooks" if (base_dir / "notebooks").exists() else base_dir
    for nb in NOTEBOOKS:
        nb_path = notebooks_dir / nb
        if not nb_path.exists():
            nb_path = base_dir / nb
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb}")
        run_notebook(nb_path, output_dir)

    workspace = Path("/kaggle/working/bangla_fake_review")
    print("Pipeline completed")
    print(f"Results: {workspace / 'results'}")
    print(f"Models: {workspace / 'models'}")
    print(f"Report: {workspace / 'report'}")

if __name__ == "__main__":
    main()
