#!/usr/bin/env python3
"""Execute the six-stage Kaggle pipeline in order."""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "01_data_acquisition_audit.ipynb",
    "02_preprocessing_eda.ipynb", 
    "03_feature_engineering.ipynb",
    "04_gold_only_training.ipynb",
    "05_pseudo_labeling.ipynb",
    "06_final_evaluation.ipynb"
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

    for nb in NOTEBOOKS:
        nb_path = base_dir / nb
        if not nb_path.exists():
            raise FileNotFoundError(nb_path)
        run_notebook(nb_path, output_dir)

    workspace = Path("/kaggle/working/bangla_fake_review")
    print("Pipeline completed")
    print(f"Results: {workspace / 'results'}")
    print(f"Models: {workspace / 'models'}")
    print(f"Report: {workspace / 'report'}")

if __name__ == "__main__":
    main()
